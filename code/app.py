import streamlit as st
from clients.neo4j_client import Neo4jClient
from clients.openai_client import OpenAiClient
from clients.langchain_client import LangChainClient
from components.intent_matching import get_input_parameter, get_request_intent
from components.extract_node_info import match_node
from constants.prompt_templates import USER_RESPONSE_TEMPLATE, INTENT_MATCHING_TEMPLATE
from constants.chatbot_responses import CHATBOT_INTRO_MESSAGE, FAILED_INTENT_MATCH, CYPHER_QUERY_ERROR, NOT_RELEVANT_USER_REQUEST, NO_RESULTS_FOUND
from constants.db_constants import DATABASE_SCHEMA
from constants.query_templates import query_map
from components.parameter_correction import ParameterCorrection
from components.create_embeddings import create_embeddings
from gui.graph_test import fetch_graph_data
import logging
import os
import re
from streamlit_agraph import agraph, Node, Edge, Config
from streamlit_image_zoom import image_zoom
from PIL import Image

from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from constants.prompt_templates import UNCOMMON_QUESTION_WORKFLOW_TEMPLATE
from langchain.prompts.prompt import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
load_dotenv()

# RAG Chatbot Orchestrator
#     1. Intent matching to determine if user request is a common or uncommon
#         - If its common, we use the extracted input parameter, update the expected Cypher query, and directly call Neo4j
#         - If its uncommon, we call GraphCypherQAChain with some example Cypher queries to generate a Cypher query
#               -If its irrelevant, we let the user know that we don't support their request
#               - If the correct cypher query is unable to be generated, perform one hop
#     2. For common and uncommon Cypher query results, we pass the user request and query result to a LLM to generate the final response

def rag_chatbot(user_input):
    embeddings_graphs = create_embeddings()
    print("---------------------------------")
    print(f"User request: {user_input}")
    openai = OpenAiClient()
    error_occurred = False

    # Get user request intent
    get_request_intent_response = get_request_intent(user_input, openai)
    intent_type = get_request_intent_response[0]
    cypher_query_response = {}

    # Irrelevant user request
    node_info = match_node(user_input) 
    if node_info is None:
        return NOT_RELEVANT_USER_REQUEST
    
    # We call LangChain+LLM to generate a Cypher query for uncommon questions 
    if intent_type == "UNCOMMON":
        uncommon_query_response = execute_uncommon_query(user_input, embeddings_graphs)
        cypher_query_response, error_occurred = uncommon_query_response['cypher_query_response'], uncommon_query_response['error_occurred']
    elif intent_type == "COMMON":
        if len(get_request_intent_response) > 1:
            question_id = int(get_request_intent_response[1])
            common_query_response = execute_common_query(openai, user_input, question_id)
            cypher_query_response, error_occurred = common_query_response['cypher_query_response'], common_query_response['error_occurred']
            # Get question_id and parameter for agraph
            question_id, parameter_for_agraph= common_query_response['question_id'], common_query_response['parameter_for_agraph']
            # Create agraph
            if parameter_for_agraph != '':
                if question_id in [1,3,4,6]:
                    nodes, edges = fetch_graph_data(question_id, parameter_for_agraph)
                    if nodes and edges:
                        config = Config(width=700, 
                                        height=300, 
                                        directed=True, 
                                        nodeHighlightBehavior=True, 
                                        hierarchical=True, 
                                        staticGraphWithDragAndDrop=True,
                                        physics={
                                            "enabled": True
                                        },
                                        layout={"hierarchical":{
                                            "levelSeparation": 180,
                                            "nodeSpacing": 150,
                                            "sortMethod": 'directed'
                                        }}
                                        
                                )
                        
                        agraph(nodes=nodes, edges=edges, config=config)
                        
                    else:
                        st.write("No nodes to display.")



    else:
        return FAILED_INTENT_MATCH

    # Final response generation
    if error_occurred:
        return cypher_query_response
    if len(cypher_query_response) == 0 or (len(cypher_query_response) > 1 and "context" in cypher_query_response[1] and len(cypher_query_response[1]["context"]) == 0):
        return NO_RESULTS_FOUND

    response = generate_final_output(openai, user_input, cypher_query_response)
    return response

def execute_uncommon_query(user_input, embeddings):
    langchain_client = LangChainClient()
    error_occurred = False
    node_info = match_node(user_input)
    embed_graph = node_info + "_embedding_graph"
    print(f"Retrieving information from the {embed_graph}.")
    print("UNCOMMON QUERY")
    try:
        vector_qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0, model_name="gpt-4"),
            chain_type="stuff", #examples are stuff, map_reduce, refine, map_rerank
            retriever=embeddings[embed_graph].as_retriever(search_type='similarity', k=15)) 
        cypher_query_documents = vector_qa.retriever.get_relevant_documents(user_input)
        
        context_text = "\n".join([doc.page_content for doc in cypher_query_documents])
        print(f"Retrieved Context: {context_text}")

        cypher_query_response = langchain_client.run_template_generation(user_input, context_text)

        # Step 4 from workflow
        # If Cypher Query returns no context ([]), then do one hop:
        # Retrieve relevant nodes 
        answer = cypher_query_response[1]
        if not answer["context"]:
            n = 2 # number of nodes for context
            cypher_n_docs = cypher_query_documents[0:n]
            first_n_docs = "\n".join([doc.page_content for doc in cypher_n_docs])
            node_names = re.findall(r"name:\s*(.*)", first_n_docs)
            print(f"FIRST N DOCS: {first_n_docs}")
            print(f"NODE NAME: {node_names}")

            neo4j = Neo4jClient()
            cypher_query_response = ""
            for i in range(n):
                cypher_query_response += f"\n---------NODE: {node_names[i]}---------\n"
                cypher_query = f"""
                    MATCH (n)-[r]->(m)  // Outgoing relationships
                    WHERE n.name = "{node_names[i]}"
                    RETURN n AS source_node, r AS relationship, m AS connected_node
                    UNION
                    MATCH (n)<-[r]-(m)  // Incoming relationships
                    WHERE n.name = "{node_names[i]}"
                    RETURN n AS source_node, r AS relationship, m AS connected_node            
                """
                
                cypher_query_response += neo4j.execute_query_one_hop(cypher_query)
            print("CYPHER QUERY EXECUTED SUCCESSFULLY!")

        #Parameter Correction - if necessary
        if len(cypher_query_documents) == 0:
            print("NOTE: No data was found from LangChain call, trying parameter correction\n")
            input_corrector = ParameterCorrection()
            updated_user_input = input_corrector.generate_response(user_input, '')
            print(f"Retrying LangChain with corrected user input: [{updated_user_input[1]}]")
            cypher_query_response = vector_qa.run(updated_user_input[1])
        
        print("RETRIEVAL RESPONSE:", cypher_query_response)
    
    except Exception as e:
        print(f"ERROR: {e}")
        cypher_query_response = CYPHER_QUERY_ERROR
        error_occurred = True
    return { 'cypher_query_response': cypher_query_response, 'error_occurred': error_occurred}

def execute_common_query(openai, user_input, question_id):
    # Obtain the question ID and extract input parameter
    neo4j = Neo4jClient()
    error_occurred = False
    input_parameter_response = get_input_parameter(user_input, openai)
    print(input_parameter_response)
    extracted_input_parameter, input_parameter_type = input_parameter_response[0], input_parameter_response[1]
    # agraph path variable
    parameter_for_agraph = extracted_input_parameter
    
    print(f"COMMON QUERY: [{question_id}|{extracted_input_parameter}|{input_parameter_type}]")
    cypher_query = neo4j.generate_common_cypher_query(question_id, extracted_input_parameter)
    try:
        # Execute the query
        cypher_query_response = neo4j.execute_query(cypher_query)
        print(f"Neo4j cypher query result: {cypher_query_response}")

        # If query execution fails, attempt to correct input parameter
        if len(cypher_query_response) == 0:
            print(f"NOTE: Common query execution failed, trying parameter correction")
            input_corrector = ParameterCorrection()
            corrected_input_response = input_corrector.generate_response(user_input, input_parameter_type)
            corrected_input_parameter, corrected_input = corrected_input_response[0], corrected_input_response[1]
            corrected_cypher_query = neo4j.generate_common_cypher_query(question_id, corrected_input_parameter)
            cypher_query_response = neo4j.execute_query(corrected_cypher_query)
            parameter_for_agraph = corrected_input_parameter
            # If corrected query fails, we call LangChain
            if len(cypher_query_response) == 0:
                print(f"NOTE: Common query execution failed AFTER correction, trying LangChain")
                langchain_client = LangChainClient()
                cypher_query_response = langchain_client.run_template_generation(corrected_input, "") # added "" as context for now
                parameter_for_agraph = ''

    except Exception as e:
        print(f"Error executing query in Neo4j: {e}")
        cypher_query_response = "An error occurred while executing the query. Please try again."
        error_occurred = True

    return { 'cypher_query_response': cypher_query_response, 'error_occurred': error_occurred, 
            'question_id': question_id, 'parameter_for_agraph': parameter_for_agraph}


# Given the user question and data, calling LLM to create chatbot's final response
def generate_final_output(openai, user_input, cypher_query_response):
    chatbot_response_template = USER_RESPONSE_TEMPLATE.format(query=user_input, cypher_query_response=cypher_query_response)
    response = openai.generate(chatbot_response_template)
    return response

# Setup StreamLit app
def main():
    # Sidebar
    image_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'visualization2.png')
    # Streamlit image_zoom
    # image = Image.open(image_path)
    
    with st.sidebar:
        st.image(image_path, caption='Database Schema', use_column_width="always")
        # image_zoom(image, mode="scroll", size=(500, 700), keep_aspect_ratio=False, zoom_factor=4.0, increment=0.2)
        # st.markdown(f'<img src="{image_path}" style="{style_image1}">',
        #             unsafe_allow_html=True)
    st.title("Model Metadata RAG Chatbot")
    

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": CHATBOT_INTRO_MESSAGE}]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Please enter your request here"):

        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Call RAG chatbot
        logging.info("Started request execution")
        response = rag_chatbot(prompt)
        logging.info("Finished request execution")

       
        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        
if __name__ == '__main__':
    main()