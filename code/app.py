import streamlit as st
from clients.neo4j_client import Neo4jClient
from clients.openai_client import OpenAiClient
from clients.langchain_client import LangChainClient
from components.new_intent_matching import get_input_parameter, get_request_intent
from components.extract_node_info import match_node
from constants.prompt_templates import USER_RESPONSE_TEMPLATE, INTENT_MATCHING_TEMPLATE
from constants.chatbot_responses import CHATBOT_INTRO_MESSAGE, FAILED_INTENT_MATCH, CYPHER_QUERY_ERROR, NOT_RELEVANT_USER_REQUEST, NO_RESULTS_FOUND
from constants.db_constants import DATABASE_SCHEMA
from constants.query_templates import query_map
from components.parameter_correction import ParameterCorrection
from gui.graph_test import fetch_graph_data
import logging
import os
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

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# RAG Chatbot Orchestrator
#     1. Intent matching to determine if user request is a common, uncommon, or irrelevant question
#         - If its common, we use the extracted input parameter, update the expected Cypher query, and directly call Neo4j
#         - If its uncommon, we call GraphCypherQAChain with some example Cypher queries to generate a Cypher query
#         - If its irrelevant, we let the user know that we don't support their request
#     2. For common and uncommon Cypher query results, we pass the user request and query result to a LLM to generate the final response
def create_embeddings():
    uri = os.getenv('NEO4J_URI')
    username = os.getenv('NEO4J_USER')
    password = os.getenv('NEO4J_PASSWORD')

    BusinessGroup = { "index_name": "BusinessGroup_idx",
                    "node_label": "BusinessGroup",
                    "text_node_properties": ["name"],
                    "embedding_node_property": "BusinessGroup_embedding"
                    }
    Column = {
    "index_name": "Column_idx",
    "node_label": "Column",
    "text_node_properties": ["name", "type"],
    "embedding_node_property": "Column_embedding"
    }

    Contact = {
    "index_name": "Contact_idx",
    "node_label": "Contact",
    "text_node_properties": ["name", "type"],
    "embedding_node_property": "Contact_embedding"
    }

    Database = {
    "index_name": "Database_idx",
    "node_label": "Database",
    "text_node_properties": ["name", "type"],
    "embedding_node_property": "Database_embedding"
    }

    DataElement = {
    "index_name": "DataElement_idx",
    "node_label": "DataElement",
    "text_node_properties": ["name", "source", "generatedForm"],
    "embedding_node_property": "DataElement_embedding"
    }

    Model = {
    "index_name": "Model_idx",
    "node_label": "Model",
    "text_node_properties": ["move_id", "name", "model_metadata"],
    "embedding_node_property": "DataElement_embedding"
    }

    ModelVersion = {
    "index_name": "ModelVersion_idx",
    "node_label": "ModelVersion",
    "text_node_properties": ["metadata", "latest_version", "performance_metrics", "name", 
                            "model_parameters", "top_features", "model_id", "version"],
    "embedding_node_property": "DataElement_embedding"
    }

    Report = {
    "index_name": "Report_idx",
    "node_label": "Report",
    "text_node_properties": ["name"],
    "embedding_node_property": "Report_embedding"
    }

    ReportField = {
    "index_name": "ReportField_idx",
    "node_label": "ReportField",
    "text_node_properties": ["name", "id"],
    "embedding_node_property": "ReportField_embedding"
    }

    ReportSection = {
    "index_name": "ReportSection_idx",
    "node_label": "ReportSection",
    "text_node_properties": ["name"],
    "embedding_node_property": "ReportSection_embedding"
    }

    Table = {
    "index_name": "Table_idx",
    "node_label": "Table",
    "text_node_properties": ["name"],
    "embedding_node_property": "Table_embedding"
    }

    User = {
    "index_name": "User_idx",
    "node_label": "User",
    "text_node_properties": ["name", "account"], #omitted entitlement for now
    "embedding_node_property": "User_embedding"
    }

    parent = [BusinessGroup, Column, Contact, Database, 
            DataElement, Model, ModelVersion, Report, 
            ReportField, ReportSection, Table, User]

    graphs = {}

    for i in range(len(parent)):
    # Create the vectorstore for our existing graph
        val = parent[i]["node_label"]
        graphs[f"{val}_embedding_graph"] = Neo4jVector.from_existing_graph(
            embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
            url=uri,
            username=username,
            password=password,
            index_name=parent[i]["index_name"],
            node_label=parent[i]["node_label"],
            text_node_properties=parent[i]["text_node_properties"],
            embedding_node_property=parent[i]["embedding_node_property"],
        )
    return graphs 

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
    if intent_type == "NONE":
        #return NOT_RELEVANT_USER_REQUEST
        intent_type = "UNCOMMON"
    
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
    print(f"Retrieving information from the {node_info}.")
    print("UNCOMMON QUERY")
    try:
        vector_qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0, model_name="gpt-4"),
            chain_type="stuff", #examples are stuff, map_reduce, refine, map_rerank
            retriever=embeddings[node_info].as_retriever(search_type='similarity', k=15)) 
        cypher_query_documents = vector_qa.retriever.get_relevant_documents(user_input)

        context_text = "\n".join([doc.page_content for doc in cypher_query_documents])
        cypher_query_response = langchain_client.run_template_generation(user_input, context_text)

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