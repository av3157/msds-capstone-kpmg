# Imports
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from constants.prompt_templates import UNCOMMON_QUESTION_WORKFLOW_TEMPLATE_ALT
from constants.db_constants import DATABASE_SCHEMA

import os

class LangChainClient:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=os.getenv('NEO4J_URI'), username=os.getenv('NEO4J_USER'), password=os.getenv('NEO4J_PASSWORD')
        )

    # This is the method that uses the LLM to generate and execute a Cypher query
    # The LLM is provided the users's question, the context from embedding graph obtained through ReitrevalQA, 
    # and the graph schema to generate and execute a Cypher query
    # Self-reflection is used to refine the Cypher query if it fails to execute
    def run_template_generation(self, user_input, context_text, max_attempts=5):
        self.graph.refresh_schema()
        
        CYPHER_GENERATION_PROMPT = PromptTemplate(
            input_variables=["query", "context", "schema"], 
            template=UNCOMMON_QUESTION_WORKFLOW_TEMPLATE_ALT
        )

        chain = GraphCypherQAChain.from_llm(
            ChatOpenAI(temperature=0, model_name="gpt-4o"),
            graph=self.graph,
            verbose=True,
            allow_dangerous_requests=True,
            prompt=CYPHER_GENERATION_PROMPT,
            return_intermediate_steps=True
        )

        user_question = {
            "query": user_input,       # The user's question
            "context": context_text,   # context from RetrievalQA
            "schema": DATABASE_SCHEMA  # database schema
        }

        attempts = 0
        result = None

        # Self-reflection with a max of 5 attempts
        while attempts < max_attempts:
            print(f"\n Attempt {attempts + 1} generating Cypher query...")
            
            result = chain.invoke(user_question)
            print(f"LangChain Cypher query steps: {result['intermediate_steps']}")

            query = result['intermediate_steps'][0]
            answer = result['intermediate_steps'][1]
            if not answer["context"]:
                print("Query failed to retrieve data. Refining query...")
                user_question['query'] += f" This was your previous query: {query['query']}. Make sure the new Cypher Query is different from this, especially the MATCH clause."
                attempts += 1
            else: 
                attempts = max_attempts

        return result['intermediate_steps'] 
