from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts.prompt import PromptTemplate
from constants.prompt_templates import UNCOMMON_QUESTION_WORKFLOW_TEMPLATE
from constants.db_constants import DATABASE_SCHEMA

import os

class LangChainClient:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=os.getenv('NEO4J_URI'), username=os.getenv('NEO4J_USER'), password=os.getenv('NEO4J_PASSWORD')
        )
    
    def run_template_generation(self, user_input, context_text):
        self.graph.refresh_schema()
        CYPHER_GENERATION_PROMPT = PromptTemplate(
            input_variables=["question", "context", "schema"], template=UNCOMMON_QUESTION_WORKFLOW_TEMPLATE
        )

        chain = GraphCypherQAChain.from_llm(
            ChatOpenAI(temperature=0, model_name="gpt-4"),
            graph=self.graph,
            verbose=True,
            allow_dangerous_requests=True,
            cypher_prompt=CYPHER_GENERATION_PROMPT,
            return_intermediate_steps=True
        )

        user_question = {
            "query": UNCOMMON_QUESTION_WORKFLOW_TEMPLATE,
            "question": user_input,    # The user's question
            "context": context_text,   # Initial or placeholder query if applicable
            "schema": DATABASE_SCHEMA  # Provide the schema here if defined elsewhere
        }

        result = chain.invoke(user_question)
        print(f"LangChain Cypher query steps: {result['intermediate_steps']}")
        return result['intermediate_steps']