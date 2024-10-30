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
    
    def run_template_generation(self, user_input):
        self.graph.refresh_schema()
        CYPHER_GENERATION_PROMPT = PromptTemplate(
            input_variables=["question"], template=UNCOMMON_QUESTION_WORKFLOW_TEMPLATE
        )
        #CYPHER_GENERATION_PROMPT = CYPHER_GENERATION_PROMPT.format(schema=DATABASE_SCHEMA, question=user_input)
        #print(CYPHER_GENERATION_PROMPT)

        chain = GraphCypherQAChain.from_llm(
            cypher_llm = ChatOpenAI(temperature=0, model_name="gpt-4"),
            qa_llm = ChatOpenAI(temperature=0), 
            graph=self.graph,
            verbose=True,
            allow_dangerous_requests=True,
            cypher_prompt=CYPHER_GENERATION_PROMPT,
            return_intermediate_steps=True
        )

        result = chain.invoke(user_input)
        print(f"LangChain Cypher query steps: {result['intermediate_steps']}")
        return result['intermediate_steps']