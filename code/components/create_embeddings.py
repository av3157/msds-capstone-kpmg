import os
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

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
    "embedding_node_property": "Model_embedding"
    }

    ModelVersion = {
    "index_name": "ModelVersion_idx",
    "node_label": "ModelVersion",
    "text_node_properties": ["metadata", "latest_version", "performance_metrics", "name", 
                            "model_parameters", "top_features", "model_id", "version"],
    "embedding_node_property": "ModelVersion_embedding"
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