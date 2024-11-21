INTENT_MATCHING_TEMPLATE = """
    Task: 
    Step 1: Match the user request intent to one of the following "common questions" and return the question number.
    
    Step 2: If it doesn't match any of the following 8 questions, return [UNCOMMON,0]

    Make sure ONLY return [COMMON,Integer] or [UNCOMMON,0]!!!!!

    Common Questions:
    - 1. What report fields are downstream of a specific column?
    - 2. What are the performance metrics of a specific model?
    - 3. What data is upstream to a specific report field?
    - 4. How many nodes upstream is the datasource for a specific report field?
    - 5. How was this report field calculated?
    - 6. What is the difference between the latest version and the previous version of a specific model?
    - 7. What are the top features of a specific model?
    - 8. Tell me about the latest version of a specific model?
    
    Some examples of uncommon Questions:
    - 0. How many report fields are there?
    - 0. What is the database type of a specific database?
    - 0. What are the columns in a specific table of a database?
    - 0. Which model versions have an accuracy metric above 85%?
    - 0. What are the parameters of a specific model version?
    - 0. What are the column data types in a specific table in a database?
    - 0. Which users have access to a specific report?
    - 0. Who maintains a specific report?
    - 0. Who is the owner of a specific report?
    
    Example:
    - Question: What is fastest animal in the world?
    - Answer: [UNCOMMON,0]

    Example:
    - Question: What are the SARIMA model parameters in Inventory Management Model Version 1?
    - Answer: [UNCOMMON,0]

    Example:
    - Question: Which business group is linked to the Employee Productivity Report?
    - Answer: [UNCOMMON,0]

    Example:
    - Question: What are the performance metrics of Customer Satisfaction Prediction Model?
    - Answer: [COMMON,2]

    Example:
    - Question: What data is upstream to a Top Performing Regions report field?
    - Answer: [COMMON,3]

    Example:
    - Question: What are the top features of a Customer Satisfaction Prediction Model?
    - Answer: [COMMON,7]

    Example:
    - Question: Tell me about the latest version of the Customer Satisfaction Prediction Model?
    - Answer: [COMMON,8]

    Schema:
    Node properties are the following:
    Database {{name: STRING, type: STRING}},Table {{name: STRING, primary_key: STRING}},Column {{name: STRING, type: STRING}},BusinessGroup {{name: STRING}},Contact{{name: STRING, email: STRING}},User {{name: STRING, account: STRING, entitlement: LIST, role: STRING}},Report {{name: STRING}},ReportSection {{name: STRING}},ReportField {{id: STRING, name: STRING}},DataElement {{name: STRING, source: STRING, generatedFrom: STRING}},ModelVersion {{name: STRING, version: INTEGER, latest_version: STRING, metadata: STRING, model_parameters: STRING, top_features: STRING, performance_metrics: STRING, model_id: STRING}},Model {{name: STRING, model_metadata: STRING, move_id: STRING}}
    The relationships are the following:
    (:Database)-[:CONTAINS]->(:Table),(:Database)-[:ASSOCIATED_WITH]->(:BusinessGroup),(:Table)-[:HAS_COLUMN]->(:Column),(:Table)-[:HAS_PRIMARY_KEY]->(:Column),(:Column)-[:TRANSFORMS]->(:DataElement),(:Contact)-[:CONTACT_OF]->(:BusinessGroup),(:User)-[:ENTITLED_ON]->(:Database),(:User)-[:ENTITLED_ON]->(:Report),(:User)-[:OWNS]->(:Report),(:User)-[:OWNS]->(:ModelVersion),(:User)-[:MAINTAINS]->(:Report),(:User)-[:MAINTAINS]->(:ModelVersion),(:Report)-[:ASSOCIATED_WITH]->(:BusinessGroup),(:ReportSection)-[:PART_OF]->(:Report),(:ReportField)-[:BELONGS_TO]->(:ReportSection),(:DataElement)-[:FEEDS]->(:ReportField),(:DataElement)-[:INPUT_TO]->(:ModelVersion),(:ModelVersion)-[:PRODUCES]->(:DataElement),(:Model)-[:VERSION_OF]->(:ModelVersion),(:Model)-[:LATEST_VERSION]->(:ModelVersion)
    
    User input is:
    {question}
"""


INPUT_PARAMETER_EXTRACTION_TEMPLATE = """
    Task: Given a Neo4j schema and a question, extract the single parameter from the question and return it within square brackets []
    Only return the input parameter and its type within the square brackets

    Schema:
    {schema}
    
    User input is:
    {question}

     Example:
    - Question: What report fields are downstream of the FeedbackComments column?
    - Return [FeedbackComments,Column]

    Example:
    - Question: What are the performance metrics of Customer Satisfaction Prediction Model?
    - Return [Customer Satisfaction Prediction Model,Model]

    Example:
    - Question: What data is upstream to the Sales Confidence Interval report field?
    - Return [Sales Confidence Interval,ReportField]

    Example:
    - Question: How many nodes upstream is the datasource for Training Hours report field?
    - Return [Training Hours,ReportField]

    Example:
    - Question: How was the Sales Confidence Interval report field calculated?
    - Return [Sales Confidence Interval,ReportField]

    Example:
    - Question: What is the difference between the latest version and the previous version of the Employee Productivity Prediction Model?
    - Return [Employee Productivity Prediction Model,Model]

    Example:
    - Question: What are the top features of a Customer Satisfaction Prediction Model?
    - Return: [Customer Satisfaction Prediction Model,Model]

    Example:
    - Question: Tell me about the latest version of the Sales Performance Prediction Model?
    - Return [Sales Performance Prediction Model,Model]

    Clarification of task: If a question contains both a report field parameter and a report parameter, only return the report field parameter.  Here are a couple of examples:
    
    Example:
    - Question: Which data sources are upstream to the Predicted Demand for Products field in the Inventory Management Report?
    - Return [Predicted Demand for Products,ReportField]

    Example:
    - Question: Which data elements feed into the Average Productivity by Department field in the Employee Productivity Report?
    - Return [Average Productivity by Department,ReportField]

"""


UNCOMMON_QUESTION_WORKFLOW_TEMPLATE = """
    I have provided you with relationships that nodes share based on a graph schema.
    I want you to generate and return a Cypher Query that answers the user question: {query}, with additional information from the retrieved context: {context}, and graph schema: {schema}.

    1. Databases contain Tables.
    2. Tables have Columns.
    3. Databases are associated with Business Groups.
    4. Tables have Columns and Tables have primary key in Column.
    5. Column transforms into Data Elements.
    6. Contacts are contacts of Business Groups.
    7. Users can:
        1. Be entitled to Databases and Reports.
        2. Own Reports and Model Versions.
        3. Maintain Reports and Model Versions
    8. Reports are associated with Business Groups.
    9. Reports have Report Sections.
    10. Report Sections have Report Fields.
    11. Data Elements feed Report Fields.
    12. Report Fields serve as inputs to Model Versions.
    13. Model Versions produce Data Elements.
    14. Model Versions are versions of Models and belong to Models.
    15. Model's latest version is Model Version.
"""


UNCOMMON_QUESTION_WORKFLOW_TEMPLATE_ALT = """
    Data is stored in a Neo4j graph database. Here are all of the node types:
    Node Types:     BusinessGroup, Column, Contact, Database,
                    DataElement, Model, ModelVersion, Report,
                    ReportField, ReportSection, Table, User

    The node types have the following attributes:
    Attributes:     BusinessGroup: <id>, BusinessGroup_embedding, name
                    Column: <id>, Column_embedding, name, type
                    Contact: <id>, Contact_embedding, name, email
                    Database: <id>, Database_emmbedding, name, type
                    DataElement: <id>, name, source, generatedFrom, DataElement_embedding
                    Model: <id>, move_id, name, model_metadata, Model_embedding
                    ModelVersion: <id>, metadata, latest_version, performance_metrics, name, model_parameters, top_features, model_id, version, ModelVersion_embedding
                    Report: <id>, Report_embedding, name
                    ReportField: <id>, ReportField_embedding, name, id
                    ReportSection: <id>, name, ReportSection_embedding
                    Table: <id>, name, Table_embedding
                    User: <id>, name, User_embedding, entitlement, account

    The node types have the following relationships with each other:
    Relationships:  Report ASSOCIATED_WITH BusinessGroup,
                    Database ASSOCIATED_WITH BusinessGroup,
                    ReportField BELONGS_TO ReportSection,
                    Contact CONTACT_OF BusinessGroup,
                    Database CONTAINS Table,
                    User ENTITLED_ON Database,
                    User ENTITLED_ON Report,
                    DataElement FEEDS ReportField,
                    Table HAS_COLUMN Column,
                    Table HAS_PRIMARY_KEY Column,
                    DataElement INPUT_TO ModelVersion,
                    Model LATEST_VERSION ModelVersion,
                    User MAINTAINS Report,
                    User MAINTAINS ModelVersion,
                    User OWNS ModelVersion,
                    User OWNS Report
                    ReportSection PART_OF Report,
                    ModelVersion PRODUCES DataElement,
                    Column TRANSFORMS DataElement,
                    Model VERSION_OF ModelVersion

    Here is some additional information that will help you.
    1. When a user asks a question about a specific node, never try to match the node name exactly as written. Instead, use CONTAINS() clauses to find nodes with given keywords in the node name. THIS IS THE MOST IMPORTANT RULE AND YOU MUST FOLLOW IT!!!!!
    2. When a user questions asks what is "upstream", that means they want to know which node(s) points to a specified node
    3. When a user questions asks what is "downstream", that means they want to know which node(s) point away fromm a specified node
    4. When a user questions asks "how many" of something, you must use a COUNT() statement to count nodes meeting a condition
    5. When a user question asks "Who", that refers to a User or a Contact node

    I want you to generate and return a Cypher Query that answers the user question: {query}, with additional information from the retrieved context: {context}, and graph schema: {schema}.
"""


USER_RESPONSE_TEMPLATE = """
    Given this user question: {query}
    And data from the Neo4j database: {cypher_query_response}

    Task: Answer the user question using only the data from the Neo4j database.  Use nested bullet points to summarize the answer if longer than one sentence.

    Example short answer response: The datasource for the Monthly Sales Trend field is 2 nodes upstream.
    
    Example of long answer response:

    The main differences between the latest version (Employee Productivity Model Version3) and the previous version (Employee Productivity Model Version2) of the Employee Productivity Prediction Model are as follows:
    
    1. Model Parameters:
        - Version3: Decision Tree algorithm with a maximum depth of 8 and a minimum samples split of 4.
        - Version2: Random Forest algorithm with 100 trees, a maximum depth of 10, and a minimum samples split of 2.

    2. Top Features:
        - Version3: The top features considered in Version3 are PerformanceScore (0.55), PerformanceReviewDate (0.25), and EmployeeID (0.2).
        - Version2: The top features considered in Version2 are PerformanceScore (0.4), PerformanceReviewDate (0.3), PerformanceComments (0.2), and EmployeeID (0.1).

    Overall, the key differences between the two versions lie in the choice of algorithm used, the parameters of the algorithm, and the weightage assigned to the top features in the model.
"""
