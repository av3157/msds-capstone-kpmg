from constants.prompt_templates import INTENT_MATCHING_TEMPLATE, INPUT_PARAMETER_EXTRACTION_TEMPLATE
from constants.db_constants import DATABASE_SCHEMA
from constants.chatbot_responses import FAILED_INTENT_MATCH
import re
from common_intent import classify_intent, common_question_embeddings

INTENT_MATCHING_COMMON_QUESTION_DELIMITER = ','

def get_request_intent(user_request, llm):
    # Intent matching
    intent_matching_response = llm.generate(INTENT_MATCHING_TEMPLATE.format(schema=DATABASE_SCHEMA, question=user_request))
    print(f"Intent matching result: {intent_matching_response}")

    intent_match_response_data = ""
    regex_match = re.search(r'\[(.*?)\]', intent_matching_response)
    if regex_match:
        intent_match_response_data = regex_match.group(1)
    else:
        return FAILED_INTENT_MATCH
    
    if intent_match_response_data == "NONE,-1":
        return ['NONE', -1]
    
    # Extract relevant data from intent matching response
    intent = classify_intent(user_request, common_question_embeddings, threshold = 0.85)
    return intent

    
def get_input_parameter(user_request, llm):
    input_parameter_response = llm.generate(INPUT_PARAMETER_EXTRACTION_TEMPLATE.format(schema=DATABASE_SCHEMA, question=user_request))
    match = re.search(r'\[(.*?)\]', input_parameter_response)
    return match.group(1).split(",")