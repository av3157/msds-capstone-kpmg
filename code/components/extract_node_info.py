# Imports
from rapidfuzz import process, fuzz
import re
import inflect 

nodes = ["BusinessGroup", "Column", "Contact", "Database", 
        "DataElement", "Model", "ModelVersion", "Report", 
        "ReportField", "ReportSection", "Table", "User"]

lower_nodes = [node.lower() for node in nodes]

p = inflect.engine()

# match node from the list with generated bigrams
def match_node(user_input):
    # cleaning user input
    text = re.sub(r'[^\w\s]', '', user_input)
    words = text.split()
    singular_words = [p.singular_noun(word) or word for word in words]
    n = len(singular_words)
    bigrams = ["".join(singular_words[i:i + 2]) for i in range(n - 1)] + singular_words
    
    # Find best matching node among generated bigrams
    best_match, best_score = None, 0
    for phrase in bigrams:
        match, score, idx = process.extractOne(phrase, lower_nodes, scorer=fuzz.ratio)
        if score > best_score:
            best_match, best_score, best_idx = match, score, idx
    
    if best_score >= 55: 
        return nodes[best_idx] 
    else: 
        return None

# Test
# user_input = "What type is the Finance_Database" # purposely misspelled 
# result = match_node(user_input)
# print(result)
