import spacy
from spacy import displacy

# text = """But Google is starting from behind. The company made a late push into hardware, and Apple’s Siri, available on iPhones, and Amazon’s Alexa software, which runs on its Echo and Dot devices, have clear leads in consumer adoption."""
# nlp = spacy.load("en_core_web_sm")
# doc = nlp(text)
# for ent in doc.ents:
#     print(ent.text, ent.start_char, ent.end_char, ent.label_)
# #displacy.serve(doc, style="ent")


import glob
import pickle
import re
import nltk
import json
def trim_entity_spans(data: list) -> list:
    """Removes leading and trailing white spaces from entity spans.

    Args:
        data (list): The data to be cleaned in spaCy JSON format.

    Returns:
        list: The cleaned data.
    """
    invalid_span_tokens = re.compile(r'\s')

    cleaned_data = []
    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                    text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])

    return cleaned_data
#
# g = glob.glob("raw_data/*.txt")
# TRAIN_DATA = []
# for files in g:
#     file_name  = files.split(".txt")[0]
#     f_text = open(files,"r")
#     text = f_text.read().strip()
#     #text = f_text.read()
#     #print(text)
#     print(files)
#     f_text.close()
#     f_ann = open(file_name+".ann","r")
#     dic = {}
#     ll = []
#     sentences = nltk.sent_tokenize(text)
#     current_sentence = sentences.pop(0)
#     current_len = len(current_sentence)
#     current_index = 0
#     last_len = len(current_sentence)
#     for line in f_ann:
#         line = line.strip()
#         if(line[0] == "T"):
#             line = line.split("\t")[1]
#             line = line.split(" ")
#             tag = line[0]
#             begin = int(line[1])
#             end = int(line[2])
#             if(begin > current_len):
#                 dic["entities"] = ll
#                 TRAIN_DATA.append((current_sentence, dic))
#                 ll = []
#                 dic = {}
#                 while(begin > current_len):
#                     try:
#                         current_sentence = sentences.pop(0)
#                         current_index = current_index + last_len + 1
#                         current_len = current_len + len(current_sentence)
#                         last_len = len(current_sentence)
#                     except:
#                         break
#             begin = begin - current_index
#             end = end - current_index
#             ss = (begin,end,tag)
#             ll.append(ss)
#     f_ann.close()
#     dic["entities"] = ll
#     TRAIN_DATA.append((current_sentence,dic))
#     #TRAIN_DAT = trim_entity_spans(TRAIN_DATA)
# pickle.dump(TRAIN_DATA,open("TRAIN_DATA_Sentence.MODEL","wb"))
# with open('your_file.txt', 'w') as f:
#     for item in TRAIN_DATA:
#         f.write("%s\n" % str(item))
# print(TRAIN_DATA)


#data = {"clusters": [[[84, 85], [168, 169]]], "sentences": [["In", "real-world", "action", "recognition", "problems", ",", "low-level", "features", "can", "not", "adequately", "characterize", "the", "rich", "spatial-temporal", "structures", "in", "action", "videos", "."], ["In", "this", "work", ",", "we", "encode", "actions", "based", "on", "attributes", "that", "describes", "actions", "as", "high-level", "concepts", "e.g.", ",", "jump", "forward", "or", "motion", "in", "the", "air", "."], ["We", "base", "our", "analysis", "on", "two", "types", "of", "action", "attributes", "."], ["One", "type", "of", "action", "attributes", "is", "generated", "by", "humans", "."], ["The", "second", "type", "is", "data-driven", "attributes", ",", "which", "are", "learned", "from", "data", "using", "dictionary", "learning", "methods", "."], ["Attribute-based", "representation", "may", "exhibit", "high", "variance", "due", "to", "noisy", "and", "redundant", "attributes", "."], ["We", "propose", "a", "discriminative", "and", "compact", "attribute-based", "representation", "by", "selecting", "a", "subset", "of", "discriminative", "attributes", "from", "a", "large", "attribute", "set", "."], ["Three", "attribute", "selection", "criteria", "are", "proposed", "and", "formulated", "as", "a", "submodular", "optimization", "problem", "."], ["A", "greedy", "optimization", "algorithm", "is", "presented", "and", "guaranteed", "to", "be", "at", "least", "-LRB-", "1-1", "/", "e", "-RRB-", "-", "approximation", "to", "the", "optimum", "."], ["Experimental", "results", "on", "the", "Olympic", "Sports", "and", "UCF101", "datasets", "demonstrate", "that", "the", "proposed", "attribute-based", "representation", "can", "significantly", "boost", "the", "performance", "of", "action", "recognition", "algorithms", "and", "outperform", "most", "recently", "proposed", "recognition", "approaches", "."]], "ner": [[[1, 4, "Task"], [6, 7, "OtherScientificTerm"], [13, 15, "OtherScientificTerm"], [17, 18, "Material"]], [[34, 35, "OtherScientificTerm"]], [[54, 55, "OtherScientificTerm"]], [[60, 61, "OtherScientificTerm"]], [[71, 72, "OtherScientificTerm"], [80, 82, "Method"]], [[84, 85, "Method"], [92, 95, "OtherScientificTerm"]], [[100, 104, "Method"], [110, 111, "OtherScientificTerm"]], [[119, 121, "Metric"], [128, 130, "Task"]], [[133, 135, "Method"]], [[159, 163, "Material"], [168, 169, "Method"], [176, 178, "Method"], [184, 185, "Method"]]], "relations": [[[13, 15, 17, 18, "FEATURE-OF"]], [], [], [], [[80, 82, 71, 72, "USED-FOR"]], [], [[110, 111, 100, 104, "USED-FOR"]], [[128, 130, 119, 121, "USED-FOR"]], [], [[159, 163, 168, 169, "EVALUATE-FOR"], [168, 169, 176, 178, "USED-FOR"], [176, 178, 184, 185, "COMPARE"]]], "doc_key": "NIPS_2014_21_abs"}
f = open("../data/scierc_data/json/dev.json","r")

TRAIN_DATA = []
for line in f:
    data = line.strip()
    data = json.loads(data)
    lll = 0
    for i in range(len(data["ner"])):
        ner_list = data["ner"][i]
        sentence = data["sentences"][i]
        index = 0
        sent = ""
        flag = 0
        ll = []
        for item in ner_list:
            begin = item[0] - lll
            #if(begin > len(sentence)):
            #    break
            end = item[1] + 1 - lll
            tag = item[2]
            sent = sent + " ".join(sentence[index:begin])
            sent = sent + " "
            nner = " ".join(sentence[begin:end])
            first_index = len(sent)
            last_index = first_index + len(nner)
            sent = sent + nner + " "
            ss = (first_index, last_index, tag)
            ll.append(ss)
            index = end
            flag = 1
        if(flag == 1):
            dic = {}
            sent = sent + " ".join(sentence[index:])
            dic["entities"] = ll
            TRAIN_DATA.append((sent, dic))
        lll = lll + len(sentence)


print(len(TRAIN_DATA))
pickle.dump(TRAIN_DATA,open("DEV_DATA_json.MODEL","wb"))