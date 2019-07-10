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

g = glob.glob("raw_data/*.txt")
TRAIN_DATA = []
for files in g:
    file_name  = files.split(".txt")[0]
    f_text = open(files,"r")
    text = f_text.read().strip()
    f_text.close()
    f_ann = open(file_name+".ann","r")
    dic = {}
    ll = []
    for line in f_ann:
        line = line.strip()
        if(line[0] == "T"):
            line = line.split("\t")[1]
            line = line.split(" ")
            ss = (int(line[1]),int(line[2]),line[0])
            ll.append(ss)
    f_ann.close()
    dic["entities"] = ll
    TRAIN_DATA.append((text,dic))
    #TRAIN_DAT = trim_entity_spans(TRAIN_DATA)
pickle.dump(TRAIN_DATA,open("TRAIN_DATA.MODEL","wb"))
print(TRAIN_DATA)
