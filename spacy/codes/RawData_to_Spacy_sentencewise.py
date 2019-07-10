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
    #text = f_text.read()
    #print(text)
    print(files)
    f_text.close()
    f_ann = open(file_name+".ann","r")
    dic = {}
    ll = []
    sentences = nltk.sent_tokenize(text)
    current_sentence = sentences.pop(0)
    current_len = len(current_sentence)
    current_index = 0
    last_len = len(current_sentence)
    for line in f_ann:
        line = line.strip()
        if(line[0] == "T"):
            line = line.split("\t")[1]
            line = line.split(" ")
            tag = line[0]
            begin = int(line[1])
            end = int(line[2])
            if(begin > current_len):
                dic["entities"] = ll
                TRAIN_DATA.append((current_sentence, dic))
                ll = []
                dic = {}
                while(begin > current_len):
                    try:
                        current_sentence = sentences.pop(0)
                        current_index = current_index + last_len + 1
                        current_len = current_len + len(current_sentence)
                        last_len = len(current_sentence)
                    except:
                        break
            begin = begin - current_index
            end = end - current_index
            ss = (begin,end,tag)
            ll.append(ss)
    f_ann.close()
    dic["entities"] = ll
    TRAIN_DATA.append((current_sentence,dic))
    #TRAIN_DAT = trim_entity_spans(TRAIN_DATA)
pickle.dump(TRAIN_DATA,open("TRAIN_DATA_Sentence.MODEL","wb"))
with open('your_file.txt', 'w') as f:
    for item in TRAIN_DATA:
        f.write("%s\n" % str(item))
print(TRAIN_DATA)
