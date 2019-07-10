import spacy
from spacy import displacy
ner_model_path = "../models/1stmodel_5iterations"

nlp = spacy.load(ner_model_path)

text = 'The model is evaluated on English and Czech newspaper texts , and is then validated on French broadcast news transcriptions .'
# nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
displacy.serve(doc, style="ent")
