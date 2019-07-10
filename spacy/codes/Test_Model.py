# import spacy
# from spacy import displacy
# nlp = spacy.load('2nd_model')
#
# text = """Multimedia answers  include  videodisc images  and heuristically-produced complete  sentences  in  text  or  text-to-speech form  .  Deictic reference  and  feedback  about the  discourse  are enabled. The  interface  thus presents the application as cooperative and conversational."""
# # nlp = spacy.load("en_core_web_sm")
# doc = nlp(text)
# for ent in doc.ents:
#     print(ent.text, ent.start_char, ent.end_char, ent.label_)
# displacy.serve(doc, style="ent")

import pickle
import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer

def evaluate(ner_model, examples):
    scorer = Scorer()
    for input_, annot in examples:
        doc_gold_text = ner_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot['entities'])
        pred_value = ner_model(input_)
        scorer.score(pred_value, gold)
    return scorer.scores


TEST_DATA = pickle.load(open("data/TEST_DATA_json.MODEL","rb"))


# TEST_DATA = [
#     ('Who is Shaka Khan?',
#      [(7, 17, 'PERSON')]),
#     ('I like London and Berlin.',
#      [(7, 13, 'LOC'), (18, 24, 'LOC')])
# ]

ner_model_path = "models/1stmodel_50iteration_local"
ner_model = spacy.load(ner_model_path) # for spaCy's pretrained use 'en_core_web_sm'
results = evaluate(ner_model, TEST_DATA)
print(results)