from gensim.models import Word2Vec
import multiprocessing
import json
import spacy
nlp =  spacy.load("en_core_web_sm")



#corpus downloaded from: https://api.semanticscholar.org/corpus/download/



EMB_DIM=300
where_to_save="data/w2v300.kv"
input_json = "data/semanticscholar/papers-2017-02-21.json"
how_many_abstracts = 10000


sentences=[]
abstracts=[]
i=0
with open(input_json, "r") as ins:
    for line in ins:
        i+=1
        if i % 100 == 0:
            print (i)
        if i>how_many_abstracts:
            break
        data = json.loads(line)
        abstracts.append(data["paperAbstract"])
        doc = nlp(data["paperAbstract"])
        for sent in doc.sents:
            sentences.append(sent.string.strip())


print (sentences[:3])

w2v = Word2Vec(sentences, size=EMB_DIM, window=5, min_count=5, negative=15, iter=10, workers=multiprocessing.cpu_count())
w2v.wv.save(where_to_save)



