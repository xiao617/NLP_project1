import json
import sys
import os

if __name__ == "__main__":
    train_f = open("training_set.json","r")
    train_cont = train_f.read()
    train_obj = json.loads(train_cont)
    
    corpus_f = open("NTUSD_Fin_word_v1.0.json", "r")
    corpus_cont = corpus_f.read()
    corpus_obj = json.loads(corpus_cont)
    corpus_dict = {}

    for i in corpus_obj:
        corpus_dict[i['token']] = float(i['market_sentiment'])

    for obj in train_obj:
        snippet = obj['snippet']
        if type(snippet) is not list:
            snippet = [snippet]
        summ = 0.0
        summ_count = 0
        for item in snippet:
            vocs = item.split(' ')
            for voc in vocs:
                if corpus_dict.__contains__(voc):
                    summ += corpus_dict[voc]
                    summ_count += 1
        if summ_count == 0:
            score = 0
        else:
            score = summ/summ_count
        print(obj['sentiment'], score)

