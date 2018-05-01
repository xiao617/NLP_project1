import nltk
import os
import sys
import json
import nltk.sentiment.vader
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    train_f = open("training_set.json","r")
    train_cont = train_f.read()
    train_obj = json.loads(train_cont)
    sentences = []
    scores = []
    for i in train_obj:
        sentences.append(i['tweet'])
        scores.append(float(i['sentiment']))
    
    nltk_scores = []
    sid = nltk.sentiment.vader.SentimentIntensityAnalyzer()
    ptr = 0;
    for sentence in sentences:
        print(sentence)
        ss = sid.polarity_scores(sentence)
        
        for k in sorted(ss):
            #print('{0}: {1}, '.format(k, ss[k]), end='')
            print(ss[k], scores[ptr])
            if abs(ss[k])<0.00001:
                scores.pop(ptr)
                ptr-=1
            else:
                nltk_scores.append(ss[k])
            break;
        ptr+=1
    print(len(nltk_scores), len(scores))
    reshaped = [[i] for i in nltk_scores]
    model = LinearRegression()
    model.fit(reshaped, scores)
    print('R-squared: %.2f' % model.score(reshaped, scores))
    rms = sqrt(mean_squared_error(reshaped, scores))
    print(rms)

