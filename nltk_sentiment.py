import nltk
import os
import sys
import json
import nltk.sentiment.vader
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, f1_score
from math import sqrt

def Value(x):
	v = 0.01
	if x > v:
		return 1.0
	elif x < -v:
		return -1.0
	else:
		return 0.0

if __name__ == "__main__":
    train_f = open("training_set_preprocess.json","r")
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
        ss = sid.polarity_scores(sentence)
        for k in sorted(ss):
            #print('{0}: {1}, '.format(k, ss[k]), end='')
            if abs(ss[k])<0.00001:
                scores.pop(ptr)
                ptr-=1
            else:
                nltk_scores.append(ss[k])
            break;
        ptr+=1
    reshaped = [[i] for i in nltk_scores]
    model = LinearRegression()
    model.fit(reshaped, scores)

    #test
    test_f = open("test_set_preprocess.json","r")
    test_cont = test_f.read()
    test_obj = json.loads(test_cont)
    sentences = []
    scores = []
    for i in train_obj:
        sentences.append(i['tweet'])
        scores.append(float(i['sentiment']))
    
    nltk_scores = []
    sid = nltk.sentiment.vader.SentimentIntensityAnalyzer()
    ptr = 0;
    for sentence in sentences:
        ss = sid.polarity_scores(sentence)
        for k in sorted(ss):
            #print('{0}: {1}, '.format(k, ss[k]), end='')
            if abs(ss[k])<0.00001:
                scores.pop(ptr)
                ptr-=1
            else:
                nltk_scores.append(ss[k])
            break;
        ptr+=1
    reshaped = [[i] for i in nltk_scores]
    predict = model.predict(reshaped)
    print('MSE: %.3f' % mean_squared_error(predict, scores))
    rms = sqrt(mean_squared_error(reshaped, scores))
    print('f1-micro: %.3f' % f1_score([Value(i) for i in predict], [Value(i) for i in scores] ,average='micro'))
    print('f1-macro: %.3f' % f1_score([Value(i) for i in predict], [Value(i) for i in scores] ,average='macro'))

    print(rms)

