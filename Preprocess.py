import json
import sys
import os
from nltk.stem.snowball import SnowballStemmer

if __name__ == "__main__":
    origin_name = sys.argv(1)
    output_name = sys.argv(2)
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    train_f = open(origin_name,"r")
    train_cont = train_f.read()
    train_obj = json.loads(train_cont)
    new_obj = []
    for obj in train_obj:
        tweet = obj['tweet'].lower()
        new_tweet = []
        for word in tweet.split(' '):
            word = word.strip(',').strip('.').strip('!').strip(':').strip(';').strip(' ').strip('#').strip('(').strip(')')
            #print(word)
            if word.find("http")>-1 or len(word)<=0:
                continue
            elif word.find('$') > -1 and len(word)-1 > word.find('$') and word[word.find('$')+1:].isalpha():
                word = '^CASHTAG'
            elif word[0] == '+':
                word = '^INCREASE'
            elif word[0] == '-':
                word = '^DECREASE'
            else:
                word = stemmer.stem(word)
                #print('>' + word)
            new_tweet.append(word)
        obj['tweet'] = ' '.join(new_tweet)
        #print(new_tweet)
        new_obj.append(obj)
    new_file = open(output_name, "w")
    new_file.write(json.dumps(new_obj))
