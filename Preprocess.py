import json
import sys
import os

if __name__ == "__main__":
    # train_f = open("training_set.json","r")
    train_f = open("test_set.json","r")
    train_cont = train_f.read()
    train_obj = json.loads(train_cont)
    new_obj = []
    for obj in train_obj:
        tweet = obj['tweet'].lower()
        new_tweet = []
        for word in tweet.split(' '):
            word = word.strip(',').strip('.').strip('!').strip(':').strip(';').strip(' ').strip('#')
            #print(word)
            if word.find("http")>-1 or len(word)<=0:
                continue
            elif word.find('$') > -1 and len(word)-1 > word.find('$') and word[word.find('$')+1:].isalpha():
                word = '^CASHTAG'
            elif (word[0] in ['+', '-']):
                try:
                    number = float(word[1:] if word[-1] != '%' else word[1:-1])
                except:
                    number = -1.0
                if number>=0.0:
                    tmp = '^INCREASE' if word[0]=='+' else '^DECREASE'
                    tmp += str(int(number)//10 if int(number) < 100 else 10)
                    tmp += word[-1] if word[-1]=='%' else ''
                    #print(tmp, word)
                    word = tmp
            new_tweet.append(word)
        obj['tweet'] = ' '.join(new_tweet)
        #print(new_tweet)
        new_obj.append(obj)
    # new_file = open("training_set_prepocessed.json", "w")
    new_file = open("test_set_prepocessed.json", "w")
    new_file.write(json.dumps(new_obj))
