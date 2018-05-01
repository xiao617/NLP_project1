# coding:utf-8
import json
import re
import math
from math import sqrt

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class DataManager:
	def __init__(self):
		self.tweet = []
		self.target = ""
		self.snippet = []
		self.sentiment = 0
        
	def cutData(cutList,string):
		string = re.sub('(\.\.+)|(\!+)', " ", string)
		string = string.replace("--", " ").replace("?", " ")
		string = string.replace("\t", " ").replace("\n", " ")
		string = string.replace("(", " ").replace(")", " ")
		testyoy = string.split(" ")
		for testline in testyoy:
			n = len(testline)
			if n > 0:
				if testline[0] == "@":
					continue
				elif testline[0] == "$":
					if n == 1:
						continue
					if testline[1]>="A" and testline[1]<="Z":
						continue
					if testline[1]>="a" and testline[1]<="z":
						continue
				if testline[n-1] == "." or testline[n-1] == ",":
					cutList.append(testline[:n-1])
				else:
					cutList.append(testline)

	def insertData(self,DataList):
		self.target = DataList["target"]
		self.sentiment = float(DataList["sentiment"])
		
		string = re.sub("https?://[\w\-]+(\.[\w\-]+)+\S*", " ", DataList["tweet"])
		DataManager.cutData(self.tweet, string)
		
		if isinstance(DataList["snippet"], list):
			for string in DataList["snippet"]:
				DataManager.cutData(self.snippet, string)
		else:
			DataManager.cutData(self.snippet, DataList["snippet"])

# load data from specific path
def load_data(path):
	file = open(path,'r')
	data = json.load(file)
	file.close()
	DataList = []
	for DataElement in data:
		tempt = DataManager()
		tempt.insertData(DataElement)
		DataList.append(tempt)

	return DataList


def count_sentiment(sentiment, counter):

	if sentiment > 0.7: counter[0] += 2
	elif sentiment > 0.2: counter[0] += 1
	elif sentiment > 0: counter[0] += 0.5
	elif sentiment < -0.7: counter[1] += 2
	elif sentiment < -0.2: counter[1] += 1
	elif sentiment < 0: counter[0] += 0.5

	return counter


def PMI(word, wc, counter):

	if word not in wc: return 0 # unseen

	PMI = [0, 0] # [bull, bear]
	word_total = wc[word][0] + wc[word][1]
	sent_total = counter[0] + counter[1]
	for i in [0, 1]:
		if wc[word][i] > 0:
			PMI[i] = math.log2( sent_total * wc[word][i] / ( counter[i] * word_total ))

	return (PMI[0] - PMI[1])


def main():

	# Training File
	DataList = load_data('training_set_preprocessed.json')

	# [bull_count, bear_count]
	wc_bi = dict()
	wc = dict()
	wc_snip = dict()
	wc_snip_bi = dict()
	counter = [0, 0]

	for x in DataList:
		counter = count_sentiment(x.sentiment, counter)
		# count tweet unigram
		for w in x.tweet:
			if not w in wc:
				wc[w] = [0, 0]
			wc[w] = count_sentiment(x.sentiment, wc[w])
		# count tweet bigram
		for i in range(0, len(x.tweet) - 1):
			w = x.tweet[i] + '_' + x.tweet[i+1]
			if not w in wc_bi:
				wc_bi[w] = [0, 0]
			wc_bi[w] = count_sentiment(x.sentiment, wc_bi[w])
		# count snippet unigram
		for w in x.snippet:
			if not w in wc_snip:
				wc_snip[w] = [0, 0]
			wc_snip[w] = count_sentiment(x.sentiment, wc_snip[w])
		# count snippet bigram
		for i in range(0, len(x.snippet) - 1):
			w = x.snippet[i] + '_' + x.snippet[i+1]
			if not w in wc_snip_bi:
				wc_snip_bi[w] = [0, 0]
			wc_snip_bi[w] = count_sentiment(x.sentiment, wc_snip_bi[w])

	# Testing File
	TestList = load_data('test_set_preprocessed.json')

	dataScore = []
	dataSentiment = []
	for row in TestList:
		# score from tweet unigram
		sc = 0.0
		for w in row.tweet:
			sc += PMI(w, wc, counter)
		# score from tweet bigram
		sc_bi = 0.0
		for i in range(0, len(row.tweet) - 1):
			w = row.tweet[i] + '_' + row.tweet[i+1]
			sc_bi += PMI(w, wc_bi, counter)
		# score from snippet unigram
		sc_snip = 0.0
		for w in row.snippet:
			sc_snip += PMI(w, wc_snip, counter)
		# score from snippet bigram
		sc_snip_bi = 0.0
		for i in range(0, len(row.snippet) - 1):
			w = row.snippet[i] + '_' + row.snippet[i+1]
			sc_snip_bi += PMI(w, wc_snip_bi, counter)
		dataSentiment.append([float(row.sentiment)])
		dataScore.append([ sc, sc_bi, sc_snip, sc_snip_bi])

	model = LinearRegression()
	model.fit(dataScore, dataSentiment)
	prediction = model.predict(dataScore)

	# for arr in dataScore:
	# 	print(arr[0], arr[1], arr[2])

	# classify
	right = 0
	wrong = 0
	for i in range(0, len(prediction)-1):
		if prediction[i] == 0:
			if dataSentiment[i] == 0: right += 1
			else: wrong += 1
		elif prediction[i][0] > 0 and dataSentiment[i][0] > 0: right += 1
		elif prediction[i][0] < 0 and dataSentiment[i][0] < 0: right += 1
		else: wrong += 1

	print('MSE: %.5f' % mean_squared_error(prediction, dataSentiment))
	print('F1: %.5f' % (right / (right+wrong)))

if __name__ == "__main__":
	main()