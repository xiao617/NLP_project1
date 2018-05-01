# coding:utf-8
import numpy as np
import json
import sys
import re
import math

import sklearn
from sklearn.linear_model import LinearRegression

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


def PMI(word, wc_bi, bull, bear):

	if word not in wc_bi:
		return 0

	if wc_bi[word][0] > 0:
		PMI_bull = math.log2( (bull+bear) * wc_bi[word][0] / ( bull * wc_bi[word][2] ))
	else:
		PMI_bull = 0

	if wc_bi[word][1] > 0:
		PMI_bear = math.log2( (bull+bear) * wc_bi[word][1] / ( bear * wc_bi[word][2] ))
	else:
		PMI_bear = 0

	return (PMI_bull - PMI_bear)


def main():

	# Training File
	TrainingFile = open('training_set.json','r')
	TrainingData = json.load(TrainingFile)
	TrainingFile.close()
	DataList = []
	for DataElement in TrainingData:
		tempt = DataManager() 
		tempt.insertData(DataElement)
		DataList.append(tempt)

	wc_bi = dict()
	wc = dict()
	bull = 0
	bear = 0

	for x in DataList:
		if x.sentiment > 0.25:
			bull += 1
		elif x.sentiment < -0.25:
			bear += 1

		# count unigram
		for w in x.tweet:
			if not w in wc:
				wc[w] = [0, 0, 0]
			if x.sentiment > 0.25:
				wc[w][0] += 1
			elif x.sentiment < -0.25:
				wc[w][1] += 1
			wc[w][2] += 1

		# count simple bigram
		for i in [0, len(x.tweet) - 2]:
			for j in [i, len(x.tweet) - 1]:
				w = x.tweet[i] + '_' + x.tweet[j]
				if not w in wc_bi:
					wc_bi[w] = [0, 0, 0]
				if x.sentiment > 0.25:
					wc_bi[w][0] += 1
				elif x.sentiment < -0.25:
					wc_bi[w][1] += 1
				wc_bi[w][2] += 1

	# Testing File
	# TestingFile = open('training_set.json','r')
	TestingFile = open('test_set.json','r')
	TestingData = json.load(TestingFile)
	TestingFile.close()
	TestList = []
	for DataElement in TestingData:
		tempt = DataManager() 
		tempt.insertData(DataElement)
		TestList.append(tempt)

	dataScore = []
	dataSentiment = []
	for row in TestList:
		dataSentiment.append([float(row.sentiment)])
		sc = 0.0
		sc_bi = 0.0
		# unigram
		for w in row.tweet:
			sc += PMI(w, wc, bull, bear)
		# distant bigram
		for i in [0, len(row.tweet) - 2]:
			for j in [i, len(row.tweet) - 1]:
				w = row.tweet[i] + '_' + row.tweet[j]
				sc_bi += PMI(w, wc_bi, bull, bear)

		dataScore.append([ 0.4 * sc + 0.6 * sc_bi ])

	# print(dataScore)

	model = LinearRegression()
	model.fit(dataScore, dataSentiment)

	print('\nR-squared: %.2f' % model.score(dataScore, dataSentiment))


if __name__ == "__main__":
	main()