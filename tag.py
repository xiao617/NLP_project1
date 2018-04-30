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


def PMI(word, wc, bull, bear):

	if word not in wc:
		return 0

	if wc[word][0] > 0:
		PMI_bull = math.log2( (bull+bear) * wc[word][0] / ( bull * (wc[word][0] + wc[word][1]) ))
	else:
		PMI_bull = 0

	if wc[word][1] > 0:
		PMI_bear = math.log2( (bull+bear) * wc[word][1] / ( bear * (wc[word][0] + wc[word][1]) ))
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

	wc = dict()
	bull = 0
	bear = 0

	for x in DataList:
		if x.sentiment > 0:
			bull += 1
		else:
			bear += 1
		for w in x.tweet:
			if not w in wc:
				wc[w] = [0, 0]
			if x.sentiment > 0:
				wc[w][0] += 1
			else:
				wc[w][1] += 1
    
	# for w in wc:
	# 	print( w + ', ' + str(PMI(w, wc, bull, bear)) )

	# Testing File
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
		for w in row.tweet:
			sc += PMI(w, wc, bull, bear)
		dataScore.append([sc])

	# print(dataScore)

	model = LinearRegression()
	model.fit(dataScore, dataSentiment)

	print('\nR-squared: %.2f' % model.score(dataScore, dataSentiment))


if __name__ == "__main__":
	main()