# coding:utf-8
import numpy as np
import json
import sys
import re
import sklearn
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from tgrocery import Grocery
from math import sqrt

class DataManager:
	def __init__(self):
		self.tweet = []
		self.target = ""
		self.snippet = []
		self.sentiment = 0
		self.group_t = 0
		self.group_s = 0
        
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
		self.sentiment = DataList["sentiment"]
		
		string = re.sub("https?://[\w\-]+(\.[\w\-]+)+\S*", " ", DataList["tweet"])
		DataManager.cutData(self.tweet, string)
		
		if isinstance(DataList["snippet"], list):
			for string in DataList["snippet"]:
				DataManager.cutData(self.snippet, string)
		else:
			DataManager.cutData(self.snippet, DataList["snippet"])
            
def WordScore(DataList,TargetList):    
	score = 0
	n = 0
	for string in DataList:
		if string in TargetList:
			score = score + TargetList[string]
			n = n + 1
	if n > 0:
		return score/n
	else:
		return 0
    
def TargetScore(DataList,Target):
	n = len(DataList)
	value = 0
	for i in range(n):
		if Target in DataList[i]:
			if i+1 < n:
				if DataList[i+1][0] == '-':
					if DataList[i+1][1] >= '0' and DataList[i+1][1] <= '9':
						value = value - 1
				elif DataList[i+1][0] == '+':
					value = value + 1
				elif DataList[i+1][0] >= '0' and DataList[i+1][0] <= '9':
					value = value + 1
	return value  

def ValueScore(DataList):
	N = len(DataList)
	value = 0
	n = 0
	for i in range(N):
		NumList = re.findall("(\+|\-)?(\d+)(\.)?(\d+)(\%)?", DataList[i])
		n = len(NumList)
		for Num in NumList:
			if '-' in Num:
				value = value - 1
			else:
				value = value + 1
	if n == 0:
		return 0
	else:
		return value/n  

def GroupValue_t(Value):
	value = float(Value)	
	v = [0.8,0.6,0.4]
	for i in range(len(v)):
		if value > v[i]:
			return v[i]
		elif value < -v[i]:
			return -v[i]
	return 0

def GroupValue_s(Value):
	value = float(Value)	
	v = [0.9,0.6,0.3]
	for i in range(len(v)):
		if value > v[i]:
			return v[i]
		elif value < -v[i]:
			return -v[i]
	return 0			

def main():
	# Word Score List
	train_t = [] 
	train_s = []
	targetIn = {}
	targetDict = dict()
	with open('NTUSD-Fin/NTUSD_Fin_hashtag_v1.0.json', 'r', encoding='utf-8') as f:
		targetIn = json.load(f)
	N = len(targetIn)
	for i in range(N):
		word = "#" + targetIn[i]['token']
		targetDict[word] = targetIn[i]['market_sentiment']
		sg = str(GroupValue_s(str(targetDict[word]/3)))
		train_s.append((sg,word))
	with open('NTUSD-Fin/NTUSD_Fin_word_v1.0.json', 'r', encoding='utf-8') as f:
		targetIn = json.load(f)
	N = len(targetIn)
	for i in range(N):
		word = targetIn[i]['token']
		targetDict[word] = targetIn[i]['market_sentiment']
		sg = str(GroupValue_s(str(targetDict[word]/3)))
		train_s.append((sg,word))

	# Training File
	TrainingFile = open('training_set.json','r')
	TrainingData = json.load(TrainingFile)
	TrainingFile.close()
	DataList = []
	grocery_t = Grocery("tweet")
	grocery_s = Grocery("snippet")
#	train_t = [] 
#	train_s = [] 
	for DataElement in TrainingData:
		tempt = DataManager() 
		tempt.insertData(DataElement)
		tempt.group_t = GroupValue_t(tempt.sentiment)
		tempt.group_s = GroupValue_s(tempt.sentiment)
		line = re.sub("https?://[\w\-]+(\.[\w\-]+)+\S*", " ", DataElement["tweet"])
		train_t.append((str(tempt.group_t),line))
		if isinstance(DataElement["snippet"], list):
			for line in DataElement["snippet"]:
				train_s.append((str(tempt.group_s),line))
		elif DataElement["snippet"] != "":
			train_s.append((str(tempt.group_s),DataElement["snippet"]))
		else:
			tempt.group_s = 0.0
		DataList.append(tempt)
	grocery_t.train(train_t)
	grocery_t.save()
	grocery_s.train(train_s)
	grocery_s.save()
    
	#outfile = open('train_1.txt', 'w', encoding='utf-8')   
	dataScore = []
	dataSentiment = []
	for row in DataList:
		dataSentiment.append([float(row.sentiment)])
		a = WordScore(row.tweet,targetDict)
		b = WordScore(row.snippet,targetDict)
		c = TargetScore(row.tweet,row.target)
		d = ValueScore(row.snippet)
		e = row.group_t
		f = row.group_s
		dataScore.append([a,b,c,d,e,f])
	#print(dataScore, file=outfile)
	#outfile.close()
	model = LinearRegression()
	model.fit(dataScore, dataSentiment)
	print('(model)R-squared: %.3f' % model.score(dataScore, dataSentiment)) #0.886
	predictions = model.predict(dataScore)
	rms = mean_squared_error(dataSentiment,predictions)
	print('RMSE: %.3f' % sqrt(rms)) #0.127
	print('MSE: %.3f' % rms) #0.016
	
	# Testing File
	TestingFile = open('test_set.json','r')
	TestingData = json.load(TestingFile)
	TestingFile.close()
	DataList = []
	new_grocery_t = Grocery('tweet')
	new_grocery_t.load()
	new_grocery_s = Grocery('snippet')
	new_grocery_s.load()
	for DataElement in TestingData:
		tempt = DataManager() 
		tempt.insertData(DataElement)
		line = re.sub("https?://[\w\-]+(\.[\w\-]+)+\S*", " ", DataElement["tweet"])
		tempt.group_t = float('{0}'.format(new_grocery_t.predict(line)))
		value = 0.0
		if isinstance(DataElement["snippet"], list):
			for line in DataElement["snippet"]:
				value = value + float('{0}'.format(new_grocery_s.predict(line)))
			value = value / len(DataElement["snippet"])
		elif DataElement["snippet"] != "":
			value = float('{0}'.format(new_grocery_s.predict(DataElement["snippet"])))
		tempt.group_s = value
		DataList.append(tempt)
    
	#outfile = open('test_1.txt', 'w', encoding='utf-8')    
	dataScore = []
	dataSentiment = []
	for row in DataList:
		dataSentiment.append([float(row.sentiment)])
		a = WordScore(row.tweet,targetDict)
		b = WordScore(row.snippet,targetDict)
		c = TargetScore(row.tweet,row.target)
		d = ValueScore(row.snippet)
		e = row.group_t
		f = row.group_s
		dataScore.append([a,b,c,d,e,f])
	#print(dataScore, file=outfile)
	#outfile.close()
	predictions = model.predict(dataScore)
	#for i, prediction in enumerate(predictions):
	#	print('Predicted: %s, Target: %s' % (prediction, dataSentiment[i]))
	print('(test)R-squared: %.3f' % model.score(dataScore, dataSentiment)) #0.430
	rms = mean_squared_error(dataSentiment,predictions)
	print('RMSE: %.3f' % sqrt(rms)) #0.287
	print('MSE: %.3f' % rms) #0.082


if __name__ == "__main__":
	main()