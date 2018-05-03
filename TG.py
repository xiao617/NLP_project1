# coding:utf-8
import json
import sys
import re
import numpy as np
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
		self.sentiment = 0.0
		self.group_t = 0.0
		self.group_s = 0.0
        
	def cutData(cutList,string):
		# Replace parts of string with " "
		string = re.sub('(\.\.+)|(\!+)', " ", string)
		string = string.replace("--", " ").replace("?", " ")
		string = string.replace("\t", " ").replace("\n", " ")
		string = string.replace("(", " ").replace(")", " ")
		# Split string and get data wanted
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
		# Deal with input json data
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
	# Use TargetList (market_sentiment of word from NTUSD-Fin) to caculate average score of Datalist
	score = 0.0
	n = 0
	for string in DataList:
		if string in TargetList:
			score = score + TargetList[string]
			n = n + 1
	if n > 0:
		return score/n
	else:
		return 0

def GroupValue_t(Value):
	# Return category value of Value (for tgrocery)
	value = float(Value)	
	v = [0.8,0.6,0.4]
	for i in range(len(v)):
		if value > v[i]:
			return v[i]
		elif value < -v[i]:
			return -v[i]
	return 0.0

def GroupValue_s(Value):
	# Return category value of Value (for tgrocery)
	value = float(Value)	
	v = [0.9,0.1]
	for i in range(len(v)):
		if value > v[i]:
			return v[i]
		elif value < -v[i]:
			return -v[i]
	return 0.0

def Value(x):
	# Return category value of Value (for f1_score)
	v = 0.01
	if x > v:
		return 1.0
	elif x < -v:
		return -1.0
	else:
		return 0.0
		

def main():
	# Get market_sentiment of word from NTUSD-Fin
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
		sg = str(GroupValue_s(str(targetDict[word]/3.5)))
		train_s.append((sg,word))
	with open('NTUSD-Fin/NTUSD_Fin_word_v1.0.json', 'r', encoding='utf-8') as f:
		targetIn = json.load(f)
	N = len(targetIn)
	for i in range(N):
		word = targetIn[i]['token']
		targetDict[word] = targetIn[i]['market_sentiment']
		sg = str(GroupValue_s(str(targetDict[word]/3.5)))
		train_s.append((sg,word))

	# Training File: Load data & Use tgrocery to train classification model
	TrainingFile = open('training_set.json','r')
	TrainingData = json.load(TrainingFile)
	TrainingFile.close()
	DataList = []
	grocery_t = Grocery("tweet")
	grocery_s = Grocery("snippet")
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
	grocery_t.train(train_t+train_s)
	grocery_t.save()
	grocery_s.train(train_s)
	grocery_s.save()
    
    # Save training data created by WordScore() and GroupValue_*()
    # Data will be uesd for LinearRegression() in BOTH.py
	outfile = open('TG_train.txt', 'w', encoding='utf-8')   
	dataScore = []
	dataSentiment = []
	for row in DataList:
		dataSentiment.append([float(row.sentiment)])
		a = WordScore(row.tweet,targetDict)
		b = WordScore(row.snippet,targetDict)
		c = row.group_t
		d = row.group_s
		dataScore.append([a,b,c,d])
		print(a, b, c, d, file=outfile)
	outfile.close()

	'''
	# Train linear regression model
	model = LinearRegression()
	model.fit(dataScore, dataSentiment)

	# Test for training data
	print('(train)R-squared: %.3f' % model.score(dataScore, dataSentiment)) #0.915
	predictions = model.predict(dataScore)
	rms = mean_squared_error(dataSentiment,predictions)
	print('RMSE: %.3f' % sqrt(rms)) #0.110
	print('MSE: %.3f' % rms) #0.012
	'''
	
	# Testing File: Load data & Use tgrocery classification model to predict
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
    
    # Save testing data created by WordScore() and classification prediction
    # Data will be uesd for LinearRegression() in BOTH.py
	outfile = open('TG_test.txt', 'w', encoding='utf-8')    
	dataScore = []
	dataSentiment = []
	for row in DataList:
		dataSentiment.append([float(row.sentiment)])
		a = WordScore(row.tweet,targetDict)
		b = WordScore(row.snippet,targetDict)
		c = row.group_t
		d = row.group_s
		dataScore.append([a,b,c,d])
		print(a, b, c, d, file=outfile)
	outfile.close()

	'''
	# Test for testing data
	predictions = model.predict(dataScore)
	x = []
	y = []
	for i, prediction in enumerate(predictions):
		y.append(Value(prediction[0]))
		x.append(Value(dataSentiment[i][0]))
	#	print('Predicted: %s, Target: %s' % (prediction, dataSentiment[i]))	
	print('(test)R-squared: %.3f' % model.score(dataScore, dataSentiment)) #0.502
	rms = mean_squared_error(dataSentiment,predictions)
	print('RMSE: %.3f' % sqrt(rms)) #0.268
	print('MSE: %.3f' % rms) #0.072
	print('f1-micro: %.3f' % f1_score(x,y,average='micro'))
	print('f1-macro: %.3f' % f1_score(x,y,average='macro'))
	'''

if __name__ == "__main__":
	main()
