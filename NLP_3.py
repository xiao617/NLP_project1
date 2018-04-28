# coding:utf-8
import numpy as np
import json
import sys
import re
import sklearn
#from sklearn.metrics import f1_score
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
		self.group = 0
        
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
		DataManager.cutData(self.tweet, string)# coding:utf-8
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
		self.group = 0
        
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

def GroupValue(Value):
	value = float(Value)	
	v = [0.8,0.6,0.4]
	for i in range(3):
		if value > v[i]:
			return v[i]
		elif value < -v[i]:
			return -v[i]
	return 0			

def main():
	# Word Score List
	targetIn = {}
	targetDict = dict()
	with open('NTUSD-Fin/NTUSD_Fin_hashtag_v1.0.json', 'r', encoding='utf-8') as f:
		targetIn = json.load(f)
	N = len(targetIn)
	for i in range(N):
		word = "#" + targetIn[i]['token']
		targetDict[word] = targetIn[i]['market_sentiment']
	with open('NTUSD-Fin/NTUSD_Fin_word_v1.0.json', 'r', encoding='utf-8') as f:
		targetIn = json.load(f)
	N = len(targetIn)
	for i in range(N):
		word = targetIn[i]['token']
		targetDict[word] = targetIn[i]['market_sentiment']

	# Training File
	TrainingFile = open('training_set.json','r')
	TrainingData = json.load(TrainingFile)
	TrainingFile.close()
	DataList = []
	grocery = Grocery("Sample")
	train_src = [] 
	for DataElement in TrainingData:
		tempt = DataManager() 
		tempt.insertData(DataElement)
		tempt.group = GroupValue(tempt.sentiment)
		line = re.sub("https?://[\w\-]+(\.[\w\-]+)+\S*", " ", DataElement["tweet"])
		train_src.append((str(tempt.group),line))
		DataList.append(tempt)
	grocery.train(train_src)
	grocery.save()
    
	#outfile = open('train_1.txt', 'w', encoding='utf-8')   
	dataScore = []
	dataSentiment = []
	for row in DataList:
		dataSentiment.append([float(row.sentiment)])
		a = WordScore(row.tweet,targetDict)
		b = WordScore(row.snippet,targetDict)
		c = TargetScore(row.tweet,row.target)
		d = ValueScore(row.snippet)
		e = row.group
		dataScore.append([a,b,c,d,e])
	#print(dataScore, file=outfile)
	#outfile.close()
	model = LinearRegression()
	model.fit(dataScore, dataSentiment)
	print('(model)R-squared: %.3f' % model.score(dataScore, dataSentiment)) #0.717
	predictions = model.predict(dataScore)
	rms = mean_squared_error(dataSentiment,predictions)
	print('RMSE: %.3f' % sqrt(rms)) #0.200
	print('MSE: %.3f' % rms) #0.040
	
	# Testing File
	TestingFile = open('test_set.json','r')
	TestingData = json.load(TestingFile)
	TestingFile.close()
	DataList = []
	new_grocery = Grocery('Sample')
	new_grocery.load()
	for DataElement in TestingData:
		tempt = DataManager() 
		tempt.insertData(DataElement)
		line = re.sub("https?://[\w\-]+(\.[\w\-]+)+\S*", " ", DataElement["tweet"])
		tempt.group = float('{0}'.format(new_grocery.predict(line)))
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
		e = row.group
		dataScore.append([a,b,c,d,e])
	#print(dataScore, file=outfile)
	#outfile.close()
	predictions = model.predict(dataScore)
	#for i, prediction in enumerate(predictions):
	#	print('Predicted: %s, Target: %s' % (prediction, dataSentiment[i]))
	print('(test)R-squared: %.3f' % model.score(dataScore, dataSentiment)) #0.335
	rms = mean_squared_error(dataSentiment,predictions)
	print('RMSE: %.3f' % sqrt(rms)) #0.310
	print('MSE: %.3f' % rms) #0.096


if __name__ == "__main__":
	main()
		
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

def GroupValue(Value):
	value = float(Value)
	v = [1,0.8,0.6,0.4,0]
	if value > v[1]:
		return v[0]
	elif value < -v[1]:
		return -v[0] 
	elif value > v[2]:
		return v[1]
	elif value < -v[2]:
		return -v[1]  
	elif value > v[3]:
		return v[2]
	elif value < -v[3]:
		return -v[2]
	else:
		return v[4]	
				

def main():
	# Word Score List
	targetIn = {}
	targetDict = dict()
	with open('NTUSD-Fin/NTUSD_Fin_hashtag_v1.0.json', 'r', encoding='utf-8') as f:
		targetIn = json.load(f)
	N = len(targetIn)
	for i in range(N):
		word = "#" + targetIn[i]['token']
		targetDict[word] = targetIn[i]['market_sentiment']
	with open('NTUSD-Fin/NTUSD_Fin_word_v1.0.json', 'r', encoding='utf-8') as f:
		targetIn = json.load(f)
	N = len(targetIn)
	for i in range(N):
		word = targetIn[i]['token']
		targetDict[word] = targetIn[i]['market_sentiment']

	# Training File
	TrainingFile = open('training_set.json','r')
	TrainingData = json.load(TrainingFile)
	TrainingFile.close()
	DataList = []
	grocery = Grocery("Sample")
	train_src = [] 
	for DataElement in TrainingData:
		tempt = DataManager() 
		tempt.insertData(DataElement)
		tempt.group = GroupValue(tempt.sentiment)
		line = re.sub("https?://[\w\-]+(\.[\w\-]+)+\S*", " ", DataElement["tweet"])
		train_src.append((str(tempt.group),line))
		DataList.append(tempt)
	grocery.train(train_src)
	grocery.save()
    
	#outfile = open('train_1.txt', 'w', encoding='utf-8')   
	dataScore = []
	dataSentiment = []
	for row in DataList:
		dataSentiment.append([float(row.sentiment)])
		a = WordScore(row.tweet,targetDict)
		b = WordScore(row.snippet,targetDict)
		#c = TargetScore(row.tweet,row.target)
		#d = ValueScore(row.snippet)
		e = row.group
		dataScore.append([a,b,e])
	#print(dataScore, file=outfile)
	#outfile.close()
	model = LinearRegression()
	model.fit(dataScore, dataSentiment)
	
	# Testing File
	TestingFile = open('test_set.json','r')
	TestingData = json.load(TestingFile)
	TestingFile.close()
	DataList = []
	new_grocery = Grocery('Sample')
	new_grocery.load()
	for DataElement in TestingData:
		tempt = DataManager() 
		tempt.insertData(DataElement)
		line = re.sub("https?://[\w\-]+(\.[\w\-]+)+\S*", " ", DataElement["tweet"])
		tempt.group = float('{0}'.format(new_grocery.predict(line)))
		DataList.append(tempt)
    
	#outfile = open('test_1.txt', 'w', encoding='utf-8')    
	dataScore = []
	dataSentiment = []
	for row in DataList:
		dataSentiment.append([float(row.sentiment)])
		a = WordScore(row.tweet,targetDict)
		b = WordScore(row.snippet,targetDict)
		#c = TargetScore(row.tweet,row.target)
		#d = ValueScore(row.snippet)
		e = row.group
		dataScore.append([a,b,e])
	#print(dataScore, file=outfile)
	#outfile.close()
	predictions = model.predict(dataScore)
	#for i, prediction in enumerate(predictions):
	#	print('Predicted: %s, Target: %s' % (prediction, dataSentiment[i]))
	print('R-squared: %.3f' % model.score(dataScore, dataSentiment)) #0.332
	rms = mean_squared_error(dataSentiment,predictions)
	print('RMSE: %.3f' % sqrt(rms)) #0.311
	print('MSE: %.3f' % rms) #0.097


if __name__ == "__main__":
	main()
