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

def Value(x):
	v = 0.01
	if x > v:
		return 1.0
	elif x < -v:
		return -1.0
	else:
		return 0.0



def main():
	dataScore = []
	infile = open('TG_train.txt', 'r', encoding='utf-8')
	for line in infile:
		datalist = []
		num = line.split(" ")
		for k in num:
			datalist.append(float(k))
		dataScore.append(datalist)
	infile.close()
	
	infile = open('PMI_train.txt', 'r', encoding='utf-8')
	n=0
	for line in infile:
		datalist = []
		num = line.split(" ")
		for k in num:
			datalist.append(float(k))
		dataScore[n] += datalist
		n += 1
	infile.close()
	
	
	# Training File
	dataSentiment = []
	TrainingFile = open('training_set.json','r')
	TrainingData = json.load(TrainingFile)
	TrainingFile.close()
	for DataElement in TrainingData:
		dataSentiment.append([float(DataElement["sentiment"])])		

	model = LinearRegression()
	model.fit(dataScore, dataSentiment)
	print('(model)R-squared: %.3f' % model.score(dataScore, dataSentiment))
	predictions = model.predict(dataScore)
	rms = mean_squared_error(dataSentiment,predictions)
	print('RMSE: %.3f' % sqrt(rms))
	print('MSE: %.3f' % rms)

	
	dataScore = []
	infile = open('TG_test.txt', 'r', encoding='utf-8')
	for line in infile:
		datalist = []
		num = line.split(" ")
		for k in num:
			datalist.append(float(k))
		dataScore.append(datalist)
	infile.close()
	
	infile = open('PMI_test.txt', 'r', encoding='utf-8')
	n=0
	for line in infile:
		datalist = []
		num = line.split(" ")
		for k in num:
			datalist.append(float(k))
		dataScore[n] += datalist
		n += 1
	infile.close()


	# Testing File
	dataSentiment = []
	TestingFile = open('test_set.json','r')
	TestingData = json.load(TestingFile)
	TestingFile.close()
	for DataElement in TestingData:
		dataSentiment.append([float(DataElement["sentiment"])])

	predictions = model.predict(dataScore)
	x = []
	y = []
	for i, prediction in enumerate(predictions):
		y.append(Value(prediction[0]))
		x.append(Value(dataSentiment[i][0]))

	print('(test)R-squared: %.3f' % model.score(dataScore, dataSentiment)) 
	rms = mean_squared_error(dataSentiment,predictions)
	print('RMSE: %.3f' % sqrt(rms))
	print('MSE: %.3f' % rms)
	print('f1-micro: %.3f' % f1_score(x,y,average='micro'))
	print('f1-macro: %.3f' % f1_score(x,y,average='macro'))


if __name__ == "__main__":
	main()