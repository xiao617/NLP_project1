# coding:utf-8
import json
import sys
class DataManager:
	def __init__(self):
		self.tweet = []
		self.target = ""
		self.snippet = ""
		self.sentiment = 0

	def insertData(self,DataList):
		testyoy = DataList["tweet"].split(" ")
		self.target = DataList["target"]
		self.snippet = DataList["snippet"]
		self.sentiment = DataList["sentiment"]

		for testline in testyoy:
			if testline !="":
				if testline[0] == "$":
					if len(testline) > 1:
						if testline[1]>="0" and testline[1]<="9":
							self.tweet.append(testline)
				elif len(testline)>4:
					if testline[0:4] == "http":
						continue
					else:
						self.tweet.append(testline)
				else:
					self.tweet.append(testline)

def main():
	TrainingFile = open('training_set.json','r')
	TrainingData = json.load(TrainingFile)
	TrainingFile.close()
	#OT = open('cut.txt','w')


	targetList = {}
	targetIn = []
	DataList = []

	for DataElement in TrainingData:
		tempt = DataManager() 
		tempt.insertData(DataElement)
		DataList.append(tempt)
		#print(tempt.tweet)

		'''
		if tempt.target in targetIn:
			targetList[tempt.target] += 1
		else:
			targetList[tempt.target] = 1 
			targetIn.append(tempt.target) 
	for nametar in targetIn:
		print(nametar,targetList[nametar])
		'''
	outputJson = []
	for row in DataList:
		DataDict = {}
		DataDict["tweet"]=row.tweet
		DataDict["target"]=row.target
		DataDict["sentiment"]=row.sentiment
		DataDict["snippet"]=row.snippet
		outputJson.append(DataDict)
	Jsonout = json.dumps(outputJson)
	jout = open('newtrain.json','w')
	jout.write(Jsonout)
	jout.close()
	#print(DataList[0])

if __name__ == "__main__":
	main()