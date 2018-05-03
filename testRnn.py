import sys, argparse, os
import keras
import _pickle as pk
import readline
import numpy as np
import json
from keras.models import load_model
from keras.models import Sequential
from keras import regularizers
from keras.models import Model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional,Activation
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
import keras.backend.tensorflow_backend as K
import tensorflow as tf
from keras.utils.data_utils import Sequence
class DataSequence(Sequence):
    def __init__(self,x_set,y_set,batch_size):
        self.batch_size = batch_size
        self.x,self.y=x_set,y_set
    def __len__(self):
        return len(self.y) // self.batch_size
    def __getitem__(self,idx):
        return self.x[idx*self.batch_size:(idx+1)*self.batch_size],self.y[idx*self.batch_size:(idx+1)*self.batch_size]
    def on_epoch_end(self):
        pass
    def __iter__(self):
        while True:
            for item in (self[i] for i in range(len(self))):
                yield item
def SimpleRnnLSTM(X_data,Y_data,X_test,Y_test):
	Batch_size = 32
	InputDim = 10
	model = Sequential()
	model.add(LSTM(output_dim=2048,batch_input_shape=(Batch_size,1,InputDim),return_sequences=True))
	#model.add(LSTM(output_dim=31,return_sequences=True))
	model.add(LSTM(output_dim=1024))
	model.add(Dense(512))
	model.add(Dense(256))
	model.add(Dense(128))
	model.add(Dense(32))
	model.add(Dense(3))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])#adam or nadam
	model.summary()
	checkpointer = ModelCheckpoint(filepath="./savemodel/nlpclass10_3.hdf5", verbose=1, save_best_only=True)
	his = model.fit_generator(DataSequence(X_data,Y_data,Batch_size),steps_per_epoch=1211,epochs=100,validation_data=DataSequence(X_test,Y_test,Batch_size),validation_steps=634,callbacks=[checkpointer])
	#print(his)
	with open('./outputlog#10.txt','w') as filelog:
		filelog.write(str(his.history))
	#model.save('simplebaseline.h5')
def datahandle(rawdata):
	X_data = []
	Y_data = []
	for row in rawdata:
		tt = []
		tt.append(row["feature"])
		X_data.append(tt)
		Y_data.append(row['sentiment'])
	return np.array(X_data),np.array(Y_data)
def testmodel(X_test,Y_test):
	model = load_model('./savemodel/nlpclass.hdf5')
	score = model.evaluate(X_test, Y_test, verbose=0,batch_size=2)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
def main():
	filein = open('Fsenttrain.json','r')
	rawtrain = json.load(filein)
	filet = open('Fsenttest.json','r')
	rawtest = json.load(filet)
	X_train,Y_train = datahandle(rawtrain)
	X_test,Y_test = datahandle(rawtest)
	Y_test = keras.utils.to_categorical(Y_test,num_classes=3)
	Y_train = keras.utils.to_categorical(Y_train,num_classes=3)
	testmodel(X_train,Y_train)
	#print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
	#SimpleRnnLSTM(X_train,Y_train,X_test,Y_test)
if __name__ == "__main__":
	main()