#init_load.py
# -*- coding: utf-8 -*-


import numpy as np
import re
import math
import io
import gc

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM,Embedding,Dropout
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import keras
import random
from keras import backend as K

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class Singleton(object):
  
  def __new__(cls, *args, **kwargs):
    if not hasattr(cls, '_instance'):
      orig = super(Singleton, cls)
      cls._instance = orig.__new__(cls, *args, **kwargs)
    return cls._instance


class Dianpin(Singleton):
	def __init__(self):
		self.text = ''
		self.chars_to_indices = None
		self.indices_to_chars = None
		self.chars = None
		self.model = None
	
	def load_text(self):
		#load initial text

		with io.open("./resources/dict.txt", "r", encoding="utf-8") as my_file:
			self.text = my_file.read() 
		my_file.close()

	def process_text(self):
	#preprocess the text
		self.text = self.text.replace(u'\xa0','')
		self.text = self.text.replace(u'    ','')
		self.text = self.text.replace(u'\n','')
		self.text = "".join(re.findall(u'[a-z0-9\w\u2E80-\u9FFF，。！？\-\\.\!\?\;\:\'\#]*',self.text))		
		self.text = self.text[int(math.ceil(len(self.text)*0.75)):]
	
	def index_text(self):
		
		self.chars = sorted(list(set(self.text)))
		
		# this dictionary is a function mapping each unique character to a unique integer
		self.chars_to_indices = dict((c, i) for i, c in enumerate(self.chars))  # map each unique character to unique integer
		
		# this dictionary is a function mapping each unique integer back to a unique character
		self.indices_to_chars = dict((i, c) for i, c in enumerate(self.chars))  # map each unique integer back to unique character
	

	#def window_transform_text(self, text, window_size, step_size):
	#	# containers for input/output pairs
	#	inputs = []
	#	outputs = []
	#
	#	p_size = len(text)
	#	i = 0
	#
	#	while i+window_size < p_size:
	#		inputs.append(text[i:i+window_size])
	#		outputs.append(text[i+window_size])
	#		i += step_size
	#	return inputs,outputs
#
#	def encode_io_pairs(self,text,window_size,step_size,chars_to_indices):
		# number of unique chars
#		chars = sorted(list(set(text)))
#		num_chars = len(chars)
#	
#		# cut up text into character input/output pairs
#		inputs, outputs = self.window_transform_text(text,window_size,step_size)
#	
#		# create empty vessels for one-hot encoded input/output
#		X = np.zeros((len(inputs), window_size, num_chars), dtype=np.bool)
#		y = np.zeros((len(inputs), num_chars), dtype=np.bool)
#	
#		# loop over inputs/outputs and transform and store in X/y
#		for i, sentence in enumerate(inputs):
#			for t, char in enumerate(sentence):
#				X[i, t, chars_to_indices[char]] = 1
#			y[i, chars_to_indices[outputs[i]]] = 1
#
#		return X,y
		
	
	def model_built(self):
		#create model
		self.load_text()
		self.process_text()
		self.index_text()
		
		window_size = 7
	
		self.model = Sequential()
		self.model.add(LSTM(128, input_shape=(window_size,len(self.chars))))
		self.model.add(Dense(len(self.chars),activation='softmax'))

		optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=1e-06)
		self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
		self.model.load_weights('./resources/best_RNN_textdata_weights.hdf5')
		
	
	def predict_next_chars(self,model,num_to_predict,window_size,chars_to_indices,indices_to_chars,chars):     
		# create output
		predicted_chars = ''
	
		back_texts = self.text.split("。")
		input_chars = random.choice(back_texts)
	
		while len(input_chars) != 7:
			if len(input_chars) > 7:
				input_chars = input_chars[:7]
			elif len(input_chars) < 7:
				input_chars = random.choice(back_texts)
			
		head_text = input_chars  
	
		for i in range(num_to_predict):
			# convert this round's predicted characters to numerical input    
			x_test = np.zeros((1, window_size, len(chars)))
			for t, char in enumerate(input_chars):
				x_test[0, t, chars_to_indices[char]] = 1.

			# make this round's prediction
			test_predict = model.predict(x_test,verbose = 0)[0]
			top_words = []
			n = 0
			while n<5:
				r = np.argmax(test_predict)
				d = indices_to_chars[r]
				top_words.append(d)
				test_predict = test_predict[test_predict != np.max(test_predict)]
				n+=1
			
			found = False
			for w in top_words:
				if w not in predicted_chars[-3:] and input_chars[-2:]+w in self.text:
					predicted_chars += w
					input_chars+=w
					input_chars = input_chars[1:]    
					found = True
					break
		
			if not found: 
				#print("not found")
				predicted_chars += top_words[0]
				input_chars += top_words[0]
				input_chars = input_chars[1:]
		
			if input_chars[-1] == ',' and len(input_chars) > 50:
				break
			
		return head_text + predicted_chars
		
	
	
	def final_predict(self):
		return self.predict_next_chars(self.model,60,7,self.chars_to_indices,self.indices_to_chars,self.chars)
		gc.collect()
