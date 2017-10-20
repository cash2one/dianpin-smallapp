
# -*- coding: utf-8 -*-

# In[1]:

import numpy as np
import re
import math
import io

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM,Embedding,Dropout
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import keras
import random
from keras import backend as K

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import sys
print(sys.getdefaultencoding())


# ## 一些文字处理函数

# In[2]:

def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    p_size = len(text)
    i = 0
    
    while i+window_size < p_size:
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])
        i += step_size
    return inputs,outputs

def encode_io_pairs(text,window_size,step_size,chars_to_indices):
    # number of unique chars
    chars = sorted(list(set(text)))
    num_chars = len(chars)
    
    # cut up text into character input/output pairs
    inputs, outputs = window_transform_text(text,window_size,step_size)
    
    # create empty vessels for one-hot encoded input/output
    X = np.zeros((len(inputs), window_size, num_chars), dtype=np.bool)
    y = np.zeros((len(inputs), num_chars), dtype=np.bool)
    
    # loop over inputs/outputs and transform and store in X/y
    for i, sentence in enumerate(inputs):
        for t, char in enumerate(sentence):
            X[i, t, chars_to_indices[char]] = 1
        y[i, chars_to_indices[outputs[i]]] = 1
        
    return X,y


# ## 文本处理




# In[46]:

def predict_next_chars(text,model,num_to_predict,window_size,chars_to_indices,indices_to_chars,chars):     
    # create output
    predicted_chars = ''
    
    back_texts = text.split("。")
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
            if w not in predicted_chars[-3:] and input_chars[-2:]+w in text:
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
        
        if input_chars[-1] == '。' and len(input_chars) > 25:
            break
            
    return head_text + predicted_chars





# # 第二个model只针对三个词

# In[13]:



# In[16]:


def test_output():

	with io.open("./resources/dict.txt", "r", encoding="utf-8") as my_file:
		 text = my_file.read() 
	my_file.close()


	# In[4]:

	print('our original text has ' + str(len(text)) + ' characters')
	
	
    
	# In[5]:

	text = text.replace(u'\xa0','')
	text = text.replace(u'    ','')
	text = text.replace(u'\n','')


	# In[6]:

	#去除emoj表情和一些很奇怪的隐藏符号
	text = "".join(re.findall(u'[a-z0-9\w\u2E80-\u9FFF，。！？\-\\.\!\?\;\:\'\#]*',text))




	# In[8]:

	text = text[int(math.ceil(len(text)*0.75)):]
	print(len(text))


	# In[9]:

	chars = sorted(list(set(text)))
	print ("this corpus has " +  str(len(chars)) + " unique characters")
	# this dictionary is a function mapping each unique character to a unique integer
	chars_to_indices = dict((c, i) for i, c in enumerate(chars))  # map each unique character to unique integer

	# this dictionary is a function mapping each unique integer back to a unique character
	indices_to_chars = dict((i, c) for i, c in enumerate(chars))  # map each unique integer back to unique character



	window_size = 7
	step_size = 2
	X,y = encode_io_pairs(text,window_size,step_size,chars_to_indices)



	model = Sequential()
	model.add(LSTM(128, input_shape=(window_size,len(chars))))
	model.add(Dense(len(chars),activation='softmax'))

	optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=1e-06)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer)
	
	model.load_weights('./resources/best_RNN_textdata_weights.hdf5')


	#final Output
	return(predict_next_chars(text,model,30,window_size,chars_to_indices,indices_to_chars,chars))
	
	
	
	