
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

# In[3]:



# In[28]:

def predict_next_chars_model2(model,input_chars,num_to_predict):     
    # create output
    predicted_chars = ''
    for i in range(num_to_predict):
        # convert this round's predicted characters to numerical input    
        x_test = np.zeros((1, window_size2, len(chars2)))
        for t, char in enumerate(input_chars):
            x_test[0, t, chars_to_indices2[char]] = 1.

        # make this round's prediction
        test_predict = model.predict(x_test,verbose = 0)[0]
        
        # translate numerical prediction back to characters
        r = np.argmax(test_predict)                           # predict class of each test input
        d = indices_to_chars2[r] 

        # update predicted_chars and input
        predicted_chars+=d
        input_chars+=d
        input_chars = input_chars[1:]
    return predicted_chars


# In[46]:

def predict_next_chars(model,num_to_predict,window_size,chars_to_indices,indices_to_chars,chars):     
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
            if w not in predicted_chars[-10:]:
                if input_chars[-1:]+w in text:
                    predicted_chars += w
                    input_chars+=w
                    input_chars = input_chars[1:]
                    
                else:
                    new_w = predict_next_chars_model2(model2,input_chars[-3:],1)
                    predicted_chars += new_w
                    input_chars += new_w
                    input_chars = input_chars[1:]
                    
                found = True
                break
        
        if not found: 
            predict_word = predict_next_chars(model,input_chars,1,window_size,chars_to_indices,indices_to_chars,chars)
            predicted_chars += predict_word
            input_chars += predict_word
            input_chars = input_chars[1:]
        
        if input_chars[-1] == '。' and len(input_chars) > 80:
            break
            
    return head_text + predicted_chars


# In[47]:




# # 第二个model只针对三个词

# In[13]:



# In[16]:




if __name__ == '__main__':
	
	
	
	with io.open("./resources/dict.txt", "r", encoding="utf-8") as my_file:
		 text = my_file.read() 
	my_file.close()


	# In[4]:

	print('our original text has ' + str(len(text)) + ' characters')

    text = unicode(text,'utf-8')
    
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
	print(str(text[:1000]))


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
	
	

	
	with io.open("./resources/dict.txt", "r", encoding="utf-8") as my_file:
		 text2 = my_file.read() 
	my_file.close()

	text2 = text2.replace(u'\xa0','')
	text2 = text2.replace(u'    ','')
	text2 = text2.replace(u'\n','')
	#去除emoj表情和一些很奇怪的隐藏符号
	text2 = "".join(re.findall(u'[a-z0-9\w\u2E80-\u9FFF，。！？\-\\.\!\?\;\:\'\#]*',text2))
	text2 = text2[:int(math.ceil(len(text2)*0.15))]
	print(len(text2))

	chars2 = sorted(list(set(text2)))
	print ("this corpus has " +  str(len(chars2)) + " unique characters")
	# this dictionary is a function mapping each unique character to a unique integer
	chars_to_indices2 = dict((c, i) for i, c in enumerate(chars2))  # map each unique character to unique integer

	# this dictionary is a function mapping each unique integer back to a unique character
	indices_to_chars2 = dict((i, c) for i, c in enumerate(chars2))  # map each unique integer back to unique character


	# In[14]:

	window_size2 = 3
	step_size2 = 1
	X2,y2 = encode_io_pairs(text2,window_size2,step_size2,chars_to_indices2)


	# In[15]:

	model2 = Sequential()
	model2.add(LSTM(128, input_shape=(window_size2,len(chars2))))
	#model.add(Dropout(0,2))
	model2.add(Dense(len(chars2),activation='softmax'))

	optimizer2 = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=1e-06)
	model2.compile(loss='categorical_crossentropy', optimizer=optimizer2)


	model.load_weights('./resources/best_RNN_textdata_weights.hdf5')
	model2.load_weights('./resources/best_RNN2_textdata_weights.hdf5')

	#final Output
	print(predict_next_chars(model,80,window_size,chars_to_indices,indices_to_chars,chars))
