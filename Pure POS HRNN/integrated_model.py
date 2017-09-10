#coding:utf-8
import keras
from keras.models import Model,Sequential
from keras.layers import Input,Dense,LSTM,Bidirectional,TimeDistributed,Masking
from keras.preprocessing.sequence import pad_sequences
import numpy as np
MAXCHAR_PER_WORD = 8
MAXWORD_PER_SENTENCE = 22
WORD_VEC = 52
POS_NUM = 26
#Input shape (Batch_size,MAXWORD_PER_SENTENCE,MAXCHAR_PER_WORD,WORD_VEC)
def HBLSTM4POS(maxword_per_sen=20,maxchar_per_word=8,word_vec_dim=52,pos_num=26):
    InputLayers = Input(shape=(maxword_per_sen,maxchar_per_word,word_vec_dim),name='InputTensor')
    Posmask = TimeDistributed(Masking(mask_value=0.0,input_shape=(8,52)),input_shape=(20,8,52))(InputLayers)
    WordLayer = TimeDistributed(Bidirectional(LSTM(52,return_sequences=False,input_shape=(8,52),name='WordVector')),input_shape=(20,8,52))(Posmask)
    #WordLayer = TimeDistributed(Bidirectional(LSTM(52,return_sequences=False,dropout=0.1,name='WordVector'),merge_mode='sum'))(Posmask)
    #Desired Current Shape (batch_size,MAXWORD_PER_SENTENCE,52)
    POS_LSTM1 = Bidirectional(LSTM(52,return_sequences=True))(WordLayer)
    POS_LSTM2 = Bidirectional(LSTM(52,return_sequences=True))(POS_LSTM1)
    Dense1 = TimeDistributed(Dense(POS_NUM*3,activation='relu'))(POS_LSTM2)
    Dense2 = TimeDistributed(Dense(POS_NUM,activation='softmax',name='POS_Output'))(Dense1)
    model = Model(inputs=InputLayers, outputs=Dense2)
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return model
'''
model = Sequential()
model.add(TimeDistributed(Masking(mask_value=0.0,input_shape=(8,52)),input_shape=(20,8,52)))
model.add(TimeDistributed(Bidirectional(LSTM(52,input_shape=(8,52)))))
model.add(Bidirectional(LSTM(52,return_sequences=True)))
model.add(TimeDistributed(Dense(26,activation='softmax')))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
'''
train_data = np.random.rand(100,20,8,52)
train_y = np.random.randint(26,size=(100,20,26))
model = HBLSTM4POS()
model.fit(train_data,train_y,batch_size=10,epochs=2,validation_split=0.1)