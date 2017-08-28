#coding:utf-8
import keras
from keras.layers import Input,Dense,TimeDistributed,LSTM,Masking,Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import os,sys

def NetworkInit(POSNum,maxlength=60):
    SentenceVector = Input(shape=(maxlength,52),dtype='float32',name = 'SentenceInput')
    pos_mask = Masking(input_shape=(maxlength,52),mask_value = 0.0)(SentenceVector)
    BLSTM1 = Bidirectional(LSTM(52*2,return_sequences=True,dropout=0.2))(pos_mask)
    BLSTM2 = Bidirectional(LSTM(52*2,return_sequences=True, dropout=0.2))(BLSTM1)
    Seg_dense = TimeDistributed(Dense(52*2*2,activation='relu'))(BLSTM2)
    Seg_Output = TimeDistributed(Dense(4,activation='softmax',name='segment_output'))(Seg_dense)
    POS_dense = TimeDistributed(Dense(52*2*3,activation='relu'))(BLSTM2)
    POS_Output = TimeDistributed(Dense(POSNum,activation='softmax',name='POS_output'))(POS_dense)
    model = Model(inputs=[SentenceVector], outputs=[Seg_Output,POS_Output])
    model.compile(optimizer='adam', loss='binary_crossentropy',loss_weights=[0.5,0.5],sample_weight_mode='temporal',metrics=['accuracy'])
    return model
def TrainNetwork(SegModel,XArrayFileDir,SegArrayFileDir,POSArrayFileDir,batchsize,Tepochs,pad = True,maxlength=60):
    #we assume the Xarray has a shape (M,len,52)
    #SegArray has a shape (M,len,4) for (B,M,E,S)
    #POSArray has a shape (M,len,POSnum)
    #We would like to train a network that predicts the segment tag
    #and the pos_tag at the same time.
    #For the pre-processing the POS-tag should be only on the E/S tag 
    #.i.e. at the end of the word (including a single character word)
    #This function does not provide any more further preprocessing,
    #except the pad_sequences
    x_filelist = os.listdir(XArrayFileDir)
    y1_filelist = os.listdir(SegArrayFileDir)
    y2_filelist = os.listdir(POSArrayFileDir)
    x_filenum = len(x_filelist)
    y1_filenum = len(y1_filelist)
    y2_filenum = len(y2_filelist)
    if not (x_filenum==y1_filenum==y2_filenum):
        print("Current Directory File incomplete.\n")
        print("File validation check failed.\n")
        return -1
    for epoch_time in range(Tepochs):
        for i in range(x_filenum):
            xarray = np.load('X'+str(i)+'.npy')
            seg_array = np.load('Y1_'+str(i)+'.npy')
            pos_array = np.load('Y2_'+str(i)+'.npy')
            if pad:
                xarray = pad_sequences(xarray,maxlen=maxlength,lenpadding='pre',truncating='pre',dtype='float32',value=0.0)
                seg_array = pad_sequences(seg_array,maxlen=maxlength,padding='pre',truncating='pre',dtype='float32',value=0.0)
                pos_array = pad_sequences(pos_array,maxlen=maxlength,padding='pre',truncating='pre',dtype='float32',value=0.0)
            current_sample_number = xarray.shape[0]
            sample_ptr = 0 
            while sample_ptr + batchsize < current_sample_number:
                Xarr = xarray[sample_ptr:sample_ptr+batchsize,:,:]
                Y1arr = seg_array[sample_ptr:sample_ptr+batchsize,:,:]
                Y2arr = pos_array[sample_ptr:sample_ptr+batchsize,:,:]
                weight_y1 = np.zeros((Xarr.shape[0],Y1arr.shape[1]), dtype=int)
                weight_y2 = np.zeros((Xarr.shape[0],Y2arr.shape[1]), dtype=int)
                for y_index1 in range(Y1arr.shape[0]):
                    for y_index2 in range(Y1arr.shape[1]):
                        if np.any(Y1arr[y_index1][y_index2]):
                            weight_y1[y_index1][y_index2]=1.0
                        weight_y2[y_index1][y_index2]=1.0
                SegModel.fit([Xarr],[Y1arr,Y2arr],epochs=1,batch_size=batchsize,verbose=1,validation_split=0.1,sample_weight=[weight_y1,weight_y2])

    


