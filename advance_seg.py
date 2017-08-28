
import keras

from keras.layers import Input,Dense,TimeDistributed,LSTM,Masking,Bidirectional
from keras.models import Model
print("Import finished.\n")

import numpy as np
MAXLEN = 60
NUM_POS = 24
SentenceVector = Input(shape=(MAXLEN,52),dtype='float32',name = 'sentence_input')

pos_mask = Masking(input_shape=(MAXLEN,52),mask_value = 0.0)(SentenceVector)

Pos_BLSTM1 = Bidirectional(LSTM(52*2,return_sequences=True,dropout=0.2))(pos_mask)

Pos_BLSTM2 = Bidirectional(LSTM(52*2,return_sequences=True, dropout=0.2))(Pos_BLSTM1)

Seg_dir_ = TimeDistributed(Dense(52*2*2,activation='relu'))(Pos_BLSTM2)

Seg_Output = TimeDistributed(Dense(4,activation='softmax',name='segment_output'))(Seg_dir_)

POS_dir_ = TimeDistributed(Dense(52*2*3,activation='relu'))(Pos_BLSTM2)

POS_Output = TimeDistributed(Dense(NUM_POS,activation='softmax',name='POS_output'))(POS_dir_)

advanced_segmodel = Model(inputs=[SentenceVector], outputs=[Seg_Output,POS_Output])
advanced_segmodel.compile(optimizer='adam', loss='binary_crossentropy',loss_weights=[0.5,0.5],sample_weight_mode='temporal',metrics=['accuracy'])

print("Finished compiling the double network")