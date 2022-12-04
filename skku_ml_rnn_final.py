# -*- coding: utf-8 -*-
"""skku_ml_rnn_.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JuWeoHTbcM0IX0JpNjNjD_ZOcoTfD40R
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Model
from keras.layers import SimpleRNN, Activation, Dense, Dropout, Input, Embedding, LSTM
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
# %matplotlib inline

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/train_is_immoral.csv',delimiter=',')

df.head()

test_csv = pd.read_csv('/content/valid_is_immoral.csv',delimiter=',')

X_train = df['sentence']
Y_train = df['label']  
X_test = test_csv['sentence']
Y_test = test_csv['label']

le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_train = Y_train.reshape(-1,1)

Y_test = le.fit_transform(Y_test)
Y_test = Y_test.reshape(-1,1)

max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix =tf.keras.utils.pad_sequences(sequences,maxlen=max_len)

def RNNmodel():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(32)(layer)
    layer = Dense(256,name='FC1',activation='relu')(layer)
    layer = Activation('relu')(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

import torch

!pip install torchmetrics

from torchmetrics.classification import BinaryStatScores
metric = BinaryStatScores()
for i in [0.01,0.015,0.02]:
  model = RNNmodel()
  for j in range(10):
    opt = keras.optimizers.Adam(learning_rate=i)
    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['acc',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
    history = model.fit(sequences_matrix,Y_train,batch_size=128,epochs=5, validation_split=0.2)
    max_words = 1000
    max_len = 150
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(X_test)
    sequences_t = tok.texts_to_sequences(X_test)
    sequences_matrix_t =tf.keras.utils.pad_sequences(sequences_t,maxlen=max_len)
    print('learning rate: ', i,", epochs: ", 5*j)
    model.evaluate(sequences_matrix_t, Y_test, batch_size=128)
    predict=model.predict(sequences_matrix_t,batch_size=128)
    results=metric(torch.Tensor(predict), torch.Tensor(Y_test))
    precision=results[0]/(results[0]+results[1])
    recall=results[0]/(results[0]+results[3])
    print(precision, recall, 2*precision*recall/(precision+recall),"\n")