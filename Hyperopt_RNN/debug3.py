import numpy as np
import os
import argparse
import pandas as pd
from PCA_for_RNN import PCA_fitting, PCA_predict
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras import models, callbacks
from keras.layers import Dense, GRU, Dropout, Flatten, Concatenate, Input, LSTM
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from keras.models import Model
from keras.utils import to_categorical
from keras import optimizers
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


from Preprocessing.LoadDataRNN import load_data_rnn

import os
os.chdir('/home/loratech/PycharmProjects/HKP_ML_DL/Hyperopt_LightGBM')

sql_result = {'time': 1}

sample_class = load_data_rnn(lag_year=5, sql_version=False)

for i in range(1):  # set = 40 if return 40 samples
    samples_set1 = sample_class.sampling(i, y_type='qoq')

x = samples_set1['x']
x = x[:,0:20,:]
x_new = x.copy()
for i in range(len(x_new)):
    for j in range(20):
        x_new[i][j] = x[i][19-j]
x = x_new.copy()
y = samples_set1['y']
y = np.array(y)
y = to_categorical(y)


def Dimension_reduction(reduced_dimensions, dimension_reduction_method='None', valid_method='shuffle'):


    if (valid_method == 'shuffle'):
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)
        x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=0.25)


    if (dimension_reduction_method == 'PCA'):
        PCA_model = PCA_fitting(x_train, reduced_dimensions)
        compressed_x_train = PCA_predict(x_train, PCA_model)
        compressed_x_valid = PCA_predict(x_valid, PCA_model)
        compressed_x_test = PCA_predict(x_test, PCA_model)
        print('reduced_dimensions:', reduced_dimensions)
        print('x_train shape after PCA:', compressed_x_train.shape)
    else:
        compressed_x_train = x_train
        compressed_x_valid = x_valid
        compressed_x_test = x_test

    return compressed_x_train, compressed_x_valid, compressed_x_test, y_train, y_valid, y_test


def RNN():



    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = Dimension_reduction(0.5, dimension_reduction_method='PCA', valid_method='shuffle')
    print(X_train.shape, X_valid.shape, Y_train.shape, Y_valid.shape)

    # split the model - need FUNCTIONAL
    input_shape = (X_train.shape[1], X_train.shape[2])
    input_img = Input(shape=input_shape)
    embedding = Dense(200, activation='tanh')(input_img)
    gru1 = GRU(4, return_sequences = True)(embedding)
    gru2 = GRU(1, return_sequences=True)(gru1)  #returns the sequence - input is first GRU output - ONE node bc we just want 1X20 output
    gru2 = Flatten()(gru2)
    gru1 = GRU(1, return_sequences = False)(gru1) #returns the hidden state forecast from first GRU output
    comb = Concatenate(axis=1)([gru2, gru1]) #combine the guess (gru1) with the sequence of hidden nodes (gru2)
    comb = Dense(21, activation='tanh')(comb) #that combined vector goes through a Dense layer - "keeps" some time seq
    comb = Dense(3, activation='softmax')(comb) #softmax choose
    model = Model(input_img, comb)
    model.summary()

    adam = optimizers.adam(lr=0.001)

    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train,
              Y_train,
              epochs=200,
              batch_size=10000,
              validation_data=(X_valid, Y_valid),
              verbose=1)



    loss_train, accuracy_train = model.evaluate(X_train, Y_train, verbose=1)
    loss_valid, accuracy_valid = model.evaluate(X_valid, Y_valid, verbose=1)
    loss_test, accuracy_test = model.evaluate(X_test, Y_test, verbose=1)

    Y_train_pred_softmax = model.predict(X_train)
    Y_valid_pred_softmax = model.predict(X_valid)
    Y_test_pred_softmax = model.predict(X_test)

    print(Y_test_pred_softmax)
    print(Y_test)

    Y_train_pred = [list(i).index(max(i)) for i in Y_train_pred_softmax]
    Y_valid_pred = [list(i).index(max(i)) for i in Y_valid_pred_softmax]
    Y_test_pred = [list(i).index(max(i)) for i in Y_test_pred_softmax]

    Y_train_true = [list(i).index(max(i)) for i in Y_train]
    Y_valid_true = [list(i).index(max(i)) for i in Y_valid]
    Y_test_true = [list(i).index(max(i)) for i in Y_test]
    print(Y_valid_pred_softmax)

    print(accuracy_score(Y_train_true, Y_train_pred), accuracy_score(Y_valid_true, Y_valid_pred), accuracy_score(Y_test_true, Y_test_pred))


    print(loss_train, accuracy_train, loss_valid, accuracy_valid, loss_test, accuracy_test)

    return 0



if __name__ == "__main__":

    RNN()
