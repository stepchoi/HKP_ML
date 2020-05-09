import numpy as np
import os
import argparse
import pandas as pd
from PCA_for_RNN import PCA_fitting, PCA_predict
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras import models, callbacks
from keras.layers import Dense, GRU, Dropout, Flatten, Concatenate, Input
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sqlalchemy import create_engine
from keras.models import Model
from keras import optimizers

from Preprocessing.LoadDataRNN import load_data_rnn

space = {
    # dimension
    'reduced_dimension' : hp.choice('reduced_dimension', np.arange(0.5, 0.75, 1)),
    # num_of_neuron
    'neuron' : hp.choice('neuron', [2, 4, 8, 16]),

    'quarter' : hp.choice('quarter', [5, 10, 20]),
    'lr' : hp.choice('lr', [0.1, 0.01, 0.001]),

    # hyperparameter
    'batch_size': hp.choice('batch_size', [1000, 2000]),
}

records = pd.DataFrame()

import os
os.chdir('/home/loratech/PycharmProjects/HKP_ML_DL/Hyperopt_LightGBM')

sql_result = {'time': 1}

sample_class = load_data_rnn(lag_year=5, sql_version=False)

for i in range(1):  # set = 40 if return 40 samples
    samples_set1 = sample_class.sampling(i, y_type='qoq')

x_full = samples_set1['x']
y = samples_set1['y']
y = np.array(y)




def Dimension_reduction(quarter, reduced_dimensions, dimension_reduction_method='None', valid_method='shuffle'):

    x = x_full[:, 0:quarter, :]
    x_new = x.copy()
    for i in range(len(x_new)):
        for j in range(quarter):
            x_new[i][j] = x[i][quarter - 1 - j]
    x = x_new.copy()

    if (valid_method == 'shuffle'):
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)
        x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=0.25)

    if reduced_dimensions == 1:
        dimension_reduction_method = 'None'

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


def RNN(space):

    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = Dimension_reduction(space['quarter'], space['reduced_dimension'], dimension_reduction_method='PCA', valid_method='shuffle')
    print(X_train.shape, X_valid.shape, Y_train.shape, Y_valid.shape)

    # model = models.Sequential()
    # model.add(GRU(4, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences = True))
    # model.add((GRu))
    # model.add(Dense(units=21, activation='tanh'))
    # model.add(Dense(3, activation='softmax'))

    # split the model - need FUNCTIONAL
    input_shape = (X_train.shape[1], X_train.shape[2])
    input_img = Input(shape=input_shape)
    gru1 = GRU(space['neuron'], return_sequences = True) (input_img)
    gru2 = GRU(1, return_sequences=True)(gru1)  #returns the sequence - input is first GRU output - ONE node bc we just want 1X20 output
    gru2 = Flatten()(gru2)
    gru1 = GRU(space['neuron'], return_sequences = False)(gru1) #returns the hidden state forecast from first GRU output
    comb = Concatenate(axis=1)([gru2, gru1]) #combine the guess (gru1) with the sequence of hidden nodes (gru2)
    comb = Dense(space['quarter']+1, activation='tanh')(comb) #that combined vector goes through a Dense layer - "keeps" some time seq
    comb = Dense(3, activation='softmax')(comb) #softmax choose
    model = Model(input_img, comb)
    model.summary()

    adam = optimizers.adam(lr=space['lr'])

    model.compile(optimizer=adam,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train,
              Y_train,
              epochs=100,
              batch_size=space['batch_size'],
              validation_data=(X_valid, Y_valid),
              verbose=1)

    loss_train, accuracy_train = model.evaluate(X_train, Y_train, space['batch_size'], verbose=1)
    loss_valid, accuracy_valid = model.evaluate(X_valid, Y_valid, space['batch_size'], verbose=1)
    loss_test, accuracy_test = model.evaluate(X_test, Y_test, space['batch_size'], verbose=1)

    return accuracy_train, accuracy_valid


def f(space):

    accuracy_train, accuracy_valid = RNN(space)

    result = {'loss': 1 - accuracy_valid,
              'accuracy_train': accuracy_train,
              'accuracy_valid': accuracy_valid,
              #'accuracy_test': accuracy_test,
              'space': space,
              'status': STATUS_OK}
    loss = 1 - accuracy_valid
    print(space)
    print(result)
    row = len(records)
    for i in result['space'].keys():
        records.loc[row, i] = result['space'][i]
    for i in result.keys():
        if i != 'space':
            records.loc[row, i] = result[i]

    return loss


if __name__ == "__main__":
    row = 0


    trials = Trials()
    best = fmin(fn=f, space=space, algo=tpe.suggest, max_evals=20,
                trials=trials)
    records.to_csv('records.csv')


    print(records)
