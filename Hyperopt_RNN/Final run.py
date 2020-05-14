import numpy as np
import os
import argparse
import pandas as pd
from PCA_for_RNN import PCA_fitting, PCA_predict
from Autoencoder_RNN import AE_fitting, AE_predict
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras import models, callbacks
from keras.layers import Dense, GRU, Dropout, Flatten, Concatenate, Input, LSTM, Embedding, Reshape
from keras import regularizers
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from keras.models import Model
from keras.utils import to_categorical
from keras import optimizers
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


from Preprocessing.LoadDataRNN import load_data_rnn

import os
os.chdir('/home/loratech/PycharmProjects/HKP_ML_DL/Hyperopt_LightGBM')



def rounding(x):

    x_below = np.logical_and(x > -0.5, x < 0.5)
    x_mid = np.logical_or(np.logical_and(x > 0.5, x < 1), np.logical_and(x < -0.5, x > -1))
    x_upper = np.logical_or(x > 1, x < -1)
    x_upper2 = x > 3
    x_upper3 = x < -3

    rounding_precision = 0.25
    x[x_below] = (x[x_below] * (10 * rounding_precision)).round(0) / (10 * rounding_precision)  # round to nearest 0.05
    x[x_mid] = (x[x_mid] * (5 * rounding_precision)).round(0) / (5 * rounding_precision)  # round to nearest 0.1
    x[x_upper] = (x[x_upper] * (2 * rounding_precision)).round(0) / (2 * rounding_precision)  # round to nearest 0.25%
    x[x_upper2] = 3
    x[x_upper3] = -3

    return x

def Dimension_reduction(x_train, x_valid, x_test, y_train, y_valid, y_test):



    AE_model = AE_fitting(x_train, 80)
    compressed_x_train = AE_predict(x_train, AE_model)
    compressed_x_valid = AE_predict(x_valid, AE_model)
    compressed_x_test = AE_predict(x_test, AE_model)
    print('x_train shape after AE:', compressed_x_train.shape)

    return compressed_x_train, compressed_x_valid, compressed_x_test, y_train, y_valid, y_test


def RNN(x_train, x_test, y_train, y_test):


    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3)
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = Dimension_reduction(x_train, x_valid, x_test, y_train, y_valid, y_test)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    # split the model - need FUNCTIONAL
    input_shape = (X_train.shape[1], X_train.shape[2])
    input_img = Input(shape=input_shape)
    embedding = Dense(50, activation='tanh')(input_img)
    embedding = Dropout(0.3)(embedding)
    gru1 = GRU(64, return_sequences=True)(embedding)
    gru2 = GRU(1, return_sequences=True)(gru1)  # returns the sequence - input is first GRU output - ONE node bc we just want 1X20 output
    gru2 = Flatten()(gru2)
    gru1 = GRU(1, return_sequences=False)(gru1)  # returns the hidden state forecast from first GRU output
    comb = Concatenate(axis=1)([gru2, gru1])  # combine the guess (gru1) with the sequence of hidden nodes (gru2)
    comb = Dense(21, activation='tanh')(comb)  # that combined vector goes through a Dense layer - "keeps" some time seq
    comb = Dense(3, activation='softmax')(comb)  # softmax choose
    model = Model(input_img, comb)
    model.summary()

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)


    model.compile(optimizer='RMSprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train,
              Y_train,
              epochs=30,
              batch_size=256,
              validation_data=(X_valid, Y_valid),
              callbacks=[reduce_lr],
              verbose=1)



    loss_train, accuracy_train = model.evaluate(X_train, Y_train, verbose=1)
    loss_valid, accuracy_valid = model.evaluate(X_valid, Y_valid, verbose=1)
    loss_test, accuracy_test = model.evaluate(X_test, Y_test, verbose=1)

    print(accuracy_train, accuracy_valid, accuracy_test)


    return accuracy_train, accuracy_valid, accuracy_test


if __name__ == "__main__":


    records = pd.DataFrame()
    sql_result = {'time': 1}

    sample_class = load_data_rnn(lag_year=5, sql_version=False)

    for i in range(1, 39):  # set = 40 if return 40 samples
        samples_set_train = sample_class.sampling(i, y_type='qoq')
        samples_set_test = sample_class.sampling(i+20, y_type='qoq')
        x_train = samples_set_train['x']
        y_train = samples_set_train['y']
        y_train = np.array(y_train)
        x_test = samples_set_test['x']
        y_test = samples_set_test['y']
        y_test = np.array(y_test)
        x_train = rounding(x_train)
        x_test = rounding(x_test)
        records.loc[i, 'accuracy_train'], records.loc[i, 'accuracy_valid'], records.loc[i, 'accuracy_test'] = RNN(x_train, x_test, y_train, y_test)
        records.to_csv('final record fantasy2.csv')

