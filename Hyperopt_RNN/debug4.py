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

sql_result = {'time': 1}

sample_class = load_data_rnn(lag_year=5, sql_version=False)

for i in range(1):  # set = 40 if return 40 samples
    samples_set1 = sample_class.sampling(i, y_type='qoq')

x = samples_set1['x']
y = samples_set1['y']
y = np.array(y)
records = pd.DataFrame()



space = {

    # rounding precision
    'rounding_precision' : hp.choice('rounding_precision', [0.25, 0.5, 1, 2, 4]),

    # dimension
    'reduced_dimension' : hp.choice('reduced_dimension', [50, 100]), # past: [508, 624, 757]
    # number of layers
    'num_GRU_layer': hp.choice('num_GRU_layer', [1, 2]),

    'verbosity': -1,

    # hyperparameter
    'neurons_GRU_layer': hp.choice('neurons_GRU_layer', [8, 32, 128]),
    'neurons_Dense_layer': hp.choice('neurons_Dense_layer', [25, 75, 150]),
    'dropout_dense': hp.choice('dropout_dense', [0, 0.25, 0.5]),
    'dropout_GRU': hp.choice('dropout_GRU', [0, 0.1, 0.2]),
    'batch_size': hp.choice('batch_size', [128, 256, 512]),
    'optimizer' : hp.choice('optimizer', ['adam', 'sgd', 'RMSprop'])
}



def Dimension_reduction(reduced_dimensions, rounding_precision, dimension_reduction_method='None', valid_method='shuffle'):

    x_below = np.logical_and(x > -0.5, x < 0.5)
    x_mid = np.logical_or(np.logical_and(x > 0.5, x < 1), np.logical_and(x < -0.5, x > -1))
    x_upper = np.logical_or(x > 1, x < -1)
    x_upper2 = x > 3
    x_upper3 = x < -3

    x[x_below] = (x[x_below] * (10 * rounding_precision)).round(0) / (10 * rounding_precision)  # round to nearest 0.05
    x[x_mid] = (x[x_mid] * (5 * rounding_precision)).round(0) / (5 * rounding_precision)  # round to nearest 0.1
    x[x_upper] = (x[x_upper] * (2 * rounding_precision)).round(0) / (2 * rounding_precision)  # round to nearest 0.25%
    x[x_upper2] = 3
    x[x_upper3] = -3


    if (valid_method == 'shuffle'):
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.4)

    if (dimension_reduction_method == 'PCA'):
        PCA_model = PCA_fitting(x_train, reduced_dimensions)
        compressed_x_train = PCA_predict(x_train, PCA_model)
        compressed_x_valid = PCA_predict(x_valid, PCA_model)
        print('reduced_dimensions:', reduced_dimensions)
        print('x_train shape after PCA:', compressed_x_train.shape)
    elif (dimension_reduction_method == 'AE'):
        AE_model = AE_fitting(x_train, reduced_dimensions)
        compressed_x_train = AE_predict(x_train, AE_model)
        compressed_x_valid = AE_predict(x_valid, AE_model)
        print('reduced_dimensions:', reduced_dimensions)
        print('x_train shape after PCA:', compressed_x_train.shape)

    return compressed_x_train, compressed_x_valid, y_train, y_valid


def RNN(space):



    X_train, X_valid, Y_train, Y_valid = Dimension_reduction(space['reduced_dimension'], space['rounding_precision'], dimension_reduction_method='AE', valid_method='shuffle')
    print(X_train.shape, X_valid.shape, Y_train.shape, Y_valid.shape)

    # split the model - need FUNCTIONAL
    input_shape = (X_train.shape[1], X_train.shape[2])
    input_img = Input(shape=input_shape)
    embedding = Dense(space['neurons_Dense_layer'], activation='tanh')(input_img)
    embedding = Dropout(space['dropout_dense'])(embedding)
    gru1 = GRU(space['neurons_GRU_layer'], return_sequences=True, dropout=space['dropout_GRU'])(embedding)
    if (space['num_GRU_layer'] == 2):
        gru1 = GRU(space['neurons_GRU_layer'], return_sequences = True, dropout = space['dropout_GRU'])(gru1)
    gru2 = GRU(1, return_sequences=True)(gru1)  # returns the sequence - input is first GRU output - ONE node bc we just want 1X20 output
    gru2 = Flatten()(gru2)
    gru1 = GRU(1, return_sequences=False)(gru1)  # returns the hidden state forecast from first GRU output
    comb = Concatenate(axis=1)([gru2, gru1])  # combine the guess (gru1) with the sequence of hidden nodes (gru2)
    comb = Dense(21, activation='tanh')(comb)  # that combined vector goes through a Dense layer - "keeps" some time seq
    comb = Dense(3, activation='softmax')(comb)  # softmax choose
    model = Model(input_img, comb)
    model.summary()

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='train_loss', factor=0.2, patience=3, min_lr=0.0001)


    model.compile(optimizer=space['optimizer'],
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train,
              Y_train,
              epochs=60,
              batch_size=space['batch_size'],
              validation_data=(X_valid, Y_valid),
              callbacks=[reduce_lr],
              verbose=1)



    loss_train, accuracy_train = model.evaluate(X_train, Y_train, verbose=1)
    loss_valid, accuracy_valid = model.evaluate(X_valid, Y_valid, verbose=1)
    #loss_test, accuracy_test = model.evaluate(X_test, Y_test, verbose=1)

    #Y_train_pred_softmax = model.predict(X_train)
    #Y_valid_pred_softmax = model.predict(X_valid)
    #Y_test_pred_softmax = model.predict(X_test)


    #Y_train_pred = [list(i).index(max(i)) for i in Y_train_pred_softmax]
    #Y_valid_pred = [list(i).index(max(i)) for i in Y_valid_pred_softmax]
    #Y_test_pred = [list(i).index(max(i)) for i in Y_test_pred_softmax]

    # Y_train_true = [list(i).index(max(i)) for i in Y_train]
    # #_valid_true = [list(i).index(max(i)) for i in Y_valid]
    #Y_test_true = [list(i).index(max(i)) for i in Y_test]
    # print(Y_valid_pred_softmax)

    #print(accuracy_score(Y_train_true, Y_train_pred), accuracy_score(Y_valid_true, Y_valid_pred), accuracy_score(Y_test_true, Y_test_pred))
    #print(accuracy_score(Y_test, Y_test_pred))

    print(loss_valid, accuracy_valid)
    #print(loss_train, accuracy_train, loss_valid, accuracy_valid, loss_test, accuracy_test)

    return accuracy_train, accuracy_valid

def f(space):

    accuracy_train, accuracy_valid = RNN(space)

    result = {'loss': 1 - accuracy_valid,
              'accuracy_train': accuracy_train,
              'accuracy_valid': accuracy_valid,
              #'accuracy_test': accuracy_test,
              'space': space,
              'status': STATUS_OK}

    records.loc[len(records), 'loss'] = 1 - accuracy_valid
    records.loc[len(records)-1, 'accuracy_train'] = accuracy_train
    records.loc[len(records)-1, 'accuracy_valid'] = accuracy_valid
    for i in result['space'].keys():
        records.loc[len(records)-1, i] = result['space'][i]
    records.to_csv('record5.csv')

    print(space)
    print(result)


    return result



if __name__ == "__main__":

    trials = Trials()
    best = fmin(fn=f, space=space, algo=tpe.suggest, max_evals=100,
                trials=trials)  # space = space for normal run; max_evals = 50

    records = pd.DataFrame()
    row = 0
    for record in trials.trials:
        print(record)
        for i in record['result']['space'].keys():
            records.loc[row, i] = record['result']['space'][i]
        record['result'].pop('space')
        for i in record['result'].keys():
            records.loc[row, i] = record['result'][i]
        row = row + 1
    records.to_csv('record6.csv')
    print(best)
