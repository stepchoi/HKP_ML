import numpy as np
import os
import argparse
import pandas as pd
from PCA_for_RNN import PCA_fitting, PCA_predict
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras import models, callbacks
from keras.layers import Dense, GRU, Dropout, Flatten
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

from Preprocessing.LoadDataRNN import load_data_rnn

space = {
    # dimension
    'reduced_dimension' : hp.choice('reduced_dimension', np.arange(0.75)), # past: [508, 624, 757]
    # number of layers
    'num_GRU_layer': hp.choice('num_GRU_layer', [1]),
    'num_Dense_layer': hp.choice('num_Dense_layer', [1]),

    'verbosity': -1,

    # hyperparameter
    'batch_size': hp.choice('batch_size', [512]),
}

import os
#os.chdir('/home/loratech/PycharmProjects/HKP_ML_DL/Hyperopt_LightGBM')
sample_class = load_data_rnn(lag_year=5, sql_version=True)

# for i in range(1):  # set = n if return 40 samples
samples_set1 = sample_class.sampling(1, y_type='qoq') # the first sample set -> include 80 quarter's samples -> x(3d), y(categorical)

sql_result = {'time': 1}

x = samples_set1['x'][0]
y = samples_set1['y'][0]
# print(x.shape, y.shape)

for q in range(39):
    x = np.concatenate((x, samples_set1['x'][q+1]))
    y = np.concatenate((y, samples_set1['y'][q+1]))
print(x.shape, y.shape)

def Dimension_reduction(reduced_dimensions, dimension_reduction_method='PCA', valid_method='shuffle'):


    if (valid_method == 'shuffle'):
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.25)


    if (dimension_reduction_method == 'PCA'):
        PCA_model = PCA_fitting(x_train, reduced_dimensions)
        compressed_x_train = PCA_predict(x_train, PCA_model)
        compressed_x_valid = PCA_predict(x_valid, PCA_model)
        #compressed_x_test = PCA_predict(x_test, PCA_model)
        print('reduced_dimensions:', reduced_dimensions)
        print('x_train shape after PCA:', compressed_x_train.shape)
    else:
        compressed_x_train = x_train
        compressed_x_valid = x_valid
        #compressed_x_test = x_test

    return compressed_x_train, compressed_x_valid, y_train, y_valid


def RNN(space):

    X_train, X_valid, Y_train, Y_valid = Dimension_reduction(space['reduced_dimension'], dimension_reduction_method='None', valid_method='shuffle')
    print(X_train.shape, X_valid.shape, Y_train.shape, Y_valid.shape,)

    model = models.Sequential()
    model.add(GRU(4, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences = False))
    model.add(Dense(units=21, activation='tanh'))
    model.add(Dense(3, activation='softmax'))
    model.summary()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train,
              Y_train,
              epochs=200,
              batch_size=space['batch_size'],
              validation_data=(X_valid, Y_valid),
              verbose=1)

    loss_train, accuracy_train = model.evaluate(X_train, Y_train, space['batch_size'], verbose=1)
    loss_valid, accuracy_valid = model.evaluate(X_valid, Y_valid, space['batch_size'], verbose=1)
    #loss_valid, accuracy_test = model.evaluate(X_test, Y_test, space['batch_size'], verbose=1)

    return accuracy_train, accuracy_valid


def f(space):

    accuracy_train, accuracy_valid = RNN(space)

    result = {'loss': 1 - accuracy_valid,
              'accuracy_train': accuracy_train,
              'accuracy_valid': accuracy_valid,
              #'accuracy_test': accuracy_test,
              'space': space,
              'status': STATUS_OK}

    print(space)
    print(result)
    #sql_result.update(result)
    #sql_result.pop('space')
    #sql_result.update(space)

    #db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    #engine = create_engine(db_string)
    #pd.DataFrame.from_records([sql_result]).to_sql('rnn_results', con=engine, if_exists='append')


    return result


if __name__ == "__main__":


    trials = Trials()
    best = fmin(fn=f, space=space, algo=tpe.suggest, max_evals=3,
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
    records.to_csv('records.csv')
    print(best)
