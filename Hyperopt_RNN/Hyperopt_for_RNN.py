import numpy as np
import os
import argparse
import pandas as pd
from PCA_for_RNN import PCA_fitting, PCA_predict
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras import models, callbacks
from keras.layers import Dense, GRU, Dropout, Flatten
from sklearn.model_selection import train_test_split

from Preprocessing.LoadDataRNN import load_data_rnn

space = {
    # dimension
    'reduced_dimension' : hp.choice('reduced_dimension', np.arange(0.66, 0.75, 0.85)), # past: [508, 624, 757]
    # number of layers
    'num_layer': hp.choice('num_layer', [1, 2, 3]),

    'verbosity': -1,

    # hyperparameter
    'neurons_GRU_layer_1': hp.choice('neurons_GRU_layer_1', [4, 8, 16]),
    'neurons_GRU_layer_2': hp.choice('neurons_GRU_layer_2', [4, 8, 16]),
    'neurons_GRU_layer_3': hp.choice('neurons_GRU_layer_3', [4, 8, 16]),
    'neurons_Dense_layer': hp.choice('neurons_Dense_layer_2', [16, 64]),
    'batch_size': hp.choice('batch_size', [512, 1024, 2048]),
    'dropout': hp.choice('dropout', [0, 0.2, 0.4])
}

sample_class = load_data_rnn(lag_year=5, sql_version=True)

for i in range(1):  # set = n if return 40 samples
    samples_set1 = sample_class.sampling(i, y_type='qoq') # the first sample set -> include 80 quarter's samples -> x(3d), y(categorical)

x = samples_set1['x'][0]
y = samples_set1['y'][0]


def Dimension_reduction(reduced_dimensions, dimension_reduction_method='PCA', valid_method='shuffle'):
    dimension_reduction_method = 'PCA'

    if (valid_method == 'shuffle'):
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.25, stratify=y)

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

    X_train, X_valid, Y_train, Y_valid = Dimension_reduction(space['reduced_dimension'])

    model = models.Sequential()
    model.add(GRU(space['neurons_GRU_layer_1'], input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences = True))
    model.add(Dropout(space['dropout']))
    if (space['num_layer'] >= 2):
        model.add(GRU(space['neurons_GRU_layer_2'], return_sequences = True))
        model.add(Dropout(space['dropout']))
    if (space['num_layer'] >= 3):
        model.add(GRU(space['neurons_GRU_layer_3'], return_sequences = True))
        model.add(Dropout(space['dropout']))
    model.add(Flatten())
    model.add(Dense(space['neurons_Dense_layer']))
    model.add(Dense(3, activation='softmax'))
    model.summary()

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='train_loss', factor=0.2,
                                            patience=3, min_lr=0.001)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train,
              Y_train,
              epochs=60,
              batch_size=space['batch_size'],
              validation_data=(X_valid, Y_valid),
              callbacks=[reduce_lr],
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

    return result


if __name__ == "__main__":

    #GPU assignment part:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_number', type=int, default=None)
    args = parser.parse_args()
    if args.gpu_number is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_number)


    print(x.shape)
    trials = Trials()
    best = fmin(fn=f, space=space, algo=tpe.suggest, max_evals=1,
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
