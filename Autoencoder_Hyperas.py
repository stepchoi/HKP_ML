from __future__ import print_function

from hyperopt import Trials, STATUS_OK, tpe, rand
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from hyperas import optim
from hyperas.distributions import choice, uniform, randint

from keras import models
from keras import layers
import numpy as np
from keras import optimizers

import pandas as pd

def data():

    data = pd.read_csv('trainingset0.csv', index_col=0)

    print(data.head())

    X_train, X_test, y_train, y_test = train_test_split(data, data, test_size=0.25, random_state=777)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=777)


    return X_train, X_val, X_test, y_train, y_val, y_test


def create_model(X_train, y_train, X_val, y_val):


    number_of_layers = {{choice[2, 3]}}
    layer_2_size = {{quniform(650, 3169, 200)}}
    layer_3_size = {{quniform(650, 3169, 200)}}
    params = {
        'num_layers': number_of_layers
        'l2_size': layer_2_size,
        'l3_size': layer_3_size,
    }

    model = models.Sequential()

    model.add(Dense(int(layer_2_size), activation='tanh', name='en1', input_shape=[3169]))
    if (number_of_layers == 3):
        model.add(Dense(int(layer_3_size), activation='tanh', name='en2'))
    model.add(Dense(units=650, activation= 'tanh', name='en3'))
    if (number_of_layers == 3):
        model.add(Dense(int(layer_3_size), activation='tanh', name='de2'))
    model.add(Dense(int(layer_2_size), activation= 'tanh', name='de2'))
    model.add(Dense(3169, name='de3'))

    model.summary()

    model.compile(optimizer='adam',
                  loss='mae',
                  metrics=['mae'])

    model.fit(X_train,
              y_train,
              epochs=30,
              batch_size=512,
              validation_data=(X_val, y_val))

    score, mae = model.evaluate(X_val, y_val, verbose=0)
    print('Test accuracy:', mae)

    out = {
        'loss': mae,
        'score': score,
        'status': STATUS_OK,
        'model_params': params,
    }

    return out


if __name__ == '__main__':



    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=40,
                                          trials=Trials())

    X_train, X_val, X_test, y_train, y_val, y_test = data()
    print("Evalutation of best performing model:")
    #print(best_model.evaluate(X_test, y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    print(best_model)