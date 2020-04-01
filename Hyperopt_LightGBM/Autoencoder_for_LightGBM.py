import numpy as np
import pandas as pd

from keras.layers import Dense
from keras.models import Sequential, Model

def AE_fitting(training_x, reduced_dimensionss):

    model = Sequential()
    if reduced_dimensionss > 700:
        second_layer = 2400
    elif reduced_dimensionss < 600:
        second_layer = 1800
    else:
        second_layer = 2000

    model.add(Dense(units=second_layer, activation='tanh', name='en1', input_shape=[3091]))
    model.add(Dense(units=reduced_dimensionss, activation='tanh', name='en2'))
    model.add(Dense(units=second_layer, activation='tanh', name='de1'))
    model.add(Dense(units=3091, name='de2'))

    model.summary()

    # extract compressed feature
    model.compile(optimizer='adam', loss='mae')

    model.fit(training_x, training_x, batch_size=2000, epochs=50)
    AE_model = Model(inputs=model.input, outputs=model.get_layer(name='en2').output)

    return AE_model

def AE_predict(x, AE_model):

    compressed_x = AE_model.predict(x)
    print('feature shape=', compressed_x.shape)

    return compressed_x



if __name__ == "__main__":

    training_x = pd.read_csv('trainingset0.csv', index_col=0)

    AE_model = AE_fitting(training_x, 508)
    training_compressed_x = AE_predict(training_x, AE_model)
    print(training_compressed_x.shape)
