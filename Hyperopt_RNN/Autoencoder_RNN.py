import numpy as np
import pandas as pd

from keras.layers import Dense
from keras.models import Sequential, Model

from sklearn.model_selection import train_test_split



def AE_fitting(x_train, reduced_dimensionss):

    x_train_new = x_train.reshape(x_train.shape[0] * x_train.shape[1], x_train.shape[2])

    model = Sequential()
    model.add(Dense(units=reduced_dimensionss + 30, activation='tanh', name='en1', input_shape=[x_train.shape[2]]))
    model.add(Dense(units=reduced_dimensionss, activation='tanh', name='en2'))
    model.add(Dense(units=reduced_dimensionss + 30, activation='tanh', name='de1'))
    model.add(Dense(units=x_train.shape[2], name='de2'))

    model.summary()

    # extract compressed feature
    model.compile(optimizer='adam', loss='mae')

    model.fit(x_train_new, x_train_new, batch_size=1024, epochs=25)
    AE_model = Model(inputs=model.input, outputs=model.get_layer(name='en2').output)

    return AE_model

def AE_predict(x, AE_model):

    x_new = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

    compressed_x_new = AE_model.predict(x_new)
    compressed_x = compressed_x_new.reshape(x.shape[0], x.shape[1], -1)
    print('feature shape=', compressed_x.shape)

    return compressed_x



if __name__ == "__main__":
    x = np.random.randn(20000000)
    x = x.reshape(5000, 20, 200)
    y = np.random.randint(3, size=5000)

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.25, stratify=y)

    PCA_model = AE_fitting(x_train, 75)
    x_valid = AE_predict(x_valid, PCA_model)
    print(x_valid.shape)
