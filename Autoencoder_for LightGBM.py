import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from keras.layers import Dense
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model


def AE_fitting(training_x):

    model = Sequential()
    model.add(Dense(units=1800, activation='tanh', name='en1', input_shape=[3169]))
    model.add(Dense(units=650, activation='tanh', name='en2'))
    model.add(Dense(units=1800, activation='tanh', name='de1'))
    model.add(Dense(units=3169, name='de2'))

    model.summary()

    # extract compressed feature
    model.compile(optimizer='adam', loss='mae')

    model.fit(training_x, training_x, batch_size=2000, epochs=50)
    feature_model = Model(inputs=model.input, outputs=model.get_layer(name='en2').output)

    return feature_model

def AE_predict(x, feature_model):

    compressed_x = feature_model.predict(x)
    print('feature shape=', compressed_x.shape)

    return compressed_x



if __name__ == "__main__":

    training_x = pd.read_csv('trainingset0.csv', index_col=0)
    #testing_x = pd.read_csv('trainingset1.csv', index_col=0)

    feature_model = AE_fitting(training_x)
    training_compressed_x = AE_predict(training_x, feature_model)
    #testing_compressed_x = AE_predict(testing_x, feature_model)
    print(training_compressed_x.shape)
