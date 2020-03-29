import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from keras.layers import Dense
from keras.models import Sequential
from keras.utils.vis_utils import plot_model


def AE(data):

    model = Sequential()
    model.add(Dense(units=1800, activation='tanh', name='en1', input_shape=[3169]))
    model.add(Dense(units=650, activation='tanh', name='en2'))
    model.add(Dense(units=1800, activation='tanh', name='de1'))
    model.add(Dense(units=3169, name='de2'))

    model.summary()

    model.fit(data, data, batch_size=512, epochs=50)

    # extract compressed feature
    feature_model = model(inputs=model.input, outputs=model.get_layer(name='en2').output)
    features = feature_model.predict(data)
    print('feature shape=', features.shape)

    return features



if __name__ == "__main__":


    features = AE(data)
