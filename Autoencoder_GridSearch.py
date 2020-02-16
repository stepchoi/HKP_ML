from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd

import os

from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from keras.utils.vis_utils import plot_model
from keras.callbacks import CSVLogger



# Function to create Autoencoder model, required for KerasClassifier
def create_Autoencoder(optimizer='adam', activation='relu', input_shape=[3169]):

    model = Sequential()

    model.add(Dense(units=1024, activation=activation, name='en1', input_shape=input_shape))
    model.add(Dense(units=256, activation=activation, name='en2'))
    model.add(Dense(units=256, name='embedding'))
    model.add(Dense(units=1024, activation=activation, name='de1'))
    model.add(Dense(units=3169, name='de2'))

    plot_model(model, to_file="result/grid_search/model.png", show_shapes=True)

    model.summary()
    print(optimizer, activation)
    # Compile the model
    model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])
    return model


if __name__ == "__main__":



    #Load the data
    training_set1 = pd.read_csv('Set1_training_sample.csv', index_col=0)
    x = training_set1.sample(frac=1, replace=False)
    x = np.array(x)

    #Save result
    if not os.path.exists("result/grid_search"):
        os.makedirs("result/grid_search")

    # create model
    model = KerasRegressor(build_fn=create_Autoencoder, epochs=50, verbose=1, batch_size = 256)


    # define the grid search parameters
    optimizer = ['adam']
    activation = ['tanh']
    param_grid = dict(optimizer=optimizer, activation=activation)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv = 2)
    y = x.copy()
    grid_result = grid.fit(x, y)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))