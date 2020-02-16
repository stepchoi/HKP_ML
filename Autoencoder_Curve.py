from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd

import statistics

import os

from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from keras.utils.vis_utils import plot_model
from keras.callbacks import CSVLogger

from sklearn.metrics import r2_score, explained_variance_score



def Autoencoder(input_shape=3169, neurons=50):

    model = Sequential()

    model.add(Dense(units=1024, activation='tanh', name='en1', input_shape=input_shape))
    model.add(Dense(units=512, name='en2'))
    model.add(Dense(units=neurons, name='en3'))
    model.add(Dense(units=512, name='de1'))
    model.add(Dense(units=1024, name='de2'))
    model.add(Dense(units=3169, name='de3'))

    return model


if __name__ == "__main__":

    # Save result
    if not os.path.exists("result/curve"):
        os.makedirs("result/curve")

    # Load the data
    x = pd.read_csv('Set1_training_sample10000.csv', index_col= 0)
    x = np.array(x)

    scoring = pd.DataFrame(columns=['Dimension', 'MAE Loss', 'Explained_variance', 'R^2 Score'])

    for i in range(1, 11):

        # Build the model
        dimension = i * 50
        model = Autoencoder(input_shape=x.shape[1:], neurons=dimension)
        plot_model(model, to_file="result/curve/train_three_layers" + str(dimension) + "model.png", show_shapes=True)
        model.summary()

        # Compile the model
        optimizer = 'adam'
        model.compile(optimizer=optimizer, loss='mae')

        # Save the result
        csv_logger = CSVLogger("result/curve/train_three_layers" + str(dimension) + "model.csv")
        model.fit(x, x, batch_size=512, epochs=50, callbacks=[csv_logger])

        # Calculate the R^2 and Explained Variance
        x_pred = model.predict(x, batch_size=1024, verbose=0)

        mae = model.evaluate(x, x, batch_size=1024, verbose=0)
        scoring.loc[len(scoring), 'Dimension'] = dimension
        scoring.loc[len(scoring)-1, 'MAE Loss'] = mae
        scoring.loc[len(scoring) - 1, 'Explained_variance'] = explained_variance_score(x, x_pred,
                                                                              multioutput='variance_weighted')
        scoring.loc[len(scoring)-1, 'R^2 Score'] = r2_score(x, x_pred, multioutput='variance_weighted')

        scoring.to_csv('scoring_three_layers.csv')

    scoring.to_csv('scoring_three_layers.csv')
