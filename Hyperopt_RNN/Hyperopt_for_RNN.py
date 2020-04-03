import pandas as pd
import numpy as np

#from PCA_for_LightGBM import PCA_fitting, PCA_predict
#from Autoencoder_for_LightGBM import AE_fitting, AE_predict

from sklearn.model_selection import train_test_split

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from keras.layers import Dense, GRU, Dropout
from keras import models, callbacks

from sklearn.decomposition import PCA



from Preprocessing.LoadData import load_data, sample_from_main

space = {
    # dimension
    'reduced_dimension' : hp.choice('reduced_dimension', np.arange(0.66, 0.75, 0.01)), # past: [508, 624, 757]

    # number of layers
    'num_layer': hp.choice('num_layer', [1, 2, 3]),

    'verbosity': -1,

    # hyperparameter
    'neurons_GRU)layer_1': hp.choice('neurons_GRU_layer_1', [4, 8, 16]),
    'neurons_GRU_layer_2': hp.choice('neurons_GRU_layer_2', [4, 8, 16]),
    'neurons_GRU_layer_3': hp.choice('neurons_GRU_layer_3', [4, 8, 16]),
    #'neurons_Dense_layer': hp.choice('neurons_Dense_layer_2', [32, 128]),
    'batch_size': hp.choice('batch_size', [512, 1024, 2048]),
    'dropout': hp.choice('dropout', [0, 0.2, 0.4])
}


def Data_Loading():

    x = np.random.randn(10000000)
    x = x.reshape(5000, 20, 100)
    y = np.random.randint(3, size=5000)


    x_RNN, x_test, y_RNN, y_test = train_test_split(x, y, test_size=0.2)
    x_train, x_valid, y_train, y_valid = train_test_split(x_RNN, y_RNN, test_size=0.25)

    return x_train, x_valid, x_test, y_train, y_valid, y_test




def RNN(space):

    x_train, x_valid, x_test, y_train, y_valid, y_test = Data_Loading()

    model = models.Sequential()
    model.add(GRU(space['neurons_layer_1'], input_shape=(X_train.shape[1], X_train.shape[2]), return_sequence = True))
    model.add(Dropout(space['dropout']))
    if (space['num_layer'] >= 2):
        model.add(GRU(space['neurons_layer_2']))
        model.add(Dropout(space['dropout']))
    if (space['num_layer'] >= 3):
        model.add(GRU(space['neurons_layer_3']))
        model.add(Dropout(space['dropout']))
    model.add(Dense(3, activation='softmax'))
    model.summary()

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='train_loss', factor=0.2,
                                            patience=3, min_lr=0.001)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train,
              y_train,
              epochs=60,
              batch_size=space['batch_size'],
              validation_data=(x_valid, y_valid),
              callbacks=[reduce_lr],
              verbose=1)

    loss_train, accuracy_train = model.evaluate(x_train, y_train, space['batch_size'], verbose=1)
    loss_valid, accuracy_valid = model.evaluate(x_valid, y_valid, space['batch_size'], verbose=1)
    loss_valid, accuracy_test = model.evaluate(x_test, y_test, space['batch_size'], verbose=1)

    return accuracy_train, accuracy_valid, accuracy_test


def f(space):

    accuracy_train, accuracy_valid, accuracy_test = RNN(space)

    result = {'loss': 1 - accuracy_valid,
              'accuracy_train': accuracy_train,
              'accuracy_valid': accuracy_valid,
              'accuracy_test': accuracy_test,
              'space': space,
              'status': STATUS_OK}

    print(space)
    print(result)

    return result


if __name__ == "__main__":

    x = np.random.randn(10000000)
    x = x.reshape(5000, 20, 100)
    y = np.random.randint(3, size=5000)
    #pca = PCA(n_components=0.66)
    #x_new = pca.fit_transform(x)

    x_new_np.rot90(x)
    print(x_new.shape)



"""

    print(data.head(5))
    print(data.shape)

    trials = Trials()
    best = fmin(fn=f, space=space, algo=tpe.suggest, max_evals=max_evals,
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
"""