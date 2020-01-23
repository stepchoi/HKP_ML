from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model
import numpy as np

def timestamp():


    text = '1997Q4'
    a = time_stamp(text)

    return text


def AE(input_shape=(69), neurons=[64, 32, 16, 8]):

    model = Sequential()

    model.add(Dense(units=neurons[0], activation='relu', name='en1', input_shape=input_shape))

    model.add(Dense(units=neurons[1], activation='relu', name='en2'))

    model.add(Dense(units=neurons[2], activation='relu', name='en3'))

    model.add(Dense(units=neurons[2], name='embedding'))

    model.add(Dense(units=neurons[1], activation='relu', name='de1'))

    model.add(Dense(units=neurons[0], activation='relu', name='de2'))

    model.add(Dense(units=69 , name='de3'))

    model.summary()
    return model

"""
def random_forest(x, y):

    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestRegressor

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=6)
    maxFeature = np.array([None])

    # Define the gridsearch inputs:
    param_grid = {'max_features': maxFeature}

    # Find the best parameters of the grid:
    gridRF = GridSearchCV(estimator=RandomForestRegressor(random_state=2, n_estimators=200),
                          param_grid=param_grid,
                          cv=5, scoring=None)
    gridRF.fit(x_train, y_train)

    # Print the accuracy of the tuned random forest:
    print('Best CV R^2: {:.3f}'.format(gridRF.best_score_))
    print('Out of Sample R^2: {:.3f}'.format(gridRF.score(x_test, y_test)))
    print('Best Parameters: ', gridRF.best_params_)

    return(gridRF.score(x_test, y_test))
"""


if __name__ == "__main__":
    from time import time

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default='usps', choices=['mnist', 'usps'])
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--save_dir', default='results/temp', type=str)
    args = parser.parse_args()
    print(args)

    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load dataset
    from MainDataLoad import load_main

    x, y = load_main(10000)

    # define the model
    neurons = [64, 32, 32, 8]
    model = AE(input_shape=x.shape[1:], neurons=neurons)
    plot_model(model, to_file=args.save_dir + '/%s-pretrain-model.png' % args.dataset, show_shapes=True)
    model.summary()

    # compile the model and callbacks
    optimizer = 'adam'
    model.compile(optimizer=optimizer, loss='mae')
    from keras.callbacks import CSVLogger
    csv_logger = CSVLogger(args.save_dir + '/%s-pretrain-log.csv' % args.dataset)

    # begin training
    t0 = time()
    model.fit(x, x, batch_size=args.batch_size, epochs=args.epochs, callbacks=[csv_logger])
    print('Training time: ', time() - t0)
    model.save(args.save_dir + '/%s-pretrain-model-%d.h5' % (args.dataset, args.epochs))

    # extract embedded features
    feature_model = Model(inputs=model.input, outputs=model.get_layer(name='embedding').output)
    features = feature_model.predict(x)
    features = np.reshape(features, newshape=(features.shape[0], -1))
    print('feature shape=', features.shape)
    print('y shape=', y.shape)


    #use features for Random Forest Regression
    embedded_RFRscore = random_forest(features, y)
    #RFRscore = random_forest(x, y)
    print(neurons[2], embedded_RFRscore)
    #print(RFRscore)
