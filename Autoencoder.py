import argparse

from keras.layers import Dense
from keras.models import Sequential


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



if __name__ == "__main__":


    # setting the hyper parameters

    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default='usps', choices=['mnist', 'usps'])
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--save_dir', default='results/temp', type=str)
    args = parser.parse_args()
    print(args)

    main = load_data(sql_version=False).sample(10000)
    print(main.info())

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
    model.fit(x, x, batch_size=args.batch_size, epochs=args.epochs, callbacks=[csv_logger])
    print('Training time: ', time() - t0)
    model.save(args.save_dir + '/%s-pretrain-model-%d.h5' % (args.dataset, args.epochs))

    # extract embedded features
    feature_model = Model(inputs=model.input, outputs=model.get_layer(name='embedding').output)
    features = feature_model.predict(x)
    features = np.reshape(features, newshape=(features.shape[0], -1))
    print('feature shape=', features.shape)
    print('y shape=', y.shape)
