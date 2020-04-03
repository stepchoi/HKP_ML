import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split


def PCA_fitting(x_train, reduced_dimensionss):

    x_train_new = x_train.reshape(x_train.shape[0]*x_train.shape[1], x_train.shape[2])
    PCA_model = PCA(n_components=reduced_dimensionss)
    PCA_model.fit(x_train_new)

    return PCA_model

def PCA_predict(x, PCA_model):

    x_new = x.reshape(x.shape[0]*x.shape[1], x.shape[2])

    compressed_x_new = PCA_model.transform(x_new)
    compressed_x = compressed_x_new.reshape(x.shape[0], x.shape[1], -1)
    print('feature shape=', compressed_x.shape)

    return compressed_x



if __name__ == "__main__":
    x = np.random.randn(20000000)
    x = x.reshape(5000, 20, 200)
    y = np.random.randint(3, size=5000)

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.25, stratify=y)

    PCA_model = PCA_fitting(x_train, 0.13)
    x_valid = PCA_predict(x_valid, PCA_model)
    print(x_valid.shape)
