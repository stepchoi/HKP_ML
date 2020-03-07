import numpy as np
import pandas as pd

from sklearn.decomposition import PCA


def PCA_fitting(training_x, reduced_dimensionss):

    PCA_model = PCA(n_components=reduced_dimensionss)
    PCA_model.fit(training_x)

    return PCA_model

def PCA_predict(x, PCA_model):

    compressed_x = PCA_model.transform(x)
    print('feature shape=', compressed_x.shape)

    return compressed_x



if __name__ == "__main__":

    training_x = pd.read_csv('trainingset0.csv', index_col=0)

    PCA_model = PCA_fitting(training_x, 508)
    training_compressed_x = PCA_predict(training_x, PCA_model)
    print(training_compressed_x.shape)
