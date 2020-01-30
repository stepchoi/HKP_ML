'''This program is adjusted based on dimension reduction-PCA.py for sql & rolling version'''

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA

from Preprocessing.Lag_TrainTestCut import full_running_cut


class myPCA:

    def __init__(self, X_std):
        self.pca = PCA()
        self.pca.fit(X_std)
        self.X_std = X_std
        self.ratio = pca.explained_variance_ratio_
        print("pca.components_",self.pca.components_.shape)
        print("pca_var_ratio",self.pca.explained_variance_ratio_.shape)

    def primary_PCA(self):
        return np.cumsum(self.ratio)

    def plot_PCA(self):
        plt.plot([i for i in range(X.shape[1])],
                 [np.sum(ratio[:i+1]) for i in range(X.shape[1])])
        plt.xticks(np.arange(X.shape[1],step=5))
        plt.yticks(np.arange(0,1.01,0.05))
        plt.grid()
        plt.savefig('Cumulative Explained Variance.png')
        plt.show()

    def secondary_PCA(self):
        pca = PCA(n_components=0.60) # Threshold for dimension reduction,float or integer
        pca.fit(X_std)
        res = pca.transform(X_std)
        res_df = pd.DataFrame(res)
        res_df.to_csv('result75.csv', index=False)
        return res

    # Recover the weights for all original dimensions
    def recover_original_dimension(self):
        X_std_df = pd.DataFrame(X_std)
        origin = pd.DataFrame(pca.components_,columns=X_std_df.columns)
        invres=pca.inverse_transform(res)#
        df = pd.DataFrame(origin)
        df.to_csv('origin.csv', index=False)


if __name__ == "__main__":

    start = time.time()

    # return a dictionary contains 40 sets
    # -> each set contain 6 arrays train_x, train_qoq, train_yoy,
    #                              test_x, test_qoq, test_yoy (qoq & yoy two types of y - already qcut into 3 parts)
    sets = full_running_cut()
    print(sets.keys())

    # roll over all sets's train_x array
    exp_dict = {}
    for set in sets.keys():
        exp_dict[set] = myPCA(sets[set]['train_x']).primary_PCA()

    print(exp_dict)

    end = time.time()
    print('PCA total running time: {}'.format(end - start))

