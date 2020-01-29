'''This program is adjusted based on dimension reduction-PCA.py for sql & rolling version'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# import plotly.offline as py
# import plotly.graph_objs as go
# import plotly.subplots as tls
# import seaborn as sns
# import matplotlib.image as mpimg
# import matplotlib
from sklearn.decomposition import PCA

np.set_printoptions(threshold=np.inf) #Show full string of the result
pd.set_option('display.max_columns', None) #Show all columns
pd.set_option('display.max_rows', None) #Show all rows
pd.set_option('max_colwidth',100) #Set the illustration length for the string, default=50
py.init_notebook_mode(connected=True)

def load_data():

    # Load the dataset and check the dimensions
    train = pd.read_csv('30years-Starting SP.csv',low_memory=False) #Importing
    train_df = pd.DataFrame(train) #Convert the original dataset to dataframe form
    print(train_df.shape)

    return train

# Calculating Eigenvectors and eigenvalues of Cov matirx
###mean_vec = np.mean(X_std, axis=0)
###cov_mat = np.cov(X_std.T)
###eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# Create a list of (eigenvalue, eigenvector) tuples
###eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the eigenvalue, eigenvector pair from high to low
###eig_pairs.sort(key = lambda x: x[0], reverse= True)

# Calculation of Explained Variance from the eigenvalues
###tot = sum(eig_vals)
###var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
###cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance#

#Plot the illustration figure for the cumulative explained variance#

def PCA(X_std):

    pca = PCA()
    pca.fit(X_std)
    ratio=pca.explained_variance_ratio_
    print("pca.components_",pca.components_.shape)
    print("pca_var_ratio",pca.explained_variance_ratio_.shape)
    plt.plot([i for i in range(X.shape[1])],
             [np.sum(ratio[:i+1]) for i in range(X.shape[1])])
    plt.xticks(np.arange(X.shape[1],step=5))
    plt.yticks(np.arange(0,1.01,0.05))
    plt.grid()
    plt.savefig('Cumulative Explained Variance.png')
    plt.show()

    # PCA
    pca = PCA(n_components=0.80) #Threshold for dimension reduction,float or integer
    pca.fit(X_std)
    res=pca.transform(X_std)

    #Recover the weights for all original dimensions
    X_std_df = pd.DataFrame(X_std)
    origin=pd.DataFrame(pca.components_,columns=X_std_df.columns)
    #invres=pca.inverse_transform(res)#

    #df = pd.DataFrame(origin)
    #df.to_csv('origin.csv', index=False)
    res_df=pd.DataFrame(res)
    res_df.to_csv('result75.csv',index=False)

    return res


if __name__ == "__main__":

    # X = load_data()
    import timeit
    from Preprocessing.Lag_TrainTestCut import full_running_cut

    start = timeit.timeit()

    sets = full_running_cut()
    # X_std = standardization(sets[1]['train_x'])
    PCA(sets[1]['train_x'])

    end = timeit.timeit()
    print('running time: {}'.format(end - start))

