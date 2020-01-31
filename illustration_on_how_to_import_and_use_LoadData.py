'''
This script is aimed to illustrate how to load data using local csv file & existing LoadData code.
Here we use PCA code for example.

Step 1: Preparation
        1.1 download LoadData_csv.zip from google drive
            -> contains main.csv, niq.csv
        1.2 link VPN
            -> connect PostgreSQL for macro_main table
'''



'''Step 2: write PCA codes'''
import time  # 2.1 import modules used in PCA codes

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# 2.2 create class contains many functions related to PCA
#     can also be directly def ....(X_std) -> import data and run PCA

class myPCA:

    def __init__(self, X_std):  # __init__ run PCA with no predetermined No. of components
        self.pca = PCA()
        self.pca.fit(X_std)
        self.X_std = X_std
        self.ratio = self.pca.explained_variance_ratio_

    def primary_PCA(self):      # primary_PCA return cummulative sum of explained_variance_ratio
        return np.cumsum(self.ratio)

    '''other def....'''


'''Step 3: parts will be ran when running this py script '''
if __name__ == "__main__":

    start = time.time() # timing function to record total running time

    '''
    Step 4: import LoadData module written and run
    
    This part will return a dictionary contains 40 sets in below structure: 
                            sets
                         /        \
                    set[1]  ...  set[40]                   
                /     |      \ 
        [train_x] [train_yoy] [train_qoq]   (qoq & yoy two types of y - already qcut into 3 parts)
        [test_x]  [test_qoq]  [test_yoy]
            |           \         /
      StandardScaler        qcut
         (Arrays)         (Arrays)            
         
    '''
    # import 'LoadData.py' module from Local Finder 'Preprocessing'
    # import load_data function from 'LoadData.py' module
    # this need update on GitHub -> Update Project from VCS (Command + T on MacBook)
    from Preprocessing.LoadData import load_data


    # run load data -> return dictionary mentioned above
    # load_data(sets_no, save_csv)
    # sets_no: decide no of sets will be returned in dictionary [for entire sets -> set as 40]
    # save_csv: will the train_x array will be saved as csv file -> if True will save (longer processing time)
    sets = load_data(2, save_csv=True)  # change to 40 for full 40 sets

    ''' e.g. 
    sets[1]['train_x']    -> This is first training set x, i.e. 1988Q1 - 2007Q4
    sets[1]['train_qoq']  -> This is first training set y - niq qoq, i.e. 1988Q1 - 2007Q4
    print(sets.keys())    => this should print # of set
    '''

    '''Step 5: use loaded data for PCA '''

    explanation_ratio_dict = {}  # create dictionary contains explained_variance_ratio for all 40 sets

    # loop entire sets for explained_variance_ratio in each sets
    for set in sets.keys():
        explanation_ratio_dict[set] = myPCA(sets[set]['train_x']).primary_PCA()

    # convert dictionary to csv and save to local
    pd.DataFrame.from_dict(explanation_ratio_dict).to_csv('explanation_ratio.csv')

    end = time.time() # timing function to record total running time
    print('PCA total running time: {}'.format(end - start))


