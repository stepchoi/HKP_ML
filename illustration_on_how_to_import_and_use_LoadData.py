'''
This script is aimed to illustrate how to load data using local csv file & existing LoadData code.
Here we use PCA code for example.

Step 1: Preparation
        1.1 download LoadData_csv.zip from google drive: https://drive.google.com/open?id=1M5dK84S6Uo71P8XxaJ5lLXYahucoBqRV
            -> contains main.csv, niq.csv
        1.2 link VPN
            -> connect PostgreSQL for macro_main table
'''



'''Step 2: write PCA codes'''
import datetime as dt
import gc
import time  # 2.1 import modules used in PCA codes

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.decomposition import PCA
from tqdm import tqdm


# 2.2 create class contains many functions related to PCA
#     can also be directly def ....(X_std) -> import data and run PCA

class myPCA:

    def __init__(self, X_std):  # __init__ run PCA with no predetermined No. of components
        self.pca = PCA()
        self.pca.fit(X_std)
        self.X_std = X_std
        self.ratio = self.pca.explained_variance_ratio_

    def primary_PCA(self):      # primary_PCA return cummulative sum of explained_variance_ratio
        print('--> start pca')
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
    from Preprocessing.LoadData import (load_data, clean_set)


    # run load data -> return dictionary mentioned above
    # load_data(sets_no, save_csv)
    # sets_no: decide no of sets will be returned in dictionary [for entire sets -> set as 40]
    # save_csv: will the train_x array will be saved as csv file -> if True will save (longer processing time)
    main = load_data()
    # main.to_csv('main_lag.csv', index = False)

    '''Step 5: use loaded data for PCA '''
    explanation_ratio_dict = {}  # create dictionary contains explained_variance_ratio for all 40 sets

    # loop entire sets for explained_variance_ratio in each sets
    period_1 = dt.datetime(2008, 3, 31)
    for i in tqdm(range(5)): # change to 40 for full 40 sets, change to False to stop saving csv
        '''training set: x -> standardize -> apply to testing set: x
            training set: y -> qcut -> apply to testing set: y'''
        testing_period = period_1 + i * relativedelta(months=3)
        train_x = clean_set(main, testing_period).standardize_x()

        explanation_ratio_dict[set] = myPCA(train_x).primary_PCA()

        del train_x
        gc.collect()

    # convert dictionary to csv and save to local
    pd.DataFrame.from_dict(explanation_ratio_dict).to_csv('explanation_ratio.csv')

    end = time.time() # timing function to record total running time
    print('PCA total running time: {}'.format(end - start))


