'''
This script is aimed to illustrate how to load data using local csv file & existing LoadData code.
Here we use PCA code for example.

Step 1: Preparation
        1.1 download LoadData_float32.zip from google drive: https://drive.google.com/open?id=197eRULH4uv-OHLc3wbsjODk1I37TpR9K
            -> contains main.csv, niq.csv, macro_main.csv
'''

import datetime as dt
import gc
import time  # 2.1 import modules used in PCA codes

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.decomposition import PCA
from tqdm import tqdm

'''Step 2: write PCA codes'''

# 2.2 create def related to PCA
def myPCA(X_std): # run PCA with no predetermined No. of components

    pca = PCA()
    pca.fit(X_std)
    ratio = pca.explained_variance_ratio_
    return np.cumsum(ratio) # return cummulative sum of explained_variance_ratio


'''Step 3: parts will be ran when running this py script '''
if __name__ == "__main__":

    start = time.time() # timing function to record total running time

    '''
    Step 4: (this part is updated) 
            4.1 load_data
            4.2 for loop -> roll over all time period
    '''
    # import 'LoadData.py' module from Local Finder 'Preprocessing'
    # import load_data, clean_set function from 'LoadData.py' module
    # this need update on GitHub -> Update Project from VCS (Command + T on MacBook)
    from Preprocessing.LoadData import (load_data)


    # 4.1. run load data -> return entire dataframe (153667, 3174) for all datacqtr (period)
    main = load_data(lag_year=5, sql_version=False)  # change sql_version -> True if trying to run this code through Postgres Database

    explanation_ratio_dict = {}  # create dictionary contains explained_variance_ratio for all 40 sets

    # 4.2. for loop -> roll over all time period from main dataset
    period_1 = dt.datetime(2008, 3, 31)

    for i in tqdm(range(2)): # change to 40 for full 40 sets, change to False to stop saving csv

        testing_period = period_1 + i * relativedelta(months=3)  # define testing period

        train_x, test_x, train_y, test_y = sample_from_datacqtr(main, y_type = 'yoy', testing_period=testing_period)

        explanation_ratio_dict[i] = myPCA(train_x)  # write explained_variance_ratio_ to dictionary

        del train_x, test_x, train_y, test_y  # delete this train_x and collect garbage -> release memory
        gc.collect()

    # convert dictionary to csv and save to local
    pd.DataFrame.from_dict(explanation_ratio_dict).to_csv('explanation_ratio.csv')

    end = time.time() # timing function to record total running time
    print('PCA total running time: {}'.format(end - start))


