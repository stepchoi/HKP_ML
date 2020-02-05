import datetime as dt
import gc
import time

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from tqdm import tqdm


def convert_to_float32(df):
    df.loc[:, df.dtypes == np.float64] = df.loc[:, df.dtypes == np.float64].astype(np.float32)

def add_lag(df):
    print('---------------------------- (step 1/3) adding lag -----------------------------')
    start = time.time()

    # df.loc[df.isnull().sum(axis=1) == 0, 'dropna'] = 1
    convert_to_float32(df)
    print(df.info())

    col = df.columns[3:]
    lag_df = []
    lag_df.append(df.dropna())

    # missing_dict = []
    # missing_dict.append(df.mask(df == 0).isnull().sum(axis=1))

    for i in tqdm(range(19)): # change to 19
        df_temp = df.groupby('gvkey').shift(i + 1)[col]
        df_temp.columns = ['{}_lag{}'.format(k, str(i+1).zfill(2)) for k in col]
        df_temp = df_temp.dropna()
        lag_df.append(df_temp)
        # missing_dict.append(df_temp.mask(df_temp == 0).isnull().sum(axis=1))

    # delete_high_zero_row(missing_dict)
    df_lag = pd.concat(lag_df, axis = 1, join='inner')

    end = time.time()
    print('(step 1/3) adding lag running time: {}'.format(end - start))
    print(df_lag.shape)
    return df_lag


def merge_dep_macro(df, sql_version):
    print('----------------- (step 2/3) adding macro & dependent variable -----------------')
    start = time.time()

    if sql_version is True:
        db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
        engine = create_engine(db_string)
        dep = pd.read_sql('SELECT * FROM niq', engine)
        macro = pd.read_sql("SELECT * FROM macro_main", engine)
    else:
        macro = pd.read_csv('macro_main.csv')
        dep = pd.read_csv('niq.csv')
        print('local version running - niq & macro_main')

    convert_to_float32(dep)
    convert_to_float32(macro)
    print(dep.info())

    dep['datacqtr'] = pd.to_datetime(dep['datacqtr'],format='%Y-%m-%d')
    macro['datacqtr'] = pd.to_datetime(macro['datacqtr'],format='%Y-%m-%d')

    dep_macro = pd.merge(macro, dep, on=['datacqtr'], how='right')
    dep_macro = dep_macro.dropna()
    print(dep_macro.isnull().sum().sum())
    df_macro_dep = pd.merge(df, dep_macro, on=['gvkey', 'datacqtr'], how='inner')  # change to df_macro

    end = time.time()
    print('(step 2/3) adding macro & dependent variable running time: {}'.format(end - start))
    print(df_macro_dep.info())

    return df_macro_dep

class clean_set:

    def __init__(self, df, testing_period):
        s = time.time()
        end = testing_period
        start = testing_period - relativedelta(years=20)
        def divide_set(df):
            return df.iloc[:, 3:-2].values, df.iloc[:, -2].values, df.iloc[:, -1].values
        self.test_x, self.test_qoq, self.test_yoy = divide_set(df.loc[df['datacqtr'] == end])
        self.train_x, self.train_qoq, self.train_yoy = divide_set(df.loc[(start <= df['datacqtr']) & (df['datacqtr'] < end)])
        e = time.time()
        print(self.train_x.shape)
        print('--> 3.1. divide test training set using {}'.format(e - s))

    def standardize_x(self, return_test_x = False):
        s = time.time()
        scaler = StandardScaler().fit(self.train_x)
        self.train_x = scaler.transform(self.train_x)
        self.test_x = scaler.transform(self.test_x)
        e = time.time()
        print('--> 3.2. standardize x using {}'.format(e - s))
        if return_test_x is True:
            return self.train_x, self.test_x
        else:
            return self.train_x

    def yoy(self):
        s = time.time()
        self.train_yoy, cut_bins = pd.qcut(self.train_yoy, q=3, labels=[0, 1, 2], retbins=True)
        self.test_yoy = pd.cut(self.test_yoy, bins=cut_bins, labels=[0, 1, 2])
        e = time.time()
        print('--> 3.3. qcut y using {}'.format(e - s))
        return self.train_yoy, self.test_yoy

    def qoq(self):
        s = time.time()
        self.train_qoq, cut_bins = pd.qcut(self.train_qoq, q=3, labels=[0, 1, 2], retbins=True)
        self.test_qoq = pd.cut(self.test_qoq, bins=cut_bins, labels=[0, 1, 2])
        e = time.time()
        print('--> 3.3. qcut y using {}'.format(e - s))
        return self.train_qoq, self.train_qoq

def cut_test_train(df, sets_no, save_csv = False):
    print('------------------- (step 3/3) cutting testing/training set --------------------')
    start_total = time.time()

    dict = {}
    testing_period = dt.datetime(2008, 3, 31)

    for i in tqdm(range(sets_no)):
        '''training set: x -> standardize -> apply to testing set: x
            training set: y -> qcut -> apply to testing set: y'''
        end = testing_period + i*relativedelta(months=3)
        start = testing_period - relativedelta(years=20)
        dict[i + 1] = {}
        print(i+1, end)


        s = time.time()
        def divide_set(df):
            return df.iloc[:, 3:-2].values, df.iloc[:, -2].values, df.iloc[:, -1].values
        dict[i+1]['test_x'], dict[i+1]['test_qoq'], dict[i+1]['test_yoy'] = divide_set(df.loc[df['datacqtr'] == end])
        dict[i+1]['train_x'], dict[i+1]['train_qoq'], dict[i+1]['train_yoy'] = divide_set(df.loc[(start <= df['datacqtr'])
                                                                                             & (df['datacqtr'] < end)])
        e = time.time()
        print('--> 3.1. divide test training set using {}'.format(e - s))

        s = time.time()
        scaler = StandardScaler().fit(dict[i+1]['train_x'])
        dict[i+1]['train_x'] = scaler.transform(dict[i+1]['train_x'])
        dict[i+1]['test_x'] = scaler.transform(dict[i+1]['test_x'])
        e = time.time()
        print('--> 3.2. standardize x using {}'.format(e - s))

        s = time.time()
        for y in ['qoq', 'yoy']:
            dict[i+1]['train_' + y], cut_bins = pd.qcut(dict[i+1]['train_' + y], q=3, labels=[0, 1, 2], retbins=True)
            dict[i+1]['test_' + y] = pd.cut(dict[i+1]['test_' + y], bins=cut_bins, labels=[0, 1, 2])
        e = time.time()
        print('--> 3.3. qcut y using {}'.format(e - s))


        if save_csv is True:
            s = time.time()
            pd.DataFrame(dict[i+1]['train_x']).to_csv('train_x_set{}.csv'.format(i), index = False, header = False)
            e = time.time()
            print('--> 3.4. saving to csv using {}'.format(e - s))

    # save_load_dict('save', dict=cut_bins, name='cut_bins') # save cut_bins to dictionary

    end_total = time.time()
    print('(step 3/3) cutting testing/training set running time: {}'.format(end_total - start_total))

    return dict

def load_data(sql_version = False):

    # import engine, select variables, import raw database
    print('-------------- start load data into different sets (-> dictionary) --------------')
    start = time.time()

    if sql_version is True:
        db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
        engine = create_engine(db_string)
        main = pd.read_sql('SELECT * FROM main', engine)
    else:
        main = pd.read_csv('main.csv')
        engine = None
        print('local version running - main')

    end = time.time()
    print('(step 0/3) read local csv - main - running time: {}'.format(end - start))

    print(main.shape)
    main['datacqtr'] = pd.to_datetime(main['datacqtr'],format='%Y-%m-%d')

    # 1. add 20 lagging factors for each variable
    main_lag = add_lag(main)
    del main
    gc.collect()

    # 2. add dependent variable & macro variables to main
    main_lag = merge_dep_macro(main_lag, sql_version)

    def save_lag_to_csv():
        start = time.time()
        main_lag.to_csv('main_lag.csv', index=False)
        end = time.time()
        print('save csv running time: {}'.format(end - start))

    print(main_lag.info())
    return main_lag

def cut_test_train_main(sets_no):
    # 3. cut training, testing set
    test_train_dict = cut_test_train(main_lag, sets_no, save_csv)
    return test_train_dict

if __name__ == "__main__":

    # actual running scripts see def above -> for import
    import os
    os.chdir('/Users/Clair/PycharmProjects/HKP_ML_DL')
    # load_data(20, save_csv = False, sql_version = False)
