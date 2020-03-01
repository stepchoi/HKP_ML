import gc
import time

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sqlalchemy import create_engine
from tqdm import tqdm


def convert_to_float32(df):
    df.loc[:, df.dtypes == np.float64] = df.loc[:, df.dtypes == np.float64].astype(np.float32)

def add_lag(df, lag_year):
    print('---------------------------- (step 1/3) adding lag -----------------------------')
    start = time.time()

    # df.loc[df.isnull().sum(axis=1) == 0, 'dropna'] = 1
    convert_to_float32(df)

    col = df.columns[3:]
    lag_df = []
    lag_df.append(df.dropna())

    # missing_dict = []
    # missing_dict.append(df.mask(df == 0).isnull().sum(axis=1))

    for i in tqdm(range(lag_year*4-1)): # change to 19
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

    dep['datacqtr'] = pd.to_datetime(dep['datacqtr'],format='%Y-%m-%d')
    macro['datacqtr'] = pd.to_datetime(macro['datacqtr'],format='%Y-%m-%d')

    dep_macro = pd.merge(macro, dep, on=['datacqtr'], how='right')
    dep_macro = dep_macro.dropna()
    df_macro_dep = pd.merge(df, dep_macro, on=['gvkey', 'datacqtr'], how='inner')  # change to df_macro

    end = time.time()
    print('(step 2/3) adding macro & dependent variable running time: {}'.format(end - start))
    print(df_macro_dep.shape)

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


def load_data(lag_year = 5, sql_version = False, sample_no = False):

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

    if not sample_no == False:
        main = main.sample(sample_no)
        print(main.info())

    end = time.time()
    print('(step 0/3) read local csv - main - running time: {}'.format(end - start))

    print(main.shape)
    main['datacqtr'] = pd.to_datetime(main['datacqtr'],format='%Y-%m-%d')

    # 1. add 20 lagging factors for each variable
    main_lag = add_lag(main, lag_year)
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

def sample_from_main(part=5):
    df = load_data()
    df = shuffle(df)

    part_len = len(df)//part
    part_dict = {}
    s = 0

    for i in range(part):
        df_part = df.iloc[s:(s + part_len), 3:-2]
        part_dict[i] = StandardScaler().fit_transform(df_part)
        s += part_len

    return part_dict

if __name__ == "__main__":

    # actual running scripts see def above -> for import
    import os
    os.chdir('/Users/Clair/PycharmProjects/HKP_ML_DL')

    main = load_data(lag_year=2)


    # dic = sample_from_main()
    # print(dic)




