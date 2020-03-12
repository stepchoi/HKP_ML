'''This code would import TABLE main and perform:

    1. add lag period
    2. add macro & y
    3. extract:
        a. by period desired
        b. by random sampling
    4. clean by:
        x -> standardization
        y -> qcut

'''
import datetime as dt
import gc
import time

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sqlalchemy import create_engine
from tqdm import tqdm


def check_print(df_list):
    df = pd.concat(df_list, axis=1)
    col = ['gvkey','datacqtr'] + [x for x in sorted(df.columns) if x not in ['gvkey','datacqtr']]
    df = df.reindex(col, axis=1)
    df.head(500).to_csv('check.csv')

    os.system("open -a '/Applications/Microsoft Excel.app' 'check.csv'")
    exit(0)

def convert_to_float32(df):

    ''''This def convert float64 to float32 to save memory usage.'''

    df.loc[:, df.dtypes == np.float64] = df.loc[:, df.dtypes == np.float64].astype(np.float32)

def add_lag(df, lag_year): # df is TABLE main, lag_year for original model design is 5 years

    ''''1. This def adds lagging periods by given lag_year.'''

    print('---------------------------- (step 1/3) adding lag -----------------------------')
    start = time.time()

    convert_to_float32(df)

    col = df.columns[2:]  # first three columns are gvkey, datacqtr(Timestamp), and sic which won't need lagging values

    lag_df = []  # create list for to be concated dataframes
    lag_df.append(df.dropna())  # current quarters, dropna remove records with missing important fields (e.g. niq....)

    for i in tqdm(range(lag_year*4-1)): # when lag_year is 5, here loop over past 19 quarter
        df_temp = df.groupby('gvkey').shift(i + 1)[col]
        df_temp.columns = ['{}_lag{}'.format(k, str(i+1).zfill(2)) for k in col] # name columns e.g. atq_lag01 -> last quarter total asset
        df_temp = df_temp.dropna(how='any')
        lag_df.append(df_temp)

    df_lag = pd.concat(lag_df, axis = 1, join='inner')

    end = time.time()
    print('(step 1/3) adding lag running time: {}'.format(end - start))
    print('after add lag: ', df_lag.shape)
    return df_lag

def merge_dep_macro(df, sql_version):

    ''''2. This def adds economic data (i.e. macro) & y.'''

    print('----------------- (step 2/3) adding macro & dependent variable -----------------')
    start = time.time()

    if sql_version is True: # sql version read TABLE from Postgre SQL
        db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
        engine = create_engine(db_string)
        dep = pd.read_sql('SELECT * FROM niq', engine)
        macro = pd.read_sql("SELECT * FROM macro_main", engine)
        stock  = pd.read_sql("SELECT * FROM stock_main", engine)
    else: # local version read TABLE from local csv files -> faster
        macro = pd.read_csv('macro_main.csv')
        dep = pd.read_csv('niq.csv')
        stock = pd.read_csv('stock_main.csv')
        print('local version running - niq, macro_main, stock_return')

    convert_to_float32(dep)
    convert_to_float32(macro)
    convert_to_float32(stock)

    dep['datacqtr'] = pd.to_datetime(dep['datacqtr'],format='%Y-%m-%d') # convert to timestamp
    macro['datacqtr'] = pd.to_datetime(macro['datacqtr'],format='%Y-%m-%d')
    stock['datacqtr'] = pd.to_datetime(stock['datacqtr'],format='%Y-%m-%d')

    merge_1 = pd.merge(stock, macro, on=['datacqtr'], how='left') # merge eco data & stock return by datacqtr
    merge_2 = pd.merge(merge_1, dep, on=['gvkey', 'datacqtr'], how='right') # add merge dependent variable

    print('delete')
    print(merge_2.shape)

    merge_2 = merge_2.dropna(how='any') # remove records with missing eco data

    del merge_1, dep, macro, stock
    gc.collect()

    merge_3 = pd.merge(df, merge_2, on=['gvkey', 'datacqtr'], how='left')
    merge_3 = merge_3.dropna(how='any') # remove records with missing eco data

    end = time.time()
    print('(step 2/3) adding macro & dependent variable running time: {}'.format(end - start))
    print('after add macro & dependent variable : ', merge_3.shape)

    return merge_3


class clean_set:

    '''4. This def converts x -> std, y -> qcut'''

    def __init__(self, train, test):

        def divide_set(df): # this funtion cut main df into df for x_variables, y_yoy, y_qoq by columns position
            return df.iloc[:, 2:-2].values, df.iloc[:, -2].values, df.iloc[:, -1].values

        self.train_x, self.train_qoq, self.train_yoy = divide_set(train)
        try:
            self.test_x, self.test_qoq, self.test_yoy = divide_set(test) # can work without test set
        except:
            pass
        print(self.train_x.shape)

    def standardize_x(self): # standardize x with train_x fit
        scaler = StandardScaler().fit(self.train_x)
        self.train_x = scaler.transform(self.train_x)
        try:
            self.test_x = scaler.transform(self.test_x) # can work without test set
            return self.train_x, self.test_x
        except:
            return self.train_x, None

    def yoy(self): # qcut y with train_y cut_bins
        self.train_yoy, cut_bins = pd.qcut(self.train_yoy, q=3, labels=[0, 1, 2], retbins=True)

        try:
            self.test_yoy = pd.cut(self.test_yoy, bins=cut_bins, labels=[0, 1, 2]) # can work without test set
            return self.train_yoy.astype(np.int8), self.test_yoy.astype(np.int8)
        except:
            return self.train_yoy.astype(np.int8), None

    def qoq(self): # qcut y with train_y cut_bins
        self.train_qoq, cut_bins = pd.qcut(self.train_qoq, q=3, labels=[0, 1, 2], retbins=True)
        try:
            self.test_qoq = pd.cut(self.test_qoq, bins=cut_bins, labels=[0, 1, 2]) # can work without test set
            return self.train_qoq.astype(np.int8), self.test_qoq.astype(np.int8)
        except:
            return self.train_qoq.astype(np.int8), None


def load_data(lag_year = 5, sql_version = False):

    '''This def consolidate steps 1 & 2 -> return big table with max(row) * max(col)'''

    # import engine, select variables, import raw database
    print('-------------- start load data into different sets (-> dictionary) --------------')
    start = time.time()

    if sql_version is True: # sql version read TABLE from Postgre SQL
        db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
        engine = create_engine(db_string)
        main = pd.read_sql('SELECT * FROM main', engine)
    else: # local version read TABLE from local csv files -> faster
        main = pd.read_csv('main.csv')
        engine = None
        print('local version running - main')

    end = time.time()
    print('(step 0/3) read local csv - main - running time: {}'.format(end - start))

    print(main.shape)
    main['datacqtr'] = pd.to_datetime(main['datacqtr'],format='%Y-%m-%d')

    # 1. add 20 lagging factors for each variable
    main_lag = add_lag(main, lag_year)
    del main
    gc.collect()

    # 2. add dependent variable & macro variables to main
    main_lag = merge_dep_macro(main_lag, sql_version) # i.e. big table

    def save_lag_to_csv(): # save big table as local csv -> not recommends, probably takes hours
        start = time.time()
        main_lag.to_csv('main_lag.csv', index=False)
        end = time.time()
        print('save csv running time: {}'.format(end - start))

    print(main_lag.info())
    return main_lag # i.e. big table

def train_test_clean(y_type, train, test = None): # y_type = ['yoy','qoq']; train, test(optional) are dataframes

    '''This def consolidate steps 4 -> return (train_x, test_x, train_y, test_y)'''

    class bcolors: # define color for waring
        WARNING = '\033[93m'
        FAIL = '\033[91m'

    if not y_type in ('yoy', 'qoq'): # send warning if y_type not yoy or qoq
        print(f"{bcolors.FAIL}y_type can only 'yoy' or 'qoq'.")
        exit(1)

    main_period = clean_set(train, test) # create class
    train_x, test_x = main_period.standardize_x() # for x

    if y_type == 'yoy': # for y
        train_y, test_y = main_period.yoy()
    elif y_type == 'qoq':
        train_y, test_y = main_period.qoq()

    return train_x, test_x, train_y, test_y

def sample_from_datacqtr(df, y_type, testing_period): # df = big table; y_type = ['yoy','qoq']; testing_period are timestamp

    '''3.a. This def extract partial from big table with selected testing_period'''

    end = testing_period
    start = testing_period - relativedelta(years=20) # define training period

    train = df.loc[(start <= df['datacqtr']) & (df['datacqtr'] < end)]  # train df = 80 quarters
    test = df.loc[df['datacqtr'] == end]                                # test df = 1 quarter

    return train_test_clean(y_type, train, test)

def sample_from_main(df, y_type, part=5): # df = big table; y_type = ['yoy','qoq']; part = cut big table into how many parts

    '''3.b. This def extract partial from big table by random sampling'''

    df = shuffle(df) # shuffle big table

    part_len = len(df) // part # find length for each parts
    dfs = {}
    s = 0

    for i in range(part):
        set = df.iloc[s:(s + part_len)] # extract from big table
        train_x, test_x, train_y, test_y = train_test_clean(y_type, set) # here has no test set, only enter each set as training sets
        dfs[i] = (train_x, train_y)
        s += part_len

        del train_x, test_x, train_y, test_y
        gc.collect()

    return dfs

def trial_main():

    '''this code run above funtions for trail run'''

    # actual running scripts see def above -> for import
    import os
    os.chdir('/Users/Clair/PycharmProjects/HKP_ML_DL')

    # 1. return main dateframe
    main = load_data(lag_year=1)

    # 2.1 if want to return (train_x, test_x, train_y, test_y) by given testing_period
    train_x, test_x, train_y, test_y = sample_from_datacqtr(main, y_type = 'yoy', testing_period = dt.datetime(2008, 3, 31))
    print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

    exit(0)

    # 2.2 if want to return (train_x, train_y) by randomly sampled from main df
    '''dfs is dictionary contains all set of (train_x, train_y)'''
    dfs = sample_from_main(main, y_type = 'yoy',part = 3)

    for k in dfs.keys():
        x, y = dfs[k]
        print(type(x))
        print(type(y))
        print(y)


if __name__ == "__main__":


    # read and return x, y from 150k (entire space)
    main = load_data(lag_year=5)
    x, y = sample_from_main(main, y_type='yoy', part=1)[0]  # change to 'qoq' and run again !!
    x_qoq, y_qoq = sample_from_main(main, y_type='qoq', part=1)[0]  # change to 'qoq' and run again !!

    # col = main.columns

    print('1. check chronological sequence ')
    print(len(main))
    from PrepareDatabase import drop_nonseq
    drop_nonseq(main)
    print(len(main))
    del main['datacqtr_no']

    # print(main.groupby(['gvkey', 'datacqtr']).filter(lambda x: len(x) > 1))

    # df_1 = main.filter(['gvkey', 'datacqtr'])
    # print(len(set(main['gvkey'])))
    # df_1['exist'] = 1
    # df = df_1.pivot(index='gvkey', columns='datacqtr', values='exist')
    # df.to_csv('check_chron.csv')
    #
    # k = 0
    # for i, row in df.iterrows():
    #     l = row.to_list()
    # print(len(df), k)


    # print('check columns in main')
    # print(col)
    # print(main.info())


    # print('2. check NaN in main')
    # print(main.isnull().sum().sum())


    print('3. check standardize in main')
    print(pd.DataFrame(x).iloc[:,:5])
    print(pd.DataFrame(x_qoq).iloc[:,:5])
    print(x == x_qoq)
    pd.DataFrame(x).describe().to_csv('describe_main.csv')


    print('4. check # of [0,1,2] in y')
    from collections import Counter
    print(type(y), Counter(y))
    print(type(y_qoq), Counter(y_qoq))


    print('6. check random classification')
    from sklearn.model_selection import train_test_split

    x_label = pd.concat([main[['gvkey', 'datacqtr']], pd.DataFrame(x)], axis=1)
    x_lgbm, x_test, y_lgbm, y_test = train_test_split(x_label, y, test_size=0.2)
    x_train, x_valid, y_train, y_valid = train_test_split(x_lgbm, y_lgbm, test_size=0.25)

    t0 = x_train['gvkey', 'datacqtr']
    t0['split'] = 0

    t1 = x_valid['gvkey', 'datacqtr']
    t1['split'] = 1

    t2 = x_test['gvkey', 'datacqtr']
    t2['split'] = 2

    df_2 = pd.concat([t0, t1, t2], axis=0)
    df_2.pivot(index='gvkey', columns='datacqtr', values='split').to_csv('check_chron_split')






