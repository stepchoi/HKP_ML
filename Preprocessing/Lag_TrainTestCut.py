import datetime as dt
import time

import pandas as pd
from Miscellaneous import save_load_dict
from dateutil.relativedelta import relativedelta
from sqlalchemy import create_engine
from tqdm import tqdm


def delete_high_zero_row(missing_dict):
    df_zero_series = pd.concat(missing_dict, axis = 1).sum(axis=1)
    print(df_zero_series)

    def print_csv(df_zero_series):
        from collections import Counter

        c = Counter(df_zero_series)
        csum = 0
        missing_dict = {}
        for missing, count in sorted(c.items()):
            csum += count
            missing_dict[missing] = csum

        df = pd.DataFrame.from_dict(missing_dict, orient='index', columns=['count'])
        df['%_count'] = df['count'] / 333325
        df['changes'] = df['%_count'].sub(df['%_count'].shift(1))

        def double_plot(df, axis1, axis2):

            # plot dataframe by two columns, axis1 = 'red', axis2 = 'blue'
            import matplotlib.pyplot as plt

            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            ax1.plot(df[axis1], color='tab:red')
            ax1.tick_params(axis='y', labelcolor='tab:red')

            ax2.plot(df[axis2], color='tab:blue')
            ax2.tick_params(axis='y', labelcolor='tab:red')

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.show()
            plt.savefig('double_plot_delete_row.png')

        double_plot(df, '%_count', 'changes')

        df.to_csv('missing_below_threshold.csv')
        id = df.loc[df['%_count'] < 0.9].sort_values(by=['%_count']).tail(1).index
        print(id)
        return id

    df = df.loc[df_zero_series < print_csv(df_zero_series)]

    return df

def add_lag(df):
    print('---------------------------- adding lag -----------------------------')
    start = time.time()

    col = df.columns[3:]
    lag_df = []
    lag_df.append(df)

    # missing_dict = []
    # missing_dict.append(df.mask(df == 0).isnull().sum(axis=1))

    for i in tqdm(range(19)): # change to 19
        df_temp = df.groupby('gvkey').shift(i + 1)[col]
        df_temp.columns = ['{}_lag{}'.format(k, str(i+1).zfill(2)) for k in col]
        lag_df.append(df_temp)
        # missing_dict.append(df_temp.mask(df_temp == 0).isnull().sum(axis=1))

    # delete_high_zero_row(missing_dict)
    df_lag = pd.concat(lag_df, axis = 1)
    print(df_lag.shape)

    end = time.time()
    print('adding lag running time: {}'.format(end - start))
    return df_lag


def merge_dep_macro(df):
    print('----------------- adding macro & dependent variable -----------------')
    start = time.time()

    try:
        dep = pd.read_csv('/Users/Clair/PycharmProjects/HKP_ML_DL/Preprocessing/niq.csv')
        print('local version running - niq')
    except:
        # macro = pd.read_sql("SELECT * FROM macro_clean", engine)
        dep = pd.read_sql('SELECT * FROM niq', engine)

    dep['datacqtr'] = pd.to_datetime(dep['datacqtr'],format='%Y-%m-%d')

    # df_macro = pd.merge(df, macro, on=['datacqtr'], how='left')
    df_macro_dep = pd.merge(df, dep, on=['gvkey', 'datacqtr'], how='left')  # change to df_macro

    end = time.time()
    print('adding macro & dependent variable running time: {}'.format(end - start))

    return df_macro_dep


def div_x_y(set_dict, sub_cut_bins):

    # 1: divide x and y
    def divide(df):
        return df.iloc[:, 3:-2].values, df.iloc[:, -2].values, df.iloc[:, -1].values

    set_dict['test_x'], set_dict['test_qoq'], set_dict['test_yoy']  = divide(set_dict['test'])
    set_dict['train_x'], set_dict['train_qoq'], set_dict['train_yoy'] = divide(set_dict['train'])

    # 2: Standardization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(set_dict['train_x'])
    set_dict['train_x'] = scaler.transform(set_dict['train_x'])
    set_dict['test_x'] = scaler.transform(set_dict['test_x'])

    # 3: qcut
    sub_cut_bins = {}
    for y in ['qoq', 'yoy']:
        set_dict['train_' + y], cut_bins = pd.qcut(set_dict['train_' + y], q=3, labels=[0,1,2], retbins=True)
        set_dict['test_' + y] = pd.cut(set_dict['test_' + y], bins=cut_bins, labels=[0,1,2])
        sub_cut_bins[y] = cut_bins

    set_dict['test'] = None
    set_dict['train'] = None
    return set_dict, sub_cut_bins

def cut_test_train(df):
    print('------------------- cutting testing/training set --------------------')
    start_total = time.time()

    dict = {}
    cut_bins = {}
    testing_period = dt.datetime(2008, 3, 31)

    for i in tqdm(range(40)): # here should be -> 40 -> for entire sets
        '''training set: x -> standardize -> apply to testing set: x
            training set: y -> qcut -> apply to testing set: y'''
        end = testing_period + i*relativedelta(months=3)
        start = testing_period - relativedelta(years=20)
        print(i+1, end)

        dict[i+1] = {}
        dict[i+1]['test'] = df.loc[df['datacqtr'] == end]
        dict[i+1]['train'] = df.loc[(start <= df['datacqtr']) & (df['datacqtr'] < end)]

        dict[i+1], cut_bins[i+1] = div_x_y(dict[i+1])
        # pd.DataFrame(dict[set_no]['train_x']).to_csv('train_x_set{}.csv'.format(i), index = False, header = False)

    save_load_dict('save', dict=cut_bins, name='cut_bins') # save cut_bins to dictionary

    end_total = time.time()
    print('cutting testing/training set running time: {}'.format(end_total - start_total))

    return dict

def full_running_cut():

    # import engine, select variables, import raw database
    print('-------- start load data into different sets (-> dictionary) --------')

    try:
        main = pd.read_csv('/Users/Clair/PycharmProjects/HKP_ML_DL/Preprocessing/main.csv')
        engine = None
        print('local version running - main')
    except:
        db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
        engine = create_engine(db_string)
        main = pd.read_sql('SELECT * FROM main', engine)

    print(main.shape)
    main['datacqtr'] = pd.to_datetime(main['datacqtr'],format='%Y-%m-%d')

    # 1. add 20 lagging factors for each variable
    main_lag = add_lag(main)
    del main

    # 2. add dependent variable & macro variables to main
    main_lag = merge_dep_macro(main_lag)

    start = time.time()
    main_lag = main_lag.dropna()
    print(main_lag.shape)
    end = time.time()
    print('dropna running time: {}'.format(end - start))

    start = time.time()
    main_lag.to_csv('main_lag.csv', index=False)
    end = time.time()
    print('save csv running time: {}'.format(end - start))

    # 3. cut training, testing set
    test_train_dict = cut_test_train(main_lag)

    return test_train_dict

if __name__ == "__main__":

    # actual running scripts see def above -> for import
    full_running_cut()
