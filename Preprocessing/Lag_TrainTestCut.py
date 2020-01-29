from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm


def delete_high_zero_row(df):
    # print(df.isnull().sum().sum())
    df_zero = df.mask(df==0).isnull().sum(axis = 1)
    print(df_zero.sum())

    def print_csv():
        c = Counter(df_zero)
        csum = 0
        missing_below_threshold = {}
        for missing, count in sorted(c.items()):
            csum += count
            missing_below_threshold[missing] = csum
        df = pd.DataFrame.from_dict(missing_below_threshold,orient = 'index', columns = ['count'])
        df['%_count'] = df['count']/333325
        df['changes'] = df['%_count'].sub(df['%_count'].shift(1))
        plt.plot(df[['%_count','changes']])
        plt.show()

        df.to_csv('missing_below_threshold.csv')
        return df.loc[df['%_count'] < 0.9].sort_values(by = ['%_count']).tail(1).index

    id = print_csv()
    print(id)
    df = df.loc[df_zero<144]
    # df.to_csv('main_del_row.csv')
    print(df.shape)

    return df

def add_lag(df):
    print('----- adding lag -----')
    col = df.iloc[:, 3:].columns

    for i in tqdm(range(19)):
        namelag = [(k + '_lag' + str(i+1).zfill(2)) for k in col]
        df[namelag] = df.groupby('gvkey').shift(i + 1)[col]
    return df


def merge_dep_macro(df, dependent_variable='next1_abs'): #dependent variables includes: next1_abs, next4_abs, epspxq_qoq, epspxq_yoy
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    print('----- adding macro & dependent variable -----')
    macro = pd.read_sql("SELECT * FROM macro_clean", engine)
    macro['datacqtr'] = ['{}Q{}'.format(y, q) for q, y in zip(macro['cquarter'], macro['cyear'])]
    macro_lst = macro.columns[2:]

    dep = pd.read_sql('SELECT gvkey, datacqtr, ' + dependent_variable + ' selected FROM epspxq', engine)

    for col in ['gvkey', 'datacqtr']:
        df[col] = df[col].astype(str)
        dep[col] = dep[col].astype(str)

    df_1 = pd.merge(df, dep, on=['gvkey', 'datacqtr'], how='left')
    df_1 = pd.merge(df_1, macro, on=['datacqtr'], how='left')

    return df_1, macro_lst


def cut_test_train(df):

    print('----- start cutting testing/training set -----')
    df = datacqtr_to_no(df)
    test_train_dict = {}
    training_ends = '2007-12-31'
    for testing_cqtr in tqdm(range(116, 157)): # datacqtr_no for 2008Q1 = 116 ~ (2018Q4 = 156 + 1)
        test_train_dict[testing_cqtr - 116] = {}
        test_train_dict[testing_cqtr - 116]['test'] = df.loc[df['datacqtr_no'] == testing_cqtr]
        test_train_dict[testing_cqtr - 116]['train'] = df.loc[df['datacqtr_no'].isin(range(testing_cqtr - 20, testing_cqtr, 1))]

    return test_train_dict


def full_running_cut():

    # import engine, select variables, import raw database
    try:
        main = pd.read_csv('main.csv') # change forward/rolling for two different fillna version
        engine = None
        print('local version running - main')
    except:
        db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
        engine = create_engine(db_string)
        main = pd.read_sql('SELECT * FROM main', engine) # change forward/rolling for two different fillna version


    # 2. add 20 lagging factors for each variable
    main_lag = add_lag(main).dropna(how='any', axis=0)
    main_lag = delete_high_zero_row(main_lag)
    print(main_lag.shape)

    # # 3. add dependent variable & macro variables to main
    # main, macro_lst = merge_dep_macro(main_lag)
    #
    # # 4. cut training, testing set
    # test_train_dict = cut_test_train(main)
    # print(test_train_dict)
    #
    # return test_train_dict

if __name__ == "__main__":

    # actual running scripts see def above -> for import
    full_running_cut()