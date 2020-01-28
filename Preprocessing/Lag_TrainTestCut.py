from collections import Counter

import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm


def check_correlation(df, threshold=0.9):
    # find high correlated items -> excel

    def high_corr(df, threshold=0.9):
        corr = df.corr().abs()
        so = corr.unstack().reset_index()
        print(so)
        # so = so.sort_values(kind="quicksort", ascending=False).to_frame().reset_index()
        so.columns = ['v1', 'v2', 'corr']
        so = so.loc[(so['v1'] != so['v2']) & (so['corr'] > threshold)].drop_duplicates(subset=['v1', 'v2'])
        return so

    high_corr_df = high_corr(df)
    print(high_corr_df)

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
        pd.DataFrame.from_dict(missing_below_threshold,orient = 'index', columns = ['count'])\
            .to_csv('missing_below_threshold.csv')

    # print_csv()
    df = df.loc[df_zero<144]
    df.to_csv('main_del_row.csv')
    print(df.shape)
    return df

def delete_high_zero_columns(df):
    # print(df.isnull().sum().sum())
    df_zero = df.mask(df==0).isnull().sum(axis = 0).sort_values()
    low_zero_col = df_zero[df_zero<318310].index.to_list()
    df = df.filter(low_zero_col)
    return df
    # pd.DataFrame(df_zero, columns = ['count']).to_csv('missing_below_threshold_columns.csv')


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

    def datacqtr_to_no(df):
        cqtr = pd.DataFrame(['{}Q{}'.format(y,q) for y in range(1979, 2020) for q in range(1,5)]).reset_index().set_index(0)
        df['datacqtr_no'] = df.datacqtr.map(cqtr['index'])
        return df

    print('----- start cutting testing/training set -----')
    df = datacqtr_to_no(df)
    test_train_dict = {}
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
        print('local version running')
    except:
        db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
        engine = create_engine(db_string)
        main = pd.read_sql('SELECT * FROM main', engine) # change forward/rolling for two different fillna version

    # check_correlation(main.iloc[:,2:])

    # 1. delete high correlation items
    del_corr = ['xsgaq_qoq', 'gdwlq_atq', 'cogsq_qoq']  # same for both forward & rolling version
    main = main.drop(main[del_corr], axis=1)

    # 2. add 20 lagging factors for each variable
    main_lag = add_lag(main).dropna(how='any', axis=0)

    # 3. add dependent variable & macro variables to main
    main, macro_lst = merge_dep_macro(main_lag)

    # 4. cut training, testing set
    test_train_dict = cut_test_train(main)
    print(test_train_dict)

    return test_train_dict

if __name__ == "__main__":

    # actual running scripts see def above -> for import
    main = pd.read_csv('main.csv')
    print(main.describe().transpose())

    # 1. delete high correlation items
    # del_corr = ['xsgaq_qoq', 'gdwlq_atq', 'cogsq_qoq']  # same for both forward & rolling version
    # main = main.drop(main[del_corr], axis=1)
    #
    # # main_del_col = delete_high_zero_columns(main)
    # main_del_row = delete_high_zero_row(main)
