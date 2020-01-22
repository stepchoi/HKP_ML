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
        main = pd.read_csv('main_rolling.csv') # change forward/rolling for two different fillna version
        engine = None
        print('local version running')
    except:
        db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
        engine = create_engine(db_string)
        main = pd.read_sql('SELECT * FROM main_forward', engine) # change forward/rolling for two different fillna version

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
    full_running_cut()

