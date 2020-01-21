import pandas as pd
from sqlalchemy import create_engine


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


def merge_dep_macro(df, dependent_variable='epspxq_nextq_abs'):
    macro = pd.read_sql("SELECT * FROM macro_clean", engine)
    macro_lst = macro.columns[2:]
    dep = pd.read_sql('SELECT gvkeydatafqtr, ' + dependent_variable + ', selected FROM main_dependent', engine)

    df_1 = pd.merge(df, dep, on=['gvkeydatafqtr'], how='left')
    df_1 = pd.merge(df_1, macro, on=['cyear', 'cquarter'], how='left')

    return df_1, macro_lst


def add_lag(df):

    col = df.iloc[:, 3:].columns

    for i in range(19):
        print(i)
        namelag = [(k + '_lag' + str(i+1).zfill(2)) for k in col]
        df[namelag] = df.iloc[:, 3:].groupby('gvkey').shift(i + 1)[col]

    return df

def cut_test_train(df):

    def datacqtr_to_no(df):
        cqtr = pd.DataFrame(set(df['datacqtr']), columns=['datacqtr']).sort_values(by=['datacqtr']).reset_index(
            drop=True).reset_index(drop=False).set_index('datacqtr')
        df['datacqtr_no'] = df.datacqtr.map(cqtr['index'])
        print(cqtr.loc['2008Q1'])
        return df

    df = datacqtr_to_no(df)
    test_train_dict = {}
    for testing_cqtr in range(113, 154) # datacqtr_no for 2008Q1 ~ (2018Q4 + 1)
        test_train_dict[testing_cqtr - 113] = {}
        test_train_dict[testing_cqtr - 113]['test'] = df.loc[df['datacqtr_no'] == testing_cqtr]
        test_train_dict[testing_cqtr - 113]['train'] = df.loc[df['datacqtr_no'].isin(range(testing_cqtr - 20, testing_cqtr, 1))]

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

    return test_train_dict

if __name__ == "__main__":

    # actual running scripts see def above -> for import
    full_running_cut()

