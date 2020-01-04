def select_variable():

    # select numeric variables -> dict
    # select label variables -> list
    # set 1 fillna classification -> dict

    format_map = pd.read_sql("SELECT * FROM format_map", engine)

    numeric_dict = {}
    for name, g in format_map.groupby('format'):
        numeric_dict[name] = g['abbreviation'].dropna().to_list()

    fillna1_dict = {}
    for name, g in format_map.groupby('fillna'):
        fillna1_dict[name] = g['abbreviation'].dropna().to_list()

    label_lst = ['gvkey', 'datacqtr', 'sic', 'cquarter', 'cyear', 'gvkeydatafqtr','datacqtr_no']

    return numeric_dict, label_lst, fillna1_dict

def convert_format(df, dict = numeric_dict, label = label_lst):

    # convert raw dataset to desired formats (YoY, QoQ, Log)
    # groupby 'gvkey'

    C1 = df.filter(label)
    C2 = df.groupby('gvkey').apply(lambda x: x[dict['QoQ']].div(x[dict['QoQ']].shift(1)).sub(1))
    C3 = df.groupby('gvkey').apply(lambda x: x[dict['YoY']].div(x[dict['YoY']].shift(4)).sub(1))
    C4 = np.log(df[dict['Log']]+1)

    df_1 = pd.concat([C1, C2, C3, C4], axis = 1)
    df_1 = df_1.replace([np.inf, -np.inf], np.nan)
    return df_1

def drop_seq(df):
    cqtr = pd.DataFrame(set(df['datacqtr']),
                        columns=['datacqtr']).sort_values(by=['datacqtr']).reset_index(drop=True)
    cqtr.columns = ['datacqtr_no', 'datacqtr']
    df_1 = pd.merge(df, cqtr, on=['datacqtr'], how='left')
    df_1 = df_1.sort_values(by=['gvkey', 'datacqtr'])
    df_1['index_seq'] = df_1.groupby('gvkey').apply(lambda x: (x['datacqtr_no'].shift(-1) - x['datacqtr_no'])).to_list()
    df_1['index_seq'] = df_1['index_seq'].fillna(1).astype(int)
    del_gvkey = set(df_1.loc[df_1['index_seq']!=1,'gvkey'])
    df_1 = df_1.loc[~df['gvkey'].isin(del_gvkey)]
    del df_1['index_seq']
    return df_1

# fillna methods

def fillna_0(df, col):
    df = df.copy(1)
    df[col] = df[col].fillna(0)
    return df

def fillna_forward(df, col):
    df = df.copy(1)
    groups = []
    for name, g in df.groupby('gvkey'):
        groups.append(g[col].apply(lambda series: series.loc[ :series.last_valid_index()].ffill())])
    return pd.concat(groups, axis = 0)

def fillna_individual(df, dict):
    df = df.copy(1)
    df[dict[0]] = df[dict[0]].fillna(0)
    for i in fillna_dict['epspxq']:
        df[i] = df[i].fillna(df['epspxq'])
    return df

del fillna_rolling_average(df, col):
    df = df.copy(1)
    df[col] = df.apply(lambda x: x.fillna(x.rolling(12, center=True, min_periods=1).mean())).filter(col)
    return df

def industry_average(df):
    ff48 = pd.read_sql("SELECT group_no, sic_beg, sic_end FROM industry_ff48_sic", engine)
    ff48_arr =
    df['ff_sic'] = df.sic.


def check_correlation(df, threshold=0.9):

    # find high correlated items -> excel

    ex_corr_col = ['epspxq', 'revtq', 'lltq', 'ltq',
                   'lctq', 'ancq', 'intanq', 'seqq', 'xoprq'] + label_col

    def high_corr(df, threshold=0.9):
        corr = df.corr().abs()
        s = corr.unstack()
        so = s.sort_values(
            kind="quicksort", ascending=False).to_frame().reset_index()
        so.columns = ['v1', 'v2', 'corr']
        so = so.loc[(so['v1'] != so['v2']) & (so['corr'] > threshold)
                    ].drop_duplicates(subset=['v1', 'v2'])
        return so

    corr_matrix = df.corr()
    high_corr_df = high_corr(df)

    # Set of all the names of deleted columns
    col_corr = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    del_corr = col_corr - set(ex_corr_col)

    # DataFrame of high correlated, description, and values
    del_value = high_corr_df.loc[(high_corr_df['v1'].isin(del_corr)) | (high_corr_df['v2'].isin(del_corr))]
    del_col_ie = pd.merge(del_value, format_map, left_on=['v1'], right_on=['abbreviation'], how='left')
    del_col_ie = pd.merge(del_col_ie, format_map, left_on=['v2'], right_on=['abbreviation'], how='left')
    del_col_ie = del_col_ie.loc[((del_col_ie['abbreviation_x'].isin(del_corr))) &
                                (~(del_col_ie['abbreviation_y'].isin(del_corr)))].sort_values(by='abbreviation_x').iloc[:, 2:]
    return del_corr, del_col_ie


def merge_dep_macro(df, dependent_variable = 'epspxq_nextq_abs'):

    macro = pd.read_sql("SELECT * FROM macro_clean", engine)
    macro_lst = macro.columns[2:]
    dep = pd.read_sql('SELECT gvkeydatafqtr, ' + dependent_variable + ', selected FROM main_dependent', engine)

    df_1 = pd.merge(df, dep, on =['gvkeydatafqtr'],how='left')
    df_1 = pd.merge(df_1, macro, on=['cyear', 'cquarter'], how='left')

    return df_1, macro_lst

def add_lag(df, col):

    def namelag(lst, i):
        return [(k + '_lag' + str(i).zfill(2)) for k in lst]

    for i in range(19):
        print(i)
        df[namelag(col,i+1)]  = df.groupby('gvkey').shift(i+1)[col]
    return df

def cut_train_test(df, testing_cqtr):
    df_train = df.loc[df['datacqtr_no'].isin(range(testing_cqtr - 20, testing_cqtr, 1))]
    df_test = df.loc[df['datacqtr_no'] == testing_cqtr]
    return df_train, df_test

class preprocessing_class:

     # import engine, select variables, import raw database

    from sqlalchemy import create_engine
    import pandas as pd
    import numpy as np

    db_string = 'postgres://postgres:DLvalue123@hkpolyu-dl-value.c1lrltigx0e7.us-east-1.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    numeric_dict, label_lst, fillna1_dict = select_variable()

    raw = pd.read_sql("SELECT * FROM raw_109", engine)
    raw = drop_seq(raw)
    raw = convert_format(raw) # convert to chosen formats
    main, macro_lst = merge_dep_macro(raw) # add dependent variable & macro variables to main


    def fillna_set1():

        # 1.1: fill YoY, QoQ -> 0
        col1 = numeric_dict['YoY'] + numeric_dict['QoQ']
        main = fillna_0(self.main, col1)

        # 1.2: fill Log, Abs -> forward fill
        col2 = numeric_dict['Abs'] + numeric_dict['Log']
        main = fillna_forward(main, col2)

        # 1.3: fill Log, Abs -> individual (refer to excel file 'explanation.xlsx'-'1.2 Fillna')
        main = fillna_individual(main, fillna1_dict)

        # 2: add lagging
        num_col = numeric_dict['YoY'] + numeric_dict['QoQ'] + numeric_dict['Abs'] + numeric_dict['Log']
        main_lag = add_lag(df, num_col).dropna(how='any', axis=0)

        # 3: split test, training set
        pca_dict = {}
        pca_dict['train'] = {}
        pca_dict['test'] = {}

        for i in range(111, )
            pca_dict['train']






        main = main.dropna(subset=dict['delete rows'], how='any', axis=0)

