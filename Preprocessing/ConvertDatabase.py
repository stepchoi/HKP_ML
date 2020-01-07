
'''
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

        for i in range(111):
            pass


def eda_hist(df,name='temp'):
    from matplotlib import pyplot as plt
    import math

    fig = plt.figure(figsize=(50, 50), dpi=80)
    plt.rcParams.update({'font.size': 6})
    k=1
    for col in df.columns.to_list():
        n = math.ceil(len(df.columns)**0.5)
        axis1 = fig.add_subplot(n,n,k)
        axis1.hist(df.loc[df[col].notnull(),col],density = True, bins=50)
        axis1.set_title(col, fontsize = 60)
        print(col,k)
        k += 1
    fig.tight_layout()
    fig.savefig(name+'.png')
'''
def select_variable(engine):

    import pandas as pd
    # create dictionary for all selected variables by different formats(yoy, qoq, nom, log), features

    format_map = pd.read_sql("SELECT * FROM format_map", engine)

    select = {}
    select['all'] = format_map['name'].to_list()
    select['label'] = ['gvkey', 'datacqtr', 'sic']

    for col in format_map.columns[2:-1]:
        select[col] = format_map.loc[format_map[col]==1, 'name'].to_list()

    select.update(format_map.loc[format_map['special'].notnull()].filter(['name','special']).set_index('name').to_dict())

    return select


def convert_format(df, dic):

    # convert raw dataset to desired formats (YoY, QoQ, Log)
    # groupby 'gvkey'

    convert_select = {}

    for k in ['yoy','qoq','log']:
        convert_select[k] = [x + '_' + k for x in dic[k]]

    label_nom = df.filter(dic['label'] + dic['nom'])

    # convert dividend to rolling 4 period sum
    df['dvy_q'] = df.groupby('gvkey').apply(lambda x: x['dvy_q'].rolling(4, min_periods=1).sum()).to_list()

    qoq_col = dic['qoq']
    qoq = df.groupby('gvkey').apply(lambda x: x[qoq_col].div(x[qoq_col].shift(1)).sub(1))
    qoq.columns = convert_select['qoq']

    yoy_col = dic['yoy']
    yoy = df.groupby('gvkey').apply(lambda x: x[yoy_col].div(x[yoy_col].shift(4)).sub(1))
    yoy.columns = convert_select['yoy']

    log_col = dic['log']
    log = np.log(df[log_col].div(df['atq']).add(1).replace([np.inf, -np.inf], np.nan))
    log.columns = convert_select['log']

    dic.update(convert_select)

    df_1 = pd.concat([label_nom, yoy, qoq, log], axis = 1)
    df_1 = df_1.replace([np.inf, -np.inf], np.nan)

    return df_1, dic

# fillna methods
class fillna:

    def __init__(self, df, dic):

        # 1.1: fill YoY, QoQ -> 0
        col1 = dic['yoy'] + dic['qoq']
        df[col1] = df[col1].fillna(0)

        # 1.2: fill more than 8 period
        self.col = set(dic['abs'] + dic['log']) - set(dic['delete_row'])

        def after_8(series):

            index_nan = [item for sublist in np.argwhere(np.isnan(series)) for item in sublist]

            index_nan_df = pd.DataFrame(index_nan)
            index_nan_df['sub'] = index_nan_df[0].sub(index_nan_df[0].shift(1))
            begin_nan = index_nan_df.loc[index_nan_df['sub'] != 1, 0].values

            for i in range(8):
                begin_nan_8 = []
                begin_nan_8.extend(begin_nan + i)
            fill0_nan = list(set(index_nan) - set(begin_nan_8))
            series[fill0_nan] = 0

        df[self.col] = df[self.col].apply(after_8)
        self.df = df

    def forward(self):
        self.df[self.col] = self.df.groupby('gvkey').apply(lambda x: x[self.col].ffill(limit = 8))
        df[self.col] = df[self.col].fillna(0)
        return self.df

    def rolling(self):
        self.df[self.col] = self.df.apply(lambda x: x.fillna(x[self.col].rolling(12, min_periods=1).mean()))
        df[self.col] = df[self.col].fillna(0)
        return self.df


if __name__ == "__main__":

    from sqlalchemy import create_engine
    import pandas as pd
    import numpy as np

    # import engine, select variables, import raw database
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    select = select_variable(engine) # dictionary with variables based on selection criteria
    all_col =  select['label'] + select['all']
    pd.DataFrame(all_col).to_csv('all_col.csv', index = False)

    raw = pd.read_sql_table('raw', engine, columns = all_col)
    raw = pd.merge(raw, sic, on=['gvkey','datacqtr'], how = 'left')
    print(raw.isnull().sum().sort_values())

    raw[select['all']] = raw[select['all']].astype(float)

    raw = drop_nonseq(raw)
    raw.to_sql('select_raw', engine)

    select = select_variable() # dictionary with variables based on selection criteria
    select.update({'label': (select['label'] + ['datacqtr_no'])})

    main, select = convert_format(raw, select)

    main = fillna(main, select).fillna_forward()
    import missingno as msno
    print(msno.bar(main))
    # main.to_sql('main_forward')

    # main = fillna(main, select).forward()
    # main.to_sql('main_forward')

