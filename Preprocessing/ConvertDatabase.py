import numpy as np
import pandas as pd
from Miscellaneous import (Timestamp)
from sqlalchemy import create_engine
from tqdm import tqdm


def select_variable(engine):

    # create dictionary for all selected variables by different formats(yoy, qoq, nom, log), features
    try:
        format_map = pd.read_csv('format_map.csv')
    except:
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

    for k in ['yoy','qoq','atq','revtq']:
        convert_select[k] = [x + '_' + k for x in dic[k]]

    label_nom = df.filter(dic['label'] + dic['nom'])
    print('------ start conversion -------')

    # special: convert dividend to rolling 4 period sum
    df['dvy_q'] = df.groupby('gvkey').apply(lambda x: x['dvy_q'].rolling(4, min_periods=1).sum()).reset_index(drop = True)
    print('finish dividends rolling accural conversion')

    qoq = df.groupby('gvkey').apply(lambda x: x[dic['qoq']].div(x[dic['qoq']].shift(1)).sub(1)).reset_index(drop = True)
    qoq.columns = convert_select['qoq']
    print('finish qoq conversion')

    yoy = df.groupby('gvkey').apply(lambda x: x[dic['yoy']].div(x[dic['yoy']].shift(4)).sub(1)).reset_index(drop = True)
    yoy.columns = convert_select['yoy']
    print('finish yoy conversion')

    atq = np.log(df[dic['atq']].apply(lambda x: x.div(df['atq']).add(1).replace([np.inf, -np.inf], np.nan)))
    atq.columns = convert_select['atq']
    print('finish atq conversion')

    revtq = np.log(df[dic['revtq']].apply(lambda x: x.div(df['revtq']).add(1).replace([np.inf, -np.inf], np.nan)))
    revtq.columns = convert_select['revtq']
    print('finish revtq conversion')

    dic.update(convert_select)

    df_1 = pd.concat([label_nom, qoq, yoy, atq, revtq], axis = 1)
    df_1 = df_1.replace([np.inf, -np.inf], np.nan)
    df_1 = Timestamp(df_1)

    def missing_count(df):
        df = pd.DataFrame(df.isnull().sum(), columns = ['missing']).reset_index(drop = False)
        sp = pd.DataFrame([x.rsplit('_', 1) for x in df['index']])
        df[['name', 'format']] = sp
        df.to_csv('df_missing.csv')
        print('export df_missing.csv')

    missing_count(df_1)
    df_1.to_csv('main_convert.csv', index=False)

    print(df_1.shape)
    return df_1, dic

def convert():

    # import engine, select variables, import raw database
    try:
        raw = pd.read_csv('raw_main.csv')
        engine = None
        print('local version running')
    except:
        db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
        engine = create_engine(db_string)
        raw = pd.read_sql('SELECT * FROM raw_main', engine)

    select = select_variable(engine) # dictionary with variables based on selection criteria
    main, select = convert_format(raw, select)
    return main

def optimal_rolling_period(df):

    diff_dict = {}
    for p in tqdm([1,4,8,12,16,20]):
        df_rolling = df.groupby('gvkey').apply(lambda x: x.rolling(p, min_periods=1).mean().shift(1))
        diff_dict[p] = df_rolling.sub(df).pow(2).mean(axis=0)

    rolling_period_series = pd.DataFrame.from_dict(diff_dict, orient='index').idxmin()
    pd.DataFrame(rolling_period_series).to_csv('rolling_period_1.csv')

    return rolling_period_series

# fillna methods
def fillna(df, rolling_period):

    # 1: fill YoY, QoQ -> 0
    print('------ start fillna -------')
    col = df.columns
    yoy_qoq_col = [x for x in col if (('yoy' in x) or ('qoq' in x))]

    del_row = pd.read_sql("SELECT name, delete_row FROM format_map", engine)
    del_row = del_row.dropna(how = 'any')['name']
    del_row = [x + '_atq' for x in del_row] + [x + '_revtq' for x in del_row] + del_row.to_list()

    df[yoy_qoq_col] = df[yoy_qoq_col].fillna(0)

    # 2: fill 0 after 8th
    def after_8(series):  # for fillna
        index_nan = [item for sublist in np.argwhere(np.isnan(series)) for item in sublist]

        index_nan_df = pd.DataFrame(index_nan)
        index_nan_df['sub'] = index_nan_df[0].sub(index_nan_df[0].shift(1))
        begin_nan = index_nan_df.loc[index_nan_df['sub'] != 1, 0].values

        begin_nan_8 = []
        for i in range(8):
            begin_nan_8.extend(begin_nan + i)

        fill0_nan = list(set(index_nan) - set(begin_nan_8))
        series[fill0_nan] = 0

        return series

    rest_col = list(set(col) - set(del_row) - set(['gvkey','datacqtr','sic']) - set(yoy_qoq_col))
    df[rest_col] = df[rest_col].apply(after_8)


    # 3: rolling average for rest except delete_row
    print(' ----- start fillna rolling -----')

    for i in set(rolling_period):
        start_missing = df.isnull().sum().sum()
        period_col = rolling_period[rolling_period == i].index.to_list()
        df[period_col] = df.groupby('gvkey').apply(lambda x: x[period_col]
                                                   .fillna(x[period_col]
                                                           .rolling(i, min_periods=i).mean().shift()))[period_col]
        end_missing = df.isnull().sum().sum()
        print('rolling period {} fillna: {}'.format(i, start_missing - end_missing))

    df[rest_col] = df[rest_col].fillna(0)
    print('end missing: {}'.format(df.isnull().sum().sum()))

    return df

if __name__ == "__main__":

    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    # 1: convert
    try:
        main = pd.read_csv('main_convert.csv')
        print('local version running')
    except:
        main = convert()

    # 2: pre_fillna -> decide rolling periods
    # try:
    #     rolling_period = pd.read_csv('rolling_period.csv',index_col='Unnamed: 0')['0']
    #     print('local version running - rolling_period')
    # except:
    rolling_period = optimal_rolling_period(main)

    # # 3: fillna
    # main_final = fillna(main, rolling_period)
    # main.to_csv('main.csv', index=False)

    # main.to_sql('main_forward', engine)
