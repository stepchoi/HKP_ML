'''This code would import TABLE raw_main and perform:
    1. convert variables to selected formats: (333325, 204)
    2. convert columns ‘datacqtr’ to timestamp (yyyy-mm-dd)
    3. delete high missing columns if there rests other formats
    4. fillna
    5. run correlation and deleted high correlation pairs(>0.9)
    6. save to TABLE main
'''

import os

import numpy as np
import pandas as pd
from LoadData import convert_to_float32
from PrepareDatabase import select_variable
from sqlalchemy import create_engine
from tqdm import tqdm


def check_print(df_list):
    df = pd.concat(df_list, axis=1)
    col = ['gvkey','datacqtr'] + [x for x in sorted(df.columns) if x not in ['gvkey','datacqtr']]
    df = df.reindex(col, axis=1)
    df.to_csv('check.csv')

    os.system("open -a '/Applications/Microsoft Excel.app' 'check.csv'")
    exit(0)

# 1: convert
def convert(abs_version):

    # import engine, select variables, import raw database
    try:
        os.chdir('/Users/Clair/PycharmProjects/HKP_ML_DL/Preprocessing/raw')
        raw = pd.read_csv('raw_main.csv')
        engine = None
        print('local version running - raw_main.csv')
    except:
        db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
        engine = create_engine(db_string)
        raw = pd.read_sql('SELECT * FROM raw_main', engine)

    convert_to_float32(raw)

    def convert_format(df, dic, abs_version):

        # convert raw dataset to desired formats (YoY, QoQ, Log)
        # groupby 'gvkey'

        convert_select = {}

        for k in ['yoy', 'qoq', 'atq', 'revtq']:
            convert_select[k] = [x + '_' + k for x in dic[k]]

        label_nom = df.filter(dic['label'] + dic['nom'])
        print('------ start conversion -------')

        # special: convert dividend to rolling 4 period sum
        df['dvy_q'] = df.groupby('gvkey').apply(lambda x: x['dvy_q'].rolling(4, min_periods=1).sum()).reset_index(
            drop=True)
        print('finish dividends rolling accural conversion')

        # qoq conversion
        if abs_version == False:
            qoq = df[dic['qoq']].div(df[dic['qoq']].shift(1)).sub(1).reset_index(drop=True)
            print('abs_version is', abs_version)
        else:
            qoq = (df[dic['qoq']]-df[dic['qoq']].shift(1)).div(np.abs(df[dic['qoq']].shift(1))).reset_index(drop=True) # absolute version
        qoq.iloc[df.groupby('gvkey').head(1).index] = np.nan
        qoq.columns = convert_select['qoq']

        print('finish qoq conversion')

        # yoy conversion
        if abs_version == False:
            yoy = df[dic['yoy']].div(df[dic['yoy']].shift(4)).sub(1).reset_index(drop=True)
        else:
            yoy = (df[dic['yoy']]-df[dic['yoy']].shift(4)).div(np.abs(df[dic['yoy']].shift(4))).reset_index(drop=True) # absolute version

        yoy.iloc[df.groupby('gvkey').head(4).index] = np.nan
        yoy.columns = convert_select['yoy']
        print('finish yoy conversion')

        # ln(*/atq + 1) conversion
        atq = np.log(df[dic['atq']].apply(lambda x: x.div(df['atq']).add(1).replace([np.inf, -np.inf], np.nan)))
        atq.columns = convert_select['atq']
        print('finish atq conversion')

        # ln(*/revtq + 1) conversion
        revtq = np.log(df[dic['revtq']].apply(lambda x: x.div(df['revtq']).add(1).replace([np.inf, -np.inf], np.nan)))
        revtq.columns = convert_select['revtq']
        print('finish revtq conversion')

        dic.update(convert_select)

        df_converted = pd.concat([label_nom, qoq, yoy, atq, revtq], axis=1) # concat all formats
        df_converted = df_converted.replace([np.inf, -np.inf], np.nan)

        def missing_count(df):
            df = pd.DataFrame(df.isnull().sum(), columns=['missing']).reset_index(drop=False)
            sp = pd.DataFrame([x.rsplit('_', 1) for x in df['index']])
            df[['name', 'format']] = sp
            df.to_csv('df_missing.csv')
            print('export df_missing.csv')

        # missing_count(df_converted)
        # convert_to_float32(df_converted)
        # df_converted.to_csv('main_convert.csv', index=False)

        print('shape of df_converted is ', df_converted.shape)

        print(df_converted['niq_qoq'].describe())
        exit(0)
        return df_converted, dic

    select, engine = select_variable()  # dictionary with variables based on selection criteria
    main, select = convert_format(raw, select, abs_version)
    print('coverted missing:', main.isnull().sum().sum())
    return main

# 2. delete high_missing if exists other formats
def delete_high_missing(df, threshold):

    df = pd.DataFrame(df.isnull().sum().sort_values(), columns=['#_missing']).reset_index(drop = False)
    df[['name','format']] = pd.DataFrame([x.rsplit('_',1) for x in df['index']])
    df['%_missing'] = df['#_missing']/333325
    df = df.filter(['index', 'name','format','#_missing','%_missing'])

    del_col = []
    for name, g in df.groupby('name'):
        if 0 < len(g.loc[g['%_missing']>threshold]) < len(g) :  # if not all columns > missing rate threshold
            del_col.extend(g.loc[g['%_missing']>threshold, 'index'].to_list())
        elif len(g.loc[g['%_missing']>threshold]) == len(g):    # if all columns > missing rate threshold
            del_col.extend(g.sort_values(by = ['%_missing'])['index'].to_list()[1:])    # only append higher 2
    print(len(del_col))
    df.loc[df['name'].isin(del_col)].to_csv('delete_high_missing.csv', index=False) # absolute version
    return del_col

# 3: fillna
def fillna(df):

    ''' 1: fill YoY, QoQ -> -1'''
    fillna_count = {}
    print('------ start fillna -------')
    col = df.columns

    important = ['atq', 'ltq', 'seqq', 'cheq', 'revtq', 'niq']
    del_row = [x for x in df.columns if (x.rsplit('_',1)[0] in important)]
    yoy_qoq_col = [x for x in col if ((('yoy' in x) or ('qoq' in x)) & (x not in del_row))]

    del_row_na1 = df[del_row].isnull().sum()
    s0 = df.isnull().sum().sum() # counting
    fillna_count['original'] = s0

    df[yoy_qoq_col] = df[yoy_qoq_col].fillna(value = -1) # yoy, qoq format columns NAN -> -1

    s1 = df.isnull().sum().sum() # counting
    fillna_count['yoy_qoq'] = s0 - s1

    ''' 2: fill 0 after 8th '''

    def after_8(series):
        index_nan = [item for sublist in np.argwhere(np.isnan(series)) for item in sublist]

        if not index_nan == []:

            index_nan_df = pd.DataFrame(index_nan)
            index_nan_df['sub'] = index_nan_df[0].sub(index_nan_df[0].shift(1))
            begin_nan = index_nan_df.loc[index_nan_df['sub'] != 1, 0].values # calculate the index increase between NAN records

            begin_nan_8 = []
            for i in range(8):
                begin_nan_8.extend(begin_nan + i)

            fill0_nan = list(set(index_nan) - set(begin_nan_8))
            series[fill0_nan] = -1 # atq, revtq, nom format columns: for consecutiv 0 after 8th period -> -1

        return series

    rest_col = list(set(col) - set(del_row) - set(['gvkey','datacqtr']) - set(yoy_qoq_col))

    df[rest_col] = df[rest_col].apply(after_8)
    s2 = df.isnull().sum().sum() # counting
    fillna_count['after 8'] = s1 - s2
    print(fillna_count)

    ''' 3: decide optimal rolling average fill NaN period '''

    def optimal_rolling_period(df):

        # decide optimal rolling period by finding least SSR -> minimum period = 1

        diff_dict = {}
        for p in tqdm([1, 4, 8, 12, 16, 20]):
            if p == 1:  # test for forward fill
                df_rolling = df.groupby('gvkey').apply(lambda x: x[rest_col].shift(1))
            else:   # test for rolling average
                df_rolling = df.groupby('gvkey').apply(lambda x: x[rest_col].rolling(p, min_periods=1).mean().shift())
            diff_dict[p] = df_rolling.sub(df[rest_col]).pow(2).mean(axis=0) # square sum of error with above methods

        rolling_period_series = pd.DataFrame.from_dict(diff_dict, orient='index').idxmin() # find minimum error option

        pd.DataFrame.from_dict(diff_dict, orient='index').to_csv('optimal_rolling_period_full.csv')
        pd.DataFrame(rolling_period_series).to_csv('rolling_period.csv')

        return rolling_period_series

    try:
        rolling_period = pd.read_csv('rolling_period.csv',index_col='Unnamed: 0')['0']
        print('local version running - rolling_period')
    except:
        rolling_period = optimal_rolling_period(df)

    ''' 4: rolling average for rest except delete_row '''

    print(' ----- start fillna rolling -----')
    for i in tqdm(set(rolling_period)):
        if not np.isnan(i):
            i = int(i)
            start_missing = df.isnull().sum().sum() # for counting and print

            period_col = [x for x in rolling_period[rolling_period == i].index.to_list() if x in rest_col]

            if i == 1:
                df[period_col] = df.groupby('gvkey').apply(lambda x: x[period_col].ffill())
            else:
                df[period_col] = df.groupby('gvkey').apply(lambda x: x[period_col]
                                                           .fillna(x[period_col].rolling(i, min_periods=1).mean().shift()))[period_col]

            end_missing = df.isnull().sum().sum() # for counting and print
            fillna_count['rolling period {}'.format(i)] = start_missing - end_missing
            print('rolling period {} fillna: {}'.format(i, start_missing - end_missing))

    s3 = df.isnull().sum().sum() # counting

    df[rest_col] = df[rest_col].fillna(value = -1)

    s4 = df.isnull().sum().sum() # counting
    del_row_na2 = df[del_row].isnull().sum()
    pd.concat([del_row_na1,del_row_na2],axis=1).to_csv('important_col_nan.csv')
    fillna_count['fill in rest missing'] = s3 - s4
    fillna_count['ending'] = s4
    pd.DataFrame.from_dict(fillna_count, orient='index').to_csv('fillna_count.csv')

    return df, del_row

# 4. delete high correlation items
def check_correlation(df, del_row, threshold=0.9):
    # find high correlated items -> excel

    corr = df.corr().abs()  # create correlation matrix
    so = corr.unstack().reset_index()   # unstack matrix
    so.columns = ['v1', 'v2', 'corr']
    so = so.loc[(so['v1'] != so['v2']) & (so['corr'] > threshold)].drop_duplicates(subset=['v1', 'v2'])  # extract highly correlated pairs

    high_corr_col = set() # create set for to_be_deleted high correlation items
    for i in range(len(corr.columns)):
        for j in range(i):
            if (corr.iloc[i, j] >= threshold) and (corr.columns[j] not in high_corr_col):
                colname = corr.columns[i]  # getting the name of column
                high_corr_col.add(colname)

    high_corr_col = high_corr_col - set(del_row)

    so.loc[so['v1'].isin(high_corr_col),'exist_v1'] = 1
    so.loc[so['v2'].isin(high_corr_col),'exist_v2'] = 1
    so.to_csv('high_corr.csv',index=False)
    return high_corr_col

def main(abs_version=False):

    # 1: convert
    try:
        main = pd.read_csv('main_convert.csv')
        print('local version running - main_convert')
    except:
        main = convert(abs_version)
    col_dict = pd.DataFrame(index = main.columns)
    convert_to_float32(main)

    # 2. delete high_missing if exists other formats
    del_col = delete_high_missing(main, 0.7)
    main = main.drop(main[del_col], axis = 1)
    print(main.shape)
    col_dict.loc[main.columns, 'low_missing'] = 1

    # 3: fillna
    main, del_row = fillna(main)
    print('del_row (i.e. non allowed fillna): ', del_row)

    # 4. delete high correlation items
    high_corr_col = check_correlation(main.iloc[:,2:], del_row, 0.9)
    # del_corr = ['xsgaq_qoq', 'gdwlq_atq', 'cogsq_qoq']  # same for both forward & rolling version
    main = main.drop(main[high_corr_col], axis=1)
    col_dict.loc[main.columns, 'low_correlation'] = 1
    col_dict.to_csv('columns_after_conversion.csv')

    convert_to_float32(main)
    main.to_csv('main.csv', index=False)
    print(main['niq_qoq'].describe())


if __name__ == "__main__":

    # main(abs_version=True)
    df1 = pd.read_csv('/Users/Clair/PycharmProjects/HKP_ML_DL/Preprocessing/raw/abs_version/main_abs.csv')
    pd.concat([df1.quantile(0.05), df1.quantile(0.95)], axis=1).to_csv('describe-abs5.csv')
    pd.concat([df1.quantile(0.02), df1.quantile(0.98)], axis=1).to_csv('describe-abs2.csv')
    pd.concat([df1.quantile(0.01), df1.quantile(0.99)], axis=1).to_csv('describe-abs1.csv')
    # print(df1.quantile(0.01))

    exit(0)
    #
    df2 = pd.read_csv('/Users/Clair/PycharmProjects/HKP_ML_DL/Preprocessing/raw/nom_version/main.csv')
    # print(df2['niq_qoq'].describe())
    # print(all([False, True]))
    print(all(df1['niq_qoq']==df2['niq_qoq']))
    # print(all(df1==df2))

    # os.chdir('/Users/Clair/PycharmProjects/HKP_ML_DL/Hyperopt_LightGBM')
