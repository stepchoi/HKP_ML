import os

import numpy as np
import pandas as pd
from LoadData import convert_to_float32
from PrepareDatabase import drop_nonseq
from sqlalchemy import create_engine


def load_data_y():
    # import engine, select variables, import raw database
    try:
        os.chdir('/Users/Clair/PycharmProjects/HKP_ML_DL/Preprocessing/raw')
        dep = pd.read_csv('raw.csv', usecols=['gvkey', 'datacqtr', 'niq', 'atq'])
        print('local version')
    except:
        db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
        engine = create_engine(db_string)
        dep = pd.read_sql('SELECT gvkey, datacqtr, niq, atq FROM raw', engine)

    return dep

def trim_outlier(df, prc=0.10):
    pmax = df.quantile(q=(1-prc), axis=0)
    print(pmax)
    df = df.mask(df>pmax,pmax, axis=1)
    return df

def qoq_yoy(df, trim=False, pmax=None):

    df = drop_nonseq(df)

    # convert to qoq, yoy
    df['next1_abs'] = df.groupby('gvkey').apply(lambda x: x['niq'].shift(-1)).to_list()

    df['qoq'] = df['next1_abs'].div(df['niq']).sub(1)  # T1/T0

    df['next4'] = df.groupby('gvkey').apply(lambda x: x['niq'].shift(-4)).to_list()

    df['yoy'] = df['next4'].div(df['niq']).sub(1)  # T4/T0


    df['past4_abs'] = df.groupby('gvkey').apply(lambda x: x['niq'].rolling(4, min_periods=4).sum()).to_list()  # rolling past 4 quarter
    df['next4_abs'] = df.groupby('gvkey').apply(lambda x: x['past4_abs'].shift(-4)).to_list()  # rolling next 4 quarter

    df['yoy_rolling'] = df['next4_abs'].div(df['past4_abs']).sub(1)  # T4/T0

    df = df.filter(['gvkey', 'datacqtr', 'qoq', 'yoy', 'yoy_rolling'])

    # print('before trim:', df.describe())

    if trim == True:
        print(pmax)
        num_list = ['qoq','yoy','yoy_rolling']
        df[num_list] = df[num_list].mask(df[num_list] > pmax[num_list], pmax[num_list], axis=1)

    # print('after trim:', df.describe())

    return df


def Timestamp(df):
    df['datacqtr'] = df['datacqtr'].apply(
        lambda x: pd.Period(x, freq='Q-DEC').to_timestamp(how='end').strftime('%Y-%m-%d'))
    return df


def main(neg_to_zero=False):
    dep = load_data_y()

    # use all positive value to decide the maximum
    pre_dep = pd.concat([dep[['gvkey','datacqtr']], dep['niq'].mask(dep['niq'] <= 0, np.nan)],axis=1)
    pre_dep = qoq_yoy(pre_dep, trim=False)
    pmax_95 = pre_dep.quantile(0.95, axis=0)
    print(pmax_95)

    # clean negative value -> 0 (use 0.000001 to facilitate calculation)
    dep['niq'] = dep['niq'].mask(dep['niq']<0,float(1e-6))
    dep = qoq_yoy(dep, trim=True, pmax=pmax_95)
    print(dep.describe())

    # convert datacqtr to timestamp
    dep = Timestamp(dep)
    convert_to_float32(dep)
    dep.to_csv('niq_main.csv', index=False)
    return dep

if __name__ == '__main__':
    main()
    # print(1/15)





