import os

import numpy as np
import pandas as pd
from LoadData import convert_to_float32
from sqlalchemy import create_engine


def load_data_y():
    # import engine, select variables, import raw database
    try:
        os.chdir('/Users/Clair/PycharmProjects/HKP_ML_DL/Preprocessing/raw')
        dep = pd.read_csv('raw_main.csv', usecols=['gvkey', 'datacqtr', 'niq', 'atq'])
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

    # fig = plt.figure(figsize=(20, 16), dpi=120)
    print(df['atq'].isnull().sum())
    df['atq'] = df['atq'].mask(df['atq'] == 0, np.nan)
    # check_print([df.head(1000)])

    print(df.loc[df['atq']==0])

    print(df['atq'].isnull().sum())
    # plt.hist(np.log(df['atq']), bins=100)
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax2.hist(df['niq'], bins=100)

    # convert to qoq, yoy
    df['next1'] = df.groupby('gvkey').apply(lambda x: x['niq'].shift(-1)).to_list()

    df['qoq'] = df['next1'].sub(df['niq']).div(df['atq'])  # T1/T0

    df['next4'] = df.groupby('gvkey').apply(lambda x: x['niq'].shift(-4)).to_list()

    df['yoy'] = df['next4'].sub(df['niq']).div(df['atq'])  # T4/T0


    df['past4_sum'] = df.groupby('gvkey').apply(lambda x: x['niq'].rolling(4, min_periods=4).sum()).to_list()  # rolling past 4 quarter
    df['next4_sum'] = df.groupby('gvkey').apply(lambda x: x['past4_sum'].shift(-4)).to_list()  # rolling next 4 quarter

    df['yoyr'] = df['next4_sum'].sub(df['past4_sum']).div(df['atq'])   # T4/T0

    print(df.isnull().sum())
    print(df.shape)

    df = df.dropna(subset=['atq','niq'], how='any')

    print(df.isnull().sum())
    print(df.shape)

    df = df.filter(['gvkey', 'datacqtr', 'atq', 'niq', 'qoq', 'yoy', 'yoyr'])

    # print(df.head(10000).describe())
    # check_print([df.head(10000)])

    from matplotlib import pyplot as plt
    for i in ['qoq', 'yoy', 'yoyr']:
        test = df[i]
        tail = 0.99
        pmax = test.quantile(tail)
        pmin = test.quantile(1-tail)
        print(pmax, pmin)
        test = test.mask(test > pmax, pmax)
        test = test.mask(test < pmin, pmin)
        plt.hist(test, bins=100)
        plt.savefig('y_distribution_{}{}.png'.format(i, tail))

    print('before trim:', df.describe())
    exit(0)


    bins = pd.DataFrame()
    for i in ['yoy','qoq','yoyr']:
        cut_df, cut_bins = pd.qcut(df[i], q=9, labels=range(9), retbins=True)
        bins[i] = list(cut_bins)

    bins.round(4).to_csv('y_qcut9.csv',index=False)


    # check_print([df.describe()])

    # print(pmax)
    # num_list = ['qoq','yoy','yoy_rolling']
    # df[num_list] = df[num_list].mask(df[num_list] > pmax[num_list], pmax[num_list], axis=1)

    return df


def Timestamp(df):
    df['datacqtr'] = df['datacqtr'].apply(
        lambda x: pd.Period(x, freq='Q-DEC').to_timestamp(how='end').strftime('%Y-%m-%d'))
    return df


def main(neg_to_zero=False):
    dep = load_data_y()

    # use all positive value to decide the maximum
    # pre_dep = pd.concat([dep[['gvkey','datacqtr']], dep['niq'].mask(dep['niq'] <= 0, np.nan)],axis=1)
    pre_dep = qoq_yoy(dep, trim=False)

    # pmax_95 = pre_dep.quantile(0.95, axis=0)
    # print(pmax_95)
    #
    # # clean negative value -> 0 (use 0.000001 to facilitate calculation)
    # dep['niq'] = dep['niq'].mask(dep['niq']<0,float(1e-6))
    # dep = qoq_yoy(dep, trim=True, pmax=pmax_95)
    # print(dep.describe())

    # convert datacqtr to timestamp
    # pre_dep = Timestamp(pre_dep)
    convert_to_float32(pre_dep)
    print(pre_dep.info())
    pre_dep.to_csv('niq_main.csv', index=False)
    return pre_dep

if __name__ == '__main__':
    main()
    # print(1/15)





