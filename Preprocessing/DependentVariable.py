import os

import numpy as np
import pandas as pd
from LoadData import convert_to_float32
from PrepareDatabase import drop_nonseq
from scipy import stats
from sqlalchemy import create_engine


def load_data_y(y='niq'):
    # import engine, select variables, import raw database
    try:
        os.chdir('/Users/Clair/PycharmProjects/HKP_ML_DL/Preprocessing/raw')
        dep = pd.read_csv('raw.csv', usecols=['gvkey', 'datacqtr', y])
        print('local version')
    except:
        db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
        engine = create_engine(db_string)
        dep = pd.read_sql('SELECT gvkey, datacqtr, ' + y + ' FROM raw', engine)

    return dep


def qoq_yoy(dep, abs_version):
    dep = drop_nonseq(dep)

    # convert to qoq, yoy
    dep['next1_abs'] = dep.groupby('gvkey').apply(lambda x: x['niq'].shift(-1)).to_list()

    if abs_version == False:
        dep['qoq'] = dep['next1_abs'].div(dep['niq']).sub(1)  # T1/T0
    else:
        dep['qoq'] = (dep['next1_abs'] - dep['niq']).div(np.abs(dep['niq']))  # T1/T0 - absolute version

    dep['past4_abs'] = dep.groupby('gvkey').apply(
        lambda x: x['niq'].rolling(4, min_periods=4).sum()).to_list()  # rolling past 4 quarter
    dep['next4_abs'] = dep.groupby('gvkey').apply(
        lambda x: x['past4_abs'].shift(-4)).to_list()  # rolling next 4 quarter

    if abs_version == False:
        dep['yoy'] = dep['next4_abs'].div(dep['past4_abs']).sub(1)  # T4/T0
    else:
        dep['yoy'] = (dep['next4_abs'] - dep['past4_abs']).div(np.abs(dep['past4_abs']))  # T4/T0 - absolute version

    dep = dep.filter(['gvkey', 'datacqtr', 'qoq', 'yoy'])
    dep = dep.replace([np.inf, -np.inf], np.nan)
    return dep


def Timestamp(df):
    df['datacqtr'] = df['datacqtr'].apply(
        lambda x: pd.Period(x, freq='Q-DEC').to_timestamp(how='end').strftime('%Y-%m-%d'))
    return df


def main(abs_version=False):
    dep = load_data_y()
    dep = qoq_yoy(dep, abs_version)
    dep = Timestamp(dep)
    convert_to_float32(dep)
    return dep


def remove_outliers(y_type, dep, by):  # remove outlier from both 'yoy' & 'qoq' y_type
    y_series = dep[y_type].dropna()

    if by == 'stv': # remove outlier by standard deviation
        y_clean = y_series.where(np.abs(stats.zscore(y_series)) < 5).dropna()
        idx = y_clean.index

    elif by == 'quantile': # remove outlier by top/bottom percentage
        Q1 = y_series.quantile(0.01)
        Q3 = y_series.quantile(0.99)
        y_clean = y_series.mask((y_series < Q1) | (y_series > Q3)).dropna()
        idx = y_clean.index
    else:
        print("Error: 'by' can only be 'stv' or 'quantile'.")
    return dep.loc[idx]

if __name__ == '__main__':

    # dep = main(abs_version=False)
    # dep.to_csv('niq.csv', index=False)
    # print(dep.describe())
    #
    # dep1 = main(abs_version=True)
    # dep1.to_csv('niq_abs.csv', index=True)
    # print(dep1.describe())
    # exit(0)
    os.chdir('/Users/Clair/PycharmProjects/HKP_ML_DL/Preprocessing/raw')

    # dep = pd.read_csv('niq.csv')
    dep = pd.read_csv('niq_abs.csv')
    print(len(dep))
    # dep = remove_outliers(y_type='qoq', dep=dep, by='quantile')    # remove outlier for qoq
    # print(len(dep))
    # print(dep.describe())


    dep = remove_outliers(y_type='yoy', dep=dep, by='quantile')  # remove outlier for yoy
    print(len(dep))
    print(dep.describe())






