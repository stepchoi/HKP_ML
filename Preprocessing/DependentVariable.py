import numpy as np
import pandas as pd
from LoadData import convert_to_float32
from PrepareDatabase import drop_nonseq
from sqlalchemy import create_engine


def load_data_y(y = 'niq'):
    # import engine, select variables, import raw database
    try:
        import os
        os.chdir('/Users/Clair/PycharmProjects/HKP_ML_DL/Preprocessing/raw')
        dep = pd.read_csv('raw.csv', usecols = ['gvkey', 'datacqtr', y])
        print('local version')
    except:
        db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
        engine = create_engine(db_string)
        dep = pd.read_sql('SELECT gvkey, datacqtr, ' + y + ' FROM raw', engine)

    return dep

def qoq_yoy(dep):
    dep = drop_nonseq(dep)

    # convert to qoq, yoy
    dep['next1_abs'] = dep.groupby('gvkey').apply(lambda x: x['niq'].shift(-1)).to_list()
    dep['qoq'] = dep['next1_abs'].div(dep['niq']).sub(1)  # T1/T0
    dep['past4_abs'] = dep.groupby('gvkey').apply(lambda x: x['niq'].rolling(4, min_periods=4).sum()).to_list()  # rolling past 4 quarter
    dep['next4_abs'] = dep.groupby('gvkey').apply(lambda x: x['past4_abs'].shift(-4)).to_list()  # rolling next 4 quarter
    dep['yoy'] = dep['next4_abs'].div(dep['past4_abs']).sub(1)  # T4/T0
    dep.to_csv('dep_full.csv', index=False)
    exit(0)
    dep = dep.filter(['gvkey', 'datacqtr', 'qoq', 'yoy'])
    dep = dep.replace([np.inf, -np.inf], np.nan)
    return dep

def Timestamp(df):
    df['datacqtr'] = df['datacqtr'].apply(lambda x: pd.Period(x, freq='Q-DEC').to_timestamp(how='end').strftime('%Y-%m-%d'))
    return df

def main():
    dep = load_data_y()
    dep = qoq_yoy(dep)
    dep = Timestamp(dep)
    convert_to_float32(dep)
    dep.to_csv('niq.csv', index=False)
    print(dep.info())

if __name__ == '__main__':
    main()