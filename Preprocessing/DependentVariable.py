import numpy as np
import pandas as pd
from PrepareDatabase import drop_nonseq
from sqlalchemy import create_engine

def load_data_y(y = 'niq'):
    # import engine, select variables, import raw database
    try:
        dep = pd.read_csv('raw.csv', usecols = ['gvkey', 'datacqtr', y])
        # engine = None
    except:
        db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
        engine = create_engine(db_string)
        dep = pd.read_sql('SELECT ' + y + ' FROM raw', engine)

    return dep

def qoq_yoy(dep):
    dep = drop_nonseq(dep)

    # convert to qoq, yoy
    dep['next1_abs'] = dep.groupby('gvkey').apply(lambda x: x['niq'].shift(1)).to_list()
    dep['qoq'] = dep['next1_abs'].div(dep['niq']).sub(1)  # T1/T0
    dep['past4_abs'] = dep.groupby('gvkey').apply(lambda x: x['niq'].rolling(4, min_periods=4).sum()).to_list()  # rolling past 4 quarter
    dep['next4_abs'] = dep.groupby('gvkey').apply(lambda x: x['past4_abs'].shift(-4)).to_list()  # rolling next 4 quarter
    dep['yoy'] = dep['next4_abs'].div(dep['past4_abs']).sub(1)  # T4/T0
    dep = dep.filter(['gvkey', 'datacqtr', 'qoq', 'yoy'])
    dep = dep.replace([np.inf, -np.inf], np.nan)
    return dep

def Timestamp(df):
    df['datacqtr'] = df['datacqtr'].apply(pd.Period(text, freq='Q-DEC').to_timestamp(how='end').strftime('%Y-%m-%d'))
    return df

if __name__ == '__main__':
    dep = load_data_y()
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)
    dep = qoq_yoy(dep)
    dep = Timestamp(dep)
    dep.to_csv('niq.csv', index=False)


