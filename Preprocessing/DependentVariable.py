import numpy as np
import pandas as pd
from PrepareDatabase import drop_nonseq
from sqlalchemy import create_engine

if __name__ == '__main__':

    # import engine, select variables, import raw database
    try:
        dep = pd.read_csv('raw.csv', usecols = ['gvkey', 'datacqtr', 'niq'])
        engine = None
    except:
        db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
        engine = create_engine(db_string)
        dep = pd.read_sql('SELECT niq FROM raw', engine)

    dep = drop_nonseq(dep)

    dep['next1_abs'] = dep.groupby('gvkey').apply(lambda x: x['niq'].shift(1)).to_list()
    dep['qoq'] = dep['next1_abs'].div(dep['niq']).sub(1) # T1/T0
    dep['past4_abs'] = dep.groupby('gvkey').apply(lambda x: x['niq'].rolling(4, min_periods=4).sum()).to_list() # rolling past 4 quarter
    dep['next4_abs'] = dep.groupby('gvkey').apply(lambda x: x['past4_abs'].shift(-4)).to_list() # rolling next 4 quarter
    dep['yoy'] = dep['next4_abs'].div(dep['past4_abs']).sub(1) # T4/T0
    dep = dep.filter(['gvkey', 'datacqtr','qoq','yoy'])
    dep = dep.replace([np.inf, -np.inf], np.nan)

    print(dep)
    dep.to_csv('niq.csv', index=False)
