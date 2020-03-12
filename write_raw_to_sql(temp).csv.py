import pandas as pd
from sqlalchemy import types, create_engine

df = pd.read_csv('')

d={}
for k,v in zip(df.dtypes.index,df.dtypes):
    if v=='object':
       d[k]=types.VARCHAR(df[k].str.len().max())
    elif v=='float64':
       d[k]=types.FLOAT(126)
    elif v=='int64':
       d[k] = types.INTEGER()

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)
df.to_sql(name='raw', con=engine, schema='public', if_exists='replace', index=False,dtype=d)
