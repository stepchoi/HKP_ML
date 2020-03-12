import pandas as pd
from sqlalchemy import types, create_engine

df = pd.read_csv('/Users/Clair/PycharmProjects/HKP_ML_DL/Preprocessing/raw/raw.csv', low_memory=False).head(1)

d={}
for k,v in zip(df.dtypes.index,df.dtypes):
    d[k]= types.String()
print(d)

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)
df.to_sql(name='raw', con=engine, schema='public', if_exists='replace', index=False,dtype=d)
