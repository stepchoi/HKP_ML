import pandas as pd
from sqlalchemy import create_engine

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)
main = pd.read_sql('SELECT * FROM main', engine)
dep = pd.read_sql('SELECT * FROM niq_main', engine)
macro = pd.read_sql("SELECT * FROM macro_main", engine)
stock  = pd.read_sql("SELECT * FROM stock_main", engine)
main.to_csv('main.csv',index=False)
dep.to_csv('niq_main.csv',index=False)
macro.to_csv('macro_main.csv',index=False)
stock.to_csv('stock_main.csv',index=False)