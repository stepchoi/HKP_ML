'''Update:
    1. using Raw SQL connect to Postgres

'''

# 1. connect to postgresql & import packages
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

db_string = "postgres://postgres:DLvalue123@hkpolyu-dl-value.c1lrltigx0e7.us-east-1.rds.amazonaws.com:5432/postgres"
engine = create_engine(db_string)

# 2. import main_format, format_map datasets
ie = pd.read_sql("SELECT * FROM format_map", engine)
main = pd.read_sql("select * from main_format", engine)
main_dep = pd.read_sql(
    "SELECT gvkeydatafqtr, epspxq_nextq_abs, selected FROM main_dependent", engine)

grouped = ie.groupby(['format'])
type_dict = {}
for name, g in grouped:
    type_dict[name] = g['abbreviation'].dropna().to_list()

num_col = list(set(ie['abbreviation'].dropna()))  # all feature columns
label_col = ['gvkey', 'datacqtr', 'datafqtr', 'gsector', 'cquarter',
             'cyear', 'cyeargvkey', 'gvkeydatafqtr']

# 2. merge dependent + indepenedent variables
main_xy = pd.merge(main,main_dep, on =['gvkeydatafqtr'],how='left')
main_xy = main_xy.dropna(subset=['epspxq_nextq_abs','atq'],axis = 0)
main_xy.iloc[:,8:] = main_xy.iloc[:,8:].astype(float)

# 3. fillna
## 3.1 YoY, QoQ -> 0
for i in ['YoY', 'QoQ']:
    main_xy[type_dict[i]] = main_xy[type_dict[i]].fillna(0)

## 3.2 Log, Abs -> forward fill
for i in ['Abs', 'Log']:
    main_xy[type_dict[i]] = main_xy.groupby('gvkey').apply(lambda x: x.fillna(method='ffill'))[type_dict[i]]

## 3.3 dedicated methods
del_col = set(['epsfxq', 'epspiq', 'lseq', 'ltmibq'] + ['oepsxq', 'ivstq'])
main_xy = main_xy.filter(set(main_xy.columns) - del_col)

main_xy['chq'] = main_xy['chq'].fillna(main_xy['cheq'])
main_xy['chq'] = main_xy['chq'].fillna(main_xy['cheq'])
main_xy[['aocipenq', 'ivltq', 'dvpq']] = main_xy[['aocipenq', 'ivltq', 'dvpq']].fillna(-6.907755)

## 3.4 rest dropna
main_xy.dropna(subset=k_lst[1:])

#4.


