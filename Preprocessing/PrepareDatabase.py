'''This code would import TABLE raw and perform:
    1. remove company with non-sequencial record (i.e. Q2 and then jump to Q4)
    2. convert ytd variable from statement of cash flow to quarterly data.
    3. save result to TABLE raw_main
'''

import numpy as np
import pandas as pd
from LoadData import convert_to_float32
from Miscellaneous import Timestamp
from sqlalchemy import create_engine


def select_variable():

    '''This def read TABLE format_map and convert into dictionary where map variables to desired formats'''
    # create dictionary for all selected variables by different formats(yoy, qoq, nom, log), features

    try:
        format_map = pd.read_csv('format_map.csv')
        engine = None
    except:
        db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
        engine = create_engine(db_string)
        format_map = pd.read_sql("SELECT * FROM format_map", engine)

    select = {} # dicionary for all different formats
    select['all'] = format_map['name'].to_list() # all numerical variables
    select['label'] = ['gvkey', 'datacqtr']  # 3 label variables

    for col in format_map.columns[2:-1]:
        select[col] = format_map.loc[format_map[col]==1, 'name'].to_list() # different formats

    select.update(format_map.loc[format_map['special'].notnull()].filter(['name','special']).set_index('name').to_dict()) # special treatment (for dividend)

    return select, engine

def convert_ytd(df, ytd_col):

    '''This def convert ytd variable from statement of cash flow to quarterly data.'''

    df_ytd = df[ytd_col].sub(df[ytd_col].shift(1)) # General conversion: Q(t) = Q(t) - Q(t-1)
    df_ytd = pd.concat([df[['gvkey','datacqtr','fqtr']], df_ytd], axis= 1) # concat label columns

    gvkey_first = df.groupby('gvkey',as_index=False).head(1).index  # Special case 1: first of different companies -> NaN
    df_ytd.loc[gvkey_first, ytd_col] = np.nan
    df_ytd.loc[df['fqtr']==1,ytd_col] = df.loc[df['fqtr']==1,ytd_col]   # Special case 2: all Q1 -> remain the same Q1 (i.e. not deduct last Q4)

    df_ytd = df_ytd.drop(df_ytd[['gvkey','datacqtr','fqtr']], axis = 1)  # remove label columns for return
    df_ytd.columns = [x + '_q' for x in df_ytd.columns] # rename converted ytd columns from '*' to '*_q'

    return df_ytd

def drop_nonseq(df):

    '''This def remove company with non-sequencial record (i.e. Q2 and then jump to Q4)'''

    # drop samples with non-sequential order during sampling period
    # add datacqtr_no to main dataset
    import pandas as pd

    cqtr = pd.DataFrame(set(df['datacqtr']), columns=['datacqtr']).sort_values(by=['datacqtr']).reset_index(drop = True)\
        .reset_index(drop = False).set_index('datacqtr')
    df['datacqtr_no'] = df.datacqtr.map(cqtr['index']) # label 'datacqtr' with integer [0, 1, ...]
    df = df.sort_values(by=['gvkey', 'datacqtr'])

    df['index_seq'] = df.groupby('gvkey').apply(lambda x: (x['datacqtr_no'].shift(-1) - x['datacqtr_no'])).to_list()
    df['index_seq'] = df['index_seq'].fillna(1).astype(int)
    del_gvkey = set(df.loc[df['index_seq']!=1,'gvkey'])
    df = df.loc[~df['gvkey'].isin(del_gvkey)]
    del df['index_seq']

    return df

if __name__ == '__main__':

    # import engine, select variables, import raw database
    select, engine = select_variable()
    ytd_col = [x[:-2] for x in select['all'] if ('_' in x)] # identify cash flow columns (ytd)
    other_col = [x for x in select['all'] if not ('_' in x)]
    all_col = ytd_col + other_col + select['label'] + ['fqtr']

    try:
        df = pd.read_csv('raw.csv', usecols = all_col)
        print('local version running')
    except:
        raw = pd.read_sql('SELECT * FROM raw', engine, columns = all_col)

    convert_to_float32(df)

    df = drop_nonseq(df) # drop non-sequential records

    df_ytd = convert_ytd(df, ytd_col)
    df = pd.concat([df[select['label'] + other_col], df_ytd], axis = 1)

    df = Timestamp(df)
    print(df)
    convert_to_float32(df)

    df.to_csv('raw_main.csv')