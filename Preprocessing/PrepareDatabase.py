def whole_print(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)

def convert_ytd(df, ytd_col):

    df_ytd = df[ytd_col].sub(df[ytd_col].shift(1))
    df_ytd = pd.concat([df[['gvkey','datacqtr','fqtr']], df_ytd], axis= 1)

    gvkey_first = df.groupby('gvkey',as_index=False).head(1).index
    df_ytd.loc[gvkey_first,ytd_col] = np.nan
    df_ytd.loc[df['fqtr']==1,ytd_col] = df.loc[df['fqtr']==1,ytd_col]

    df_ytd = df_ytd.drop(df_ytd[['gvkey','datacqtr','fqtr']], axis = 1)
    df_ytd.columns = [x + '_q' for x in df_ytd.columns]

    return df_ytd

def drop_nonseq(df):

    # drop samples with non-sequential order during sampling period
    # add datacqtr_no to main dataset
    import pandas as pd

    cqtr = pd.DataFrame(set(df['datacqtr']), columns=['datacqtr']).sort_values(by=['datacqtr']).reset_index(drop = True).reset_index(drop = False).set_index('datacqtr')
    df['datacqtr_no'] = df.datacqtr.map(cqtr['index'])
    df = df.sort_values(by=['gvkey', 'datacqtr'])

    df['index_seq'] = df.groupby('gvkey').apply(lambda x: (x['datacqtr_no'].shift(-1) - x['datacqtr_no'])).to_list()
    df['index_seq'] = df['index_seq'].fillna(1).astype(int)
    del_gvkey = set(df.loc[df['index_seq']!=1,'gvkey'])
    df = df.loc[~df['gvkey'].isin(del_gvkey)]
    del df['index_seq']

    return df

if __name__ == '__main__':

    from sqlalchemy import create_engine
    import pandas as pd
    import numpy as np
    from Preprocessing import select_variable

    # import engine, select variables, import raw database
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    select = select_variable(engine)
    ytd_col = [x[:-2] for x in select['all'] if ('_' in x)]
    other_col = [x for x in select['all'] if not ('_' in x)]
    all_col = ytd_col + other_col + select['label'] + ['fqtr']

    df = pd.read_csv('raw.csv', usecols = all_col)
    df = drop_nonseq(df)

    df_ytd = convert_ytd(df, ytd_col)
    df = pd.concat([df[select['label'] + other_col], df_ytd], axis = 1)

    print(df)
    df.to_csv('raw_main.csv')