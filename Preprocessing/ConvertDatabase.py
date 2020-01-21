from sqlalchemy import create_engine
import pandas as pd
import numpy as np

def whole_print(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
        
def select_variable(engine):

    # create dictionary for all selected variables by different formats(yoy, qoq, nom, log), features
    try:
        format_map = pd.read_csv('format_map.csv')
    except:
        format_map = pd.read_sql("SELECT * FROM format_map", engine)

    select = {}
    select['all'] = format_map['name'].to_list()
    select['label'] = ['gvkey', 'datacqtr', 'sic']

    for col in format_map.columns[2:-1]:
        select[col] = format_map.loc[format_map[col]==1, 'name'].to_list()

    select.update(format_map.loc[format_map['special'].notnull()].filter(['name','special']).set_index('name').to_dict())

    return select


def convert_format(df, dic):

    # convert raw dataset to desired formats (YoY, QoQ, Log)
    # groupby 'gvkey'

    convert_select = {}

    for k in ['yoy','qoq','atq','revtq']:
        convert_select[k] = [x + '_' + k for x in dic[k]]

    label_nom = df.filter(dic['label'] + dic['nom'])
    print('------ start conversion -------')

    # special: convert dividend to rolling 4 period sum
    df['dvy_q'] = df.groupby('gvkey').apply(lambda x: x['dvy_q'].rolling(4, min_periods=1).sum()).reset_index(drop = True)
    print('finish dividends rolling accural conversion')

    qoq = df.groupby('gvkey').apply(lambda x: x[dic['qoq']].div(x[dic['qoq']].shift(1)).sub(1)).reset_index(drop = True)
    qoq.columns = convert_select['qoq']
    print('finish qoq conversion')

    yoy = df.groupby('gvkey').apply(lambda x: x[dic['yoy']].div(x[dic['yoy']].shift(4)).sub(1)).reset_index(drop = True)
    yoy.columns = convert_select['yoy']
    print('finish yoy conversion')

    atq = np.log(df[dic['atq']].apply(lambda x: x.div(df['atq']).add(1).replace([np.inf, -np.inf], np.nan)))
    atq.columns = convert_select['atq']
    print('finish atq conversion')

    revtq = np.log(df[dic['revtq']].apply(lambda x: x.div(df['revtq']).add(1).replace([np.inf, -np.inf], np.nan)))
    revtq.columns = convert_select['revtq']
    print('finish revtq conversion')

    dic.update(convert_select)

    df_1 = pd.concat([label_nom, qoq, yoy, atq, revtq], axis = 1)
    df_1 = df_1.replace([np.inf, -np.inf], np.nan)

    def missing_count(df):
        df = pd.DataFrame(df.isnull().sum(), columns = ['missing']).reset_index(drop = False)
        sp = pd.DataFrame([x.rsplit('_', 1) for x in df['index']])
        df[['name', 'format']] = sp
        df.to_csv('df_missing.csv')
        print('export df_missing.csv')

    # missing_count(df_1)
    # df_1.to_csv('main_convert.csv', index=False)

    print(df_1.shape)
    return df_1, dic

# fillna methods
def after_8(series):  # for fillna
    index_nan = [item for sublist in np.argwhere(np.isnan(series)) for item in sublist]

    index_nan_df = pd.DataFrame(index_nan)
    index_nan_df['sub'] = index_nan_df[0].sub(index_nan_df[0].shift(1))
    begin_nan = index_nan_df.loc[index_nan_df['sub'] != 1, 0].values

    begin_nan_8 = []
    for i in range(8):
        begin_nan_8.extend(begin_nan + i)

    fill0_nan = list(set(index_nan) - set(begin_nan_8))
    series[fill0_nan] = 0

    return series

class fillna:
    def __init__(self, df, dic):

        # 1.1: fill YoY, QoQ -> 0
        print('------ start fillna -------')
        col1 = dic['yoy'] + dic['qoq']
        df[col1] = df[col1].fillna(0)

        # 1.2: fill more than 8 period
        delete_row = [x+'_atq' for x in dic['delete_row']] + [x+'_revtq' for x in dic['delete_row']]
        self.col = list(set(dic['atq'] + dic['revtq']) - set(delete_row))
        df[self.col] = df[self.col].apply(after_8)
        self.df = df

    def forward(self):
        self.df[self.col] = self.df.groupby('gvkey').apply(lambda x: x[self.col].ffill(limit = 8))
        self.df[self.col] = self.df[self.col].fillna(0)
        print('finish fillna forward')

        return self.df

    def rolling(self):
        self.df[self.col] = self.df.groupby('gvkey').apply(lambda x: x[self.col].fillna(x[self.col].rolling(12, min_periods=1).mean()))[self.col]
        self.df[self.col] = self.df[self.col].fillna(0)
        print('finish fillna rolling')
        return self.df

if __name__ == "__main__":

    # import engine, select variables, import raw database
    try:
        raw = pd.read_csv('raw_main.csv')
        engine = None
        print('local version running')
    except:
        db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
        engine = create_engine(db_string)
        raw = pd.read_sql('SELECT * FROM raw_main', engine)

    select = select_variable(engine) # dictionary with variables based on selection criteria
    all_col = select['label'] + select['all']  # all means all numeric variables selected
    # whole_print(raw.isnull().sum().sort_values())


    main, select = convert_format(raw, select)

    # rolling = fillna(main, select).rolling()
    # whole_print(rolling.isnull().sum().sort_values())
    # print('rolling left missing: ' + str(rolling.isnull().sum().sum()))
    # main.to_csv('main_rolling.csv', index=False)

    forward = fillna(main, select).forward()
    whole_print(forward.isnull().sum().sort_values())
    print('forward left missing: ' + str(forward.isnull().sum().sum()))
    main.to_csv('main_forward.csv', index=False)

    # main.to_sql('main_forward', engine)
