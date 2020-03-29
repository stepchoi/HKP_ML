import numpy as np
import pandas as pd
from LoadData import convert_to_float32
from PrepareDatabase import drop_nonseq
from sqlalchemy import create_engine

def trim_outlier(df, prc=0.01):
    pmax = df.quantile(q=(1-prc),axis=0)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(pmax)
    df = df.mask(df>pmax,pmax, axis=0)
    return df

def import_sp():
    try:
        import os
        os.chdir('/Users/Clair/PycharmProjects/HKP_ML_DL/Preprocessing/raw')
        sp = pd.read_csv('/Users/Clair/PycharmProjects/HKP_ML_DL/Hyperopt_LightGBM/macro_main.csv', usecols=['datacqtr', 'S&P_qoq','S&P_yoy'])
        stock = pd.read_csv('stock_raw.csv', usecols=['gvkey', 'datadate', 'prccm'])
    except:
        db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
        engine = create_engine(db_string)
        sp = pd.read_sql('SELECT datacqtr, S&P_qoq, S&P_qoq FROM macro_main', engine)
        stock = pd.read_sql('SELECT gvkey, datadate, prccm FROM stock_raw', engine)

    return sp, stock

def main():

    sp, stock = import_sp() # read df stock, macro_main
    stock.columns = ['gvkey','datacqtr','stock_price']

    stock = stock.dropna(how='any') # drop records with NaN stock_price
    stock = stock.drop_duplicates(subset=['gvkey', 'datacqtr'], keep='last') # drop duplicated records due to change ticker

    stock['datacqtr'] = pd.to_datetime(stock['datacqtr'], format='%Y%m%d').dt.strftime('%Y-%m-%d') # convert to timestamp
    stock['month'] = pd.to_datetime(stock['datacqtr'], format='%Y-%m-%d').dt.strftime('%m').astype(int)
    stock = stock.loc[stock['month'].isin([3, 6, 9, 12])] # select quarter end record

    stock = drop_nonseq(stock).reset_index(drop=True)  # drop non-sequential records
    del stock['datacqtr_no']    # this columns is create with drop_nonseq function

    stock.loc[stock['stock_price'] < 1] = np.nan

    print(len(stock))
    del stock['month']

    '''
    1. change sp500 -> macro to 1 q later
    2. upload macro/price -> sqldb + remote pc
    '''

    new = pd.merge(stock, sp, how='left', on='datacqtr')    # merge individual stock price & s&p500 from macro_main

    new_ret = new['stock_price'].pct_change(periods=1).shift(-1)
    new_ret.iloc[new.groupby('gvkey').tail(1).index] = np.nan

    new_ret_yoy = new['stock_price'].pct_change(periods=4).shift(-1)
    new_ret_yoy.iloc[new.groupby('gvkey').tail(1).index] = np.nan
    new_ret_yoy.iloc[new.groupby('gvkey').head(3).index] = np.nan

    new = pd.concat([new, new_ret, new_ret_yoy], axis =1)
    new.columns = ['gvkey','datacqtr','stock_price','S&P_qoq','S&P_yoy','stock_ret_qoq','stock_ret_yoy']

    # trim outlier by a maximum value
    print(new.describe())
    new[['stock_ret_qoq', 'stock_ret_yoy']] = trim_outlier(new[['stock_ret_qoq', 'stock_ret_yoy']])
    print(new.describe())

    new['return_qoq'] = new['stock_ret_qoq'] - new['S&P_qoq']   # calculate relative return to equivalent S&P500 return
    new['return_yoy'] = new['stock_ret_yoy'] - new['S&P_yoy']   # calculate relative return to equivalent S&P500 return

    convert_to_float32(new)

    new = new.filter(['gvkey','datacqtr','return_qoq','return_yoy'])
    print(new.info())
    new.to_csv('stock_main.csv', index=False)

if __name__ == "__main__":
    main()