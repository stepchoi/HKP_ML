import numpy as np
import pandas as pd
from LoadData import convert_to_float32
from PrepareDatabase import drop_nonseq
from sqlalchemy import create_engine

def import_sp():
    try:
        import os
        os.chdir('/Users/Clair/PycharmProjects/HKP_ML_DL/Hyperopt_LightGBM')
        sp = pd.read_csv('macro_main.csv', usecols=['datacqtr', 'S&P_qoq'])
        stock = pd.read_csv('stock_raw.csv', usecols=['gvkey', 'datadate', 'prccm'])
    except:
        db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
        engine = create_engine(db_string)
        sp = pd.read_sql('SELECT datacqtr, S&P_qoq FROM macro_main', engine)
        sp = pd.read_sql('SELECT gvkey, datadate, prccm FROM stock_raw', engine)

    return sp, stock

def main():

    sp, stock = import_sp() # read df stock, macro_main
    stock.columns = ['gvkey','datacqtr','stock_price']

    stock = stock.dropna(how='any') # drop records with NaN stock_price
    stock = stock.drop_duplicates(subset=['gvkey', 'datacqtr'], keep='last') # drop duplicated records due to change ticker

    stock['datacqtr'] = pd.to_datetime(stock['datacqtr'], format='%Y%m%d').dt.strftime('%Y-%m-%d') # convert to timestamp
    stock['month'] = pd.to_datetime(stock['datacqtr'], format='%Y-%m-%d').dt.strftime('%m').astype(int)
    stock = stock.loc[stock['month'].isin([3, 6, 9, 12])] # select quarter end record

    stock = drop_nonseq(stock)  # drop non-sequential records
    del stock['datacqtr_no']    # this columns is create with drop_nonseq function

    print(len(stock))
    del stock['month']


    new = pd.merge(stock, sp, how='left', on='datacqtr')    # merge individual stock price & s&p500 from macro_main

    new_ret = new.groupby('gvkey').apply(lambda x: x['stock_price'].div(x['stock_price'].shift(1)).sub(1)).reset_index(drop=True)   # qoq return for stock
    new = pd.concat([new, new_ret], axis =1)
    print(new)
    new.columns = ['gvkey','datacqtr','stock_price','S&P_qoq','stock_ret']
    new = new.replace([np.inf, -np.inf], np.nan)
    new['return'] = new['stock_ret'] - new['S&P_qoq']   # calculate relative return to equivalent S&P500 return
    convert_to_float32(new)
    new = new.filter(['gvkey','datacqtr','return'])
    print(new.info())
    new.to_csv('stock_main.csv', index=False)

if __name__ == "__main__":
    import os
    os.chdir('/Users/Clair/PycharmProjects/HKP_ML_DL/Hyperopt_LightGBM')
    dep.
    stock = drop_nonseq(stock)  # drop
