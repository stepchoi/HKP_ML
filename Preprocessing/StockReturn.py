import numpy as np
import pandas as pd
from LoadData import convert_to_float32
from sqlalchemy import create_engine

def import_sp():
    try:
        import os
        os.chdir('/Users/Clair/PycharmProjects/HKP_ML_DL')
        sp = pd.read_csv('macro_main.csv', usecols=['datacqtr', 'S&P_qoq'])
        stock = pd.read_csv('stock_raw.csv', usecols=['gvkey', 'datadate', 'prccm'])
    except:
        db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
        engine = create_engine(db_string)
        sp = pd.read_sql('SELECT datacqtr, S&P_qoq FROM macro_main', engine)
        sp = pd.read_sql('SELECT gvkey, datadate, prccm FROM stock_raw', engine)

    return sp, stock

def main():
    sp, stock = import_sp()
    stock.columns = ['gvkey','datacqtr','stock_price']
    stock['datacqtr'] = pd.to_datetime(stock['datacqtr'], format='%Y%m%d').dt.strftime('%Y-%m-%d') # convert to timestamp
    stock['month'] = pd.to_datetime(stock['datacqtr'], format='%Y-%m-%d').dt.strftime('%m').astype(int)  # convert to timestamp
    stock = stock.loc[stock['month'].isin([3,6,9,12])]
    del stock['month']

    new = pd.merge(stock, sp, how='left', on='datacqtr')

    new_ret = new.groupby('gvkey').apply(lambda x: x['stock_price'].div(x['stock_price'].shift(1)).sub(1)).reset_index(drop=True)
    new = pd.concat([new, new_ret], axis =1)
    new.columns = ['gvkey','datacqtr','stock_price','S&P_qoq','stock_ret']
    new = new.replace([np.inf, -np.inf], np.nan)
    new['return'] = new['stock_ret'] - new['S&P_qoq']
    print(new)
    convert_to_float32(new)
    new = new.filter(['gvkey','datacqtr','return'])
    print(new.info())
    # new.to_csv('stock_main.csv', index=False)

if __name__ == "__main__":
    main()
    # print(main)