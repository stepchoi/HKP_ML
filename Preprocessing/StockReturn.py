import numpy as np
import pandas as pd
from LoadData import convert_to_float32


def main():
    stock = pd.read_csv('stock.csv', usecols=['gvkey','datadate','prccm'])
    stock.columns = ['gvkey','datacqtr','stock_price']
    stock['datacqtr'] = pd.to_datetime(stock['datacqtr'], format='%Y%m%d').apply(lambda x: x.strftime('%Y-%m-%d'))  # convert to timestamp
    stock['month'] = pd.to_datetime(stock['datacqtr'], format='%Y-%m-%d').apply(lambda x: x.strftime('%m')).astype(int)  # convert to timestamp
    stock = stock.loc[stock['month'].isin([3,6,9,12])]
    del stock['month']

    sp = pd.read_csv('GSPC.csv', usecols=['Date','Close','Adj Close'])
    sp.columns = ['datacqtr','sp500','sp500_adj']
    sp['datacqtr'] = [x[:7] for x in sp['datacqtr']]
    sp['datacqtr'] = sp['datacqtr'].apply(lambda x: pd.Period(x, freq='M').to_timestamp(how='end').strftime('%Y-%m-%d'))

    new = pd.merge(stock, sp, how='left', on='datacqtr')

    new_ret = new.groupby('gvkey').apply(lambda x: x.iloc[:,2:].div(x.iloc[:,2:].shift(1)).sub(1)).reset_index(drop=True)
    new_ret.columns = ['stock_ret','sp500_ret','sp500_ret_adj']
    new = pd.concat([new, new_ret], axis =1)
    new = new.replace([np.inf, -np.inf], np.nan)
    new['return'] = new['stock_ret'] - new['sp500_ret']
    new['return_adj'] = new['stock_ret'] - new['sp500_ret_adj']
    convert_to_float32(new)
    print(new.info())

    new.to_csv('stock_return.csv', index=False)

if __name__ == "__main__":
    main()
