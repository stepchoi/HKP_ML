import time

import pandas as pd
from sqlalchemy import create_engine


def main(sql_version=False, save_csv=False):

    print('-------------- start load data --------------')
    start = time.time()

    if sql_version is True:  # sql version read TABLE from Postgre SQL
        db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
        engine = create_engine(db_string)
        main = pd.read_sql('SELECT * FROM main', engine)
    else:  # local version read TABLE from local csv files -> faster
        main = pd.read_csv('main.csv')
        engine = None
        print('local version running - main')

    main = main.dropna(how='any',axis=0)

    rnn_dict = {}

    for datacqtr in set(main['datacqtr']):
        rnn_dict[datacqtr] = main.loc[main['datacqtr']==datacqtr].set_index('gvkey')
        rnn_dict_len[datacqtr] = len(main.loc[main['datacqtr']==datacqtr])
        if save_csv == True:
            rnn_dict[datacqtr].to_csv(datacqtr+'.csv', index=False)

    end = time.time()
    print('LoadDataRNN running time {}'.format(end - start))

    return rnn_dict

if __name__ == '__main__':
    rnn_dict = main(sql_version=False, save_csv=False)
