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
        main = pd.read_csv('Hyperopt_LightGBM/main.csv')
        engine = None
        print('local version running - main')

    main = main.dropna(how='any',axis=0)    # drop rows with NaN on important columns e.g. niq, atq...

    rnn_dict = {}

    for datacqtr in set(main['datacqtr']):
        rnn_dict[datacqtr] = main.loc[main['datacqtr']==datacqtr].set_index('gvkey')  # set gvkey as index for each quarterly DataFrame

        if save_csv == True:    # if set save_csv==True, save each quarterly df into individual csv, named after 'datacqtr'
            rnn_dict[datacqtr].to_csv(datacqtr+'.csv', index=False)

    end = time.time()
    print('LoadDataRNN running time {}'.format(end - start))    # enrire processing takes about 20s

    return rnn_dict

if __name__ == '__main__':
    rnn_dict = main(sql_version=False, save_csv=False)
