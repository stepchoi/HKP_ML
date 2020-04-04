import datetime as dt
import gc

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from tqdm import tqdm

from Preprocessing.LoadData import load_data


class load_data_rnn:
    '''df = big table; y_type = ['yoy','qoq']; testing_period are timestamp'''

    def __init__(self,lag_year, sql_version):

        self.all_bins = self.get_all_bins(sql_version)

        main = load_data(lag_year=lag_year, sql_version=sql_version)
        main.iloc[:,2:-3] = StandardScaler().fit_transform(main.iloc[:,2:-3])

        self.fincol = main.columns[2:156].to_list()
        self.ecocol = main.columns[-14:-3].to_list()
        self.lag_year = lag_year

        self.arr_3d_dict, self.y_dict = self.reshape_3d(main)

        del main
        gc.collect()

    def qcut_y(self, df, col):  # qcut y with train_y cut_bins

        self.qcut_q = 3

        bins = {}
        period_1 = dt.datetime(2008, 3, 31)
        for i in tqdm(range(40)):  # change to 40 for full 40 sets, change to False to stop saving csv
            end = period_1 + i * relativedelta(months=3)  # define testing period
            start = end - relativedelta(years=20)  # define training period
            train = df.loc[(start <= df['datacqtr']) & (df['datacqtr'] < end)]  # train df = 80 quarters
            for i in range(self.qcut_q):
                try:
                    train[col], cut_bins = pd.qcut(train[col], q=self.qcut_q, labels=range(self.qcut_q-i),
                                                   retbins=True, duplicates='drop')
                    print('range', self.qcut_q-i)
                    self.qcut_range = self.qcut_q-i
                    break
                except:
                    continue
            bins[end.strftime('%Y-%m-%d')] = cut_bins
        # d = pd.DataFrame.from_dict(bins, orient='index',columns=[0,1,2,3])
        return bins

    def get_all_bins(self, sql_version):
        if sql_version == True:
            db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
            engine = create_engine(db_string)
            self.niq = pd.read_sql('SELECT * FROM niq_main', engine)
        else:
            self.niq = pd.read_csv('niq_main.csv')

        self.niq['datacqtr'] = pd.to_datetime(self.niq['datacqtr'])
        all_bins = {}
        for y in ['qoq', 'yoy_rolling']: # 'yoy',
            all_bins[y] = self.qcut_y(self.niq, y)
        return all_bins

    def reshape_3d(self, main):
        arr_3d_dict = {}
        y_dict = {}
        y_dict['yoyr'] = {}
        y_dict['qoq'] = {}

        for qtr in tqdm(set(main['datacqtr'])):
            arr = []
            period = main.loc[main['datacqtr']==qtr]
            arr.append(period[self.fincol + self.ecocol].values)
            for lag in range(self.lag_year * 4 - 1):  # when lag_year is 5, here loop over past 19 quarter
                x_col = ['{}_lag{}'.format(k, str(lag + 1).zfill(2)) for k in self.fincol] + self.ecocol
                arr.append(period[x_col].values)
            arr_3d_dict[qtr] = np.array(arr)

            # (20, company, v) -> (company, 20, v)
            arr_3d_dict[qtr] = np.rot90(arr_3d_dict[qtr], axes=(0, 1))
            y_dict['qoq'][qtr] = period['qoq'].values
            y_dict['yoyr'][qtr] = period['yoy_rolling'].values

        return arr_3d_dict, y_dict

    def sampling(self, sample_no, y_type):

        period_base = dt.datetime(2008, 3, 31)

        end = period_base + sample_no * relativedelta(months=3)  # start period of training set
        start = end - relativedelta(years=20)

        # sample for y
        cut_bins = self.all_bins[y_type][end.strftime('%Y-%m-%d')]


        # sample for x
        samples = {}
        samples['x'] = []
        samples['y'] = []

        for k in self.arr_3d_dict.keys():
            if (k>=start) & (k<end):
                y = pd.cut(self.y_dict[y_type][k], bins=cut_bins, labels=range(self.qcut_range))
                samples['y'].append(y)
                samples['x'].append(self.arr_3d_dict[k])

        # print(len(samples), len(samples['x']), len(samples['y']), len(samples['x'][0]), len(samples['y'][0]))
        return samples

if __name__ == '__main__':

    # import os
    # os.chdir('/Users/Clair/PycharmProjects/HKP_ML_DL/Hyperopt_LightGBM')

    # samples_set1 equivalent to the first csv in LightGBM version
    # it contains 80 3d_array
    # each 3d_array = (20, companies, variables=165)

    sample_class = load_data_rnn(lag_year=5, sql_version=True)

    for i in range(1): # set = 40 if return 40 samples
        samples_set1 = sample_class.sampling(i, y_type='qoq')
        print(samples_set1['x'])
        print(samples_set1['y'])
        pass