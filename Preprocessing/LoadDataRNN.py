import datetime as dt
import gc

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from tqdm import tqdm

from Hyperopt_LightGBM.LoadData import load_data


class load_data_rnn:
    '''df = big table; y_type = ['yoy','qoq']; testing_period are timestamp'''

    def __init__(self,lag_year, sql_version):

        self.all_bins = self.get_all_bins(sql_version)

        main = load_data(lag_year=lag_year, sql_version=sql_version)

        scaler = StandardScaler()
        print('f')
        df = pd.DataFrame(scaler.fit_transform(main.iloc[:,2:-4]), columns=main.columns[2:-4])
        main = pd.concat([main.iloc[:,:2], df, main.iloc[:,-4:]], axis=1)
        print('f')

        # print(main)

        self.fincol = main.columns[2:156].to_list()
        self.ecocol = main.columns[-15:-4].to_list()
        # print(self.fincol, self.ecocol)
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

            train[col], cut_bins = pd.qcut(train[col], q=self.qcut_q, labels=range(self.qcut_q), retbins=True)
            print(cut_bins)

            cut_bins[0] = -np.inf
            cut_bins[3] = np.inf
            # print(cut_bins[0])
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
            exist = pd.read_csv('exist.csv')

        self.niq['k']=self.niq['gvkey'].astype(str) + self.niq['datacqtr'].astype(str)
        e=exist['gvkey'].astype(str) + exist['datacqtr'].astype(str)
        self.niq = self.niq.loc[self.niq['k'].isin(e)]
        print(self.niq.shape)
        self.niq['datacqtr'] = pd.to_datetime(self.niq['datacqtr'])
        all_bins = {}
        for y in ['qoq', 'yoyr']: # 'yoy',
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
            # print(period.isnull().sum())

            arr.append(period[self.fincol + self.ecocol].values)
            for lag in range(self.lag_year * 4 - 1):  # when lag_year is 5, here loop over past 19 quarter
                x_col = ['{}_lag{}'.format(k, str(lag + 1).zfill(2)) for k in self.fincol] + self.ecocol
                arr.append(period[x_col].values)
            arr_3d_dict[qtr] = np.array(arr)

            # (20, company, v) -> (company, 20, v)
            arr_3d_dict[qtr] = np.rot90(arr_3d_dict[qtr], axes=(0, 1))
            y_dict['qoq'][qtr] = period['qoq'].values
            y_dict['yoyr'][qtr] = period['yoyr'].values

        return arr_3d_dict, y_dict

    def sampling(self, sample_no, y_type):

        period_base = dt.datetime(2008, 3, 31)

        end = period_base + sample_no * relativedelta(months=3)  # start period of training set
        start = end - relativedelta(years=20)

        # sample for y
        cut_bins = self.all_bins[y_type][end.strftime('%Y-%m-%d')]

        # sample for x
        samples = {}

        for k in self.arr_3d_dict.keys():
            if (k>=start) & (k<end):
                # db = pd.DataFrame(self.y_dict[y_type][k])
                # da = pd.DataFrame(y)
                # print(db.loc[da.isnull()])
                # print(da.isnull().sum())
                y_list.append(y)
                x_list.append(self.arr_3d_dict[k])

        samples['x'] = pd.concat(x_list)
        samples['y'] = pd.concat(y_list)
        print(samples['x'].shape)
        print(samples['y'].shape)

        from collections import Counter
        print(Counter(samples['y']))

        samples['y'] = pd.cut(samples['y'], bins=cut_bins, labels=range(self.qcut_q))
        print(Counter(samples['y']))

        # print(len(samples), len(samples['x']), len(samples['y']), len(samples['x'][0]), len(samples['y'][0]))
        return samples

if __name__ == '__main__':

    import os
    os.chdir('/home/loratech/PycharmProjects/HKP_ML_DL/Hyperopt_LightGBM')

    # samples_set1 equivalent to the first csv in LightGBM version
    # it contains 80 3d_array
    # each 3d_array = (20, companies, variables=165)

    sample_class = load_data_rnn(lag_year=0, sql_version=False)

    for i in range(1): # set = 40 if return 40 samples
        samples_set1 = sample_class.sampling(i, y_type='qoq')


        x = samples_set1['x'][0]
        y = samples_set1['y'][0]
        print(x.shape)

        for q in range(79):
            x = np.concatenate((x, samples_set1['x'][q + 1]))
            y = np.concatenate((y, samples_set1['y'][q + 1]))

        from collections import Counter
        print(Counter(y))

        # print(np.isnan(x).sum())
        # print(np.isnan(y).sum())

        # print(x.shape)
        # print(y.shape)
        pass