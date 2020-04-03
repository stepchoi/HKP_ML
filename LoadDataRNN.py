import datetime as dt
import gc

import numpy as np
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from Preprocessing.LoadData import load_data


class clean_rnn:
    '''df = big table; y_type = ['yoy','qoq']; testing_period are timestamp'''

    def __init__(self,lag_year, sql_version):

        main = load_data(lag_year=lag_year, sql_version=sql_version)
        main.iloc[:,2:-3] = StandardScaler().fit_transform(main.iloc[:,2:-3])

        self.fincol = main.columns[2:156].to_list()
        self.ecocol = main.columns[-14:-3].to_list()
        self.lag_year = lag_year

        self.arr_3d_dict = self.reshape_3d(main)

        del main
        gc.collect()

    def reshape_3d(self, main):
        arr_3d_dict = {}

        for qtr in tqdm(set(main['datacqtr'])):
            arr = []
            period = main.loc[main['datacqtr']==qtr]
            arr.append(period[self.fincol + self.ecocol].values)
            for lag in range(self.lag_year * 4 - 1):  # when lag_year is 5, here loop over past 19 quarter
                x_col = ['{}_lag{}'.format(k, str(lag + 1).zfill(2)) for k in self.fincol] + self.ecocol
                arr.append(period[x_col].values)
            arr_3d_dict[qtr] = np.array(arr)

            print(qtr, arr_3d_dict[qtr].shape)

        return arr_3d_dict

    def sampling(self, sample_no):

        period_1 = dt.datetime(2008, 3, 31)
        samples = []
        for i in tqdm(range(sample_no)):  # to be changed
            end = period_1 + i * relativedelta(months=3)  # start period of training set
            start = end - relativedelta(years=20)

            for k in self.arr_3d_dict.keys():
                if (k>=start) & (k<end):
                    samples.append(self.arr_3d_dict[k])
        print(len(samples))
        return samples

if __name__ == '__main__':
    # samples_set1 equivalent to the first csv in LightGBM version
    # it contains 80 3d_array
    # each 3d_array = (20, companies, variables=154)

    samples_set1 = clean_rnn(lag_year=5, sql_version=True).sampling(1)