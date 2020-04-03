import datetime as dt

import numpy as np
from sklearn.preprocessing import StandardScaler

from Preprocessing.LoadData import load_data


class clean_rnn:
    '''df = big table; y_type = ['yoy','qoq']; testing_period are timestamp'''

    def __init__(self):

        '''main = [gvkey, datacqtr + all_x] * [all_datacqtr]'''
        self.main = load_data(lag_year=0, sql_version=True).iloc[:,:-3]

        '''qtr_main = [gvkey, datacqtr + all_x] * [single datacqtr]'''
        self.qtr_dict_df = self.split_by_datacqtr()

        self.col = main.columns[2:-3]
        print(self.col)

        self.x_dict = {}
        self.y_dict = {}

    def split_by_datacqtr(self):
        qtr_dict_df = {}
        for qtr in set(self.main['datacqtr']):
            qtr_dict_df[qtr] = self.main.loc[self.main['datacqtr']==qtr]
            print(qtr_dict_df)
        return qtr_dict_df

    def reshape_3d(self, sample_no):

        period_1 = dt.datetime(1988, 3, 31)

        arr_3d_dict = {}

        for i in tqdm(range(sample_no)): # to be changed

            start = period_1 + i * relativedelta(months=3)  # start period of training set

            '''filter gvkey'''
            gvkey = set(self.qtr_dict_df[start])
            for k in tqdm(range(20)):
                lag_period = start + (k-20)*relativedelta(months=3)
                gvkey = gvkey & set(self.qtr_dict_df[lag_period])

            print(len(gvkey), gvkey)

            arr = []
            for k in tqdm(range(20)):
                lag_period = start + (k-20) * relativedelta(months=3)
                period_df = self.qtr_dict_df[lag_period].loc[self.qtr_dict_df['gvkey'].isin(gvkey)]
                print(period_df)
                period_df_std = StandardScaler().fit(period_df.iloc[:,2:])
                print(period_df_std)
                arr.append(period_df_std)

            arr_3d_dict[i] = np.array(arr)
            print(arr_3d_dict[i].shape)

        return arr_3d_dict

if __name__ == '__main__':
    arr_3d_dict = clean_rnn().reshape_3d(sample_no=1)