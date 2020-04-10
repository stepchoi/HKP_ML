import datetime as dt
import os
from collections import Counter

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import f1_score, r2_score, fbeta_score, precision_score, recall_score, \
    accuracy_score, cohen_kappa_score, hamming_loss, jaccard_score
from tqdm import tqdm


def prepare():
    '''map gvkey, datacqtr to consensus data'''

    # read raw tables
    gvkey = pd.read_csv('gvkey_sic', header=None)[0].to_list()  # our gvkey used
    name_map = pd.read_csv('name_map.csv', usecols=['gvkey', 'cusip']).drop_duplicates()  # use to map cusip to gvkey
    name_map['cusip'] = name_map['cusip'].astype(str).apply(lambda x: x.zfill(9)[:-1])  # convert cusip9 -> cusip8

    ibes = pd.read_csv('ibes_summary.csv', usecols=['CUSIP', 'STATPERS', 'FPEDATS', 'MEASURE', 'FISCALP', 'MEDEST',
                                                    'MEANEST'])  # ibes consensus data

    ibes.columns = [x.lower() for x in ibes.columns]
    ibes['cusip'] = ibes['cusip'].astype(str).apply(lambda x: x.zfill(8))

    ibes = pd.merge(ibes, name_map, on='cusip', how='inner')  # map cusip to gvkey
    ibes = ibes.loc[ibes['gvkey'].isin(gvkey)]  # filter gvkey we used
    ibes = ibes.loc[ibes['measure']=='NET']  # filter NET (net income) as measure

    ibes_dict = {}
    for name, g in ibes.groupby('fiscalp'):
        # g.to_csv('ibes_{}.csv'.format(name))
        ibes_dict[name] = g

    return ibes_dict['ANN'], ibes_dict['QTR']

def filter_date():
    ''' use last obsevation before cut-off (right before qtr, 3Q before ann)'''

    try:
        ann = pd.read_csv('ibes_ANN.csv')  # for annual forecast
        qtr = pd.read_csv('ibes_QTR.csv')  # for quarterly forecast
        print('local version run - ann, qtr')
    except:
        ann, qtr = prepare()

    # 1. filter observation date
    for df in [ann, qtr]:   # convert to timestamp
        df['statpers'] = pd.to_datetime(df['statpers'], format='%Y%m%d')
        df['fpedats'] = pd.to_datetime(df['fpedats'], format='%Y%m%d')

    qtr = qtr.loc[qtr['statpers']<=qtr['fpedats']] # filter quarterly forecast right before respective periods
    ann = ann.loc[ann['statpers']<=ann['fpedats'].apply(lambda x: x - relativedelta(months=9))] # filter quarterly forecast for 3Q before respective periods

    df_list = []
    for df in [ann, qtr]:
        df = df.sort_values(by=['gvkey','fpedats','statpers'])
        name = list(set(df['fiscalp']))[0]
        print(name)

        df = df.groupby(['gvkey', 'fpedats']).last().reset_index(drop=False)
        # df.to_csv('ibes_{}_last.csv'.format(name))
        df_list.append(df)
    return df_list[0], df_list[1]


class convert:
    def __init__(self, df):
        ''' input last observation estimation -> rename columns -> drop_nonseq -> add 'atq' '''
        atq = pd.read_csv('/Users/Clair/PycharmProjects/HKP_ML_DL/Hyperopt_LightGBM/niq_main.csv',
                          usecols=['gvkey','datacqtr','atq'])
        atq['datacqtr'] = pd.to_datetime(atq['datacqtr']) # prepare atq from original db

        df = df.filter(['gvkey','fpedats', 'medest','meanest'])
        df.columns = ['gvkey','datacqtr', 'medest','meanest'] # filter and rename

        df = self.drop_nonseq(df)
        self.df = pd.merge(atq, df, on=['gvkey','datacqtr'], how='right') # add 'atq'
        self.num_col = ['medest', 'meanest']

    def drop_nonseq(self, df):
        ''' fill in NaN for no recording period '''

        df['datacqtr'] = pd.to_datetime(df['datacqtr'])
        df_med = df.pivot(index='datacqtr',columns='gvkey',values='medest').unstack(0).reset_index(drop=False)
        df_mean = df.pivot(index='datacqtr',columns='gvkey',values='meanest').unstack(0).reset_index(drop=False)
        df = pd.merge(df_med, df_mean, on=['gvkey','datacqtr'], how='outer')
        df.columns = ['gvkey','datacqtr','medest','meanest']
        print('after drop non seq: ', df.shape)
        return df

    def qoq(self):
        ''' for QTR estimation -> convert to qoq format '''
        print('start qoq')
        d = self.df[self.num_col].shift(-1) - self.df[self.num_col]
        self.df[['medest', 'meanest']] = d.apply(lambda x: x.div(self.df['atq'])) # convert to qoq
        self.df.iloc[self.df.groupby('gvkey').tail(1).index, 2:] = np.nan   # remove effect due to cross gvkey operation
        return self.df

    def yoy(self):
        ''' for ANN estimation -> convert to yoy format '''
        print('start yoy')
        d = self.df[self.num_col].shift(-4) - self.df[self.num_col]
        self.df[['medest', 'meanest']] = d.apply(lambda x: x.div(self.df['atq']))
        self.df.iloc[self.df.groupby('gvkey').tail(4).index, 2:] = np.nan
        return self.df


def eval(Y_test, Y_test_pred):
    ''' calculate accuracy score with actual & concensus estimation (after qcut) '''

    result = {'loss': - accuracy_score(Y_test, Y_test_pred),
              'accuracy_score_test': accuracy_score(Y_test, Y_test_pred),
              'precision_score_test': precision_score(Y_test, Y_test_pred, average='micro'),
              'recall_score_test': recall_score(Y_test, Y_test_pred, average='micro'),
              'f1_score_test': f1_score(Y_test, Y_test_pred, average='micro'),
              'f0.5_score_test': fbeta_score(Y_test, Y_test_pred, beta=0.5, average='micro'),
              'f2_score_test': fbeta_score(Y_test, Y_test_pred, beta=2, average='micro'),
              'r2_score_test': r2_score(Y_test, Y_test_pred),
              # 'roc_auc_score_test': roc_auc_score(Y_test, Y_test_pred, average='micro'),
              # 'confusion_matrix': confusion_matrix(Y_test, Y_test_pred),
              "cohen_kappa_score": cohen_kappa_score(Y_test, Y_test_pred, labels=None),
              "hamming_loss": hamming_loss(Y_test, Y_test_pred),
              "jaccard_score": jaccard_score(Y_test, Y_test_pred, labels=None, average='macro')}
    # print(result)
    # pd.DataFrame.from_dict(result, orient='index', columns=['score'])
    return result

class evaluate:

    '''input converted estimation (qoq, yoy) -> qcut estimation/actual -> evaluate score'''

    def __init__(self, ibes_df, y_type, q):

        '''input converted estimation (qoq, yoy) -> add 'niq' actual values'''

        self.qcut_q = q # define class objects
        self.y_type = y_type

        self.niq = pd.read_csv('/Users/Clair/PycharmProjects/HKP_ML_DL/Hyperopt_LightGBM/niq_main.csv',
                               usecols=['gvkey', 'datacqtr', 'qoq', 'yoyr'])  # read actual niq
        self.niq['datacqtr'] = pd.to_datetime(self.niq['datacqtr'])

        self.all_bins = self.get_all_bins() # get qcut bins for all rolling period

        self.ibes_df = pd.merge(self.niq[['gvkey', 'datacqtr', y_type]], ibes_df, on=['gvkey', 'datacqtr'], how='right')
        print(ibes_df.shape)
        print(self.ibes_df.shape)


    def qcut_y(self, df, col):  # qcut y with train_y cut_bins

        ''' qcut each training set period -> return cut_bins for each training set '''

        bins = {}
        period_1 = dt.datetime(2008, 3, 31)

        for i in tqdm(range(40)):  # change to 40 for full 40 sets, change to False to stop saving csv

            end = period_1 + i * relativedelta(months=3)  # define testing period
            start = end - relativedelta(years=20)  # define training period
            train = df.loc[(start <= df['datacqtr']) & (df['datacqtr'] < end)]  # train df = 80 quarters
            train[col], cut_bins = pd.qcut(train[col], q=self.qcut_q, labels=range(self.qcut_q), retbins=True)

            bins[end.strftime('%Y-%m-%d')] = {}
            bins[end.strftime('%Y-%m-%d')]['cut_bins'] = cut_bins

            cut_bins[0] = -np.inf
            cut_bins[-1] = np.inf

        return bins

    def get_all_bins(self):

        ''' qcut for 'qoq' and 'yoyr' '''

        all_bins = {}
        for y in ['qoq', 'yoyr']: # 'yoy',
            all_bins[y] = self.qcut_y(self.niq, y)
        return all_bins

    def dict_to_df(self, dict, measure):

        ''' convert dictionary to dataframe for to_csv '''

        df = pd.DataFrame()
        for date in dict.keys():
            for score in dict[date].keys():
                df.loc[date, score] = dict[date][score]
        df['type'] = measure
        return df

    def eval_all(self):

        ''' cut actual/estimation testing for each set period (*40) '''

        med_records = {}
        mean_records = {}

        r2 = {}

        for i in tqdm(set(self.ibes_df['datacqtr'])):  # evaluate all testing period

            if (i <= pd.Timestamp(2018, 1, 1, 1)) & (i >= pd.Timestamp(2008, 1, 1, 1)):

                testing_period = self.ibes_df.loc[self.ibes_df['datacqtr'] == i]

                cut_bins = self.all_bins[self.y_type][i.strftime('%Y-%m-%d')]['cut_bins']

                self.all_bins[self.y_type][i.strftime('%Y-%m-%d')]['test_len_act'] = testing_period[self.y_type].notnull().sum()
                self.all_bins[self.y_type][i.strftime('%Y-%m-%d')]['test_len_est'] = testing_period['medest'].notnull().sum()

                testing_period = testing_period.dropna(how='any')

                df_print = testing_period.copy()

                r2[i] = r2_score(testing_period[self.y_type], testing_period['medest'])

                for col in [self.y_type, 'medest','meanest']:    # qcut testing period (actual & consensus)
                    testing_period[col] = pd.cut(testing_period[col], bins=cut_bins, labels=range(self.qcut_q))
                    self.all_bins[self.y_type][i.strftime('%Y-%m-%d')]['test_count_{}'.format(col)] = list(
                        dict(Counter(testing_period[col].to_list())).values())

                # if i == pd.Timestamp(2017, 9, 30, 0, 0, 0):
                #     check_print([df_print, testing_period], sort=False)

                med_records[i.strftime('%Y-%m-%d')] = eval(testing_period[self.y_type], testing_period['medest'])
                mean_records[i.strftime('%Y-%m-%d')] = eval(testing_period[self.y_type], testing_period['meanest'])

        print(pd.DataFrame.from_records(r2, index=[0]).transpose())
        df_full = pd.concat([self.dict_to_df(med_records, 'medest'), self.dict_to_df(mean_records, 'meanest')], axis=0)

        pd.DataFrame(self.all_bins[self.y_type]).transpose().to_csv('cutbins_{}_test_act_est.csv'.format(self.y_type))

        return df_full

def main():
    try:
        ann = pd.read_csv('ibes_ANN_last.csv')  # for annual forecast
        qtr = pd.read_csv('ibes_QTR_last.csv')  # for quarterly forecast
        print('local version run - ann_last, qtr_last')
    except:
        ann, qtr = filter_date()

    q = 6   # define qcut bins

    # # convert QTR estimation to qoq and evaluate
    qtr = convert(qtr).qoq()
    df_full = evaluate(ibes_df=qtr, y_type='qoq', q=q).eval_all()
    df_full.sort_index().to_csv('consensus_qoq{}.csv'.format(q))

    # convert ANN estimation to yoy and evaluate
    ann = convert(ann).yoy()
    df_full_ann = evaluate(ibes_df=ann, y_type='yoyr', q=q).eval_all()
    df_full_ann.sort_index().to_csv('consensus_yoyr{}.csv'.format(q))

if __name__ == '__main__':
    os.chdir('/Users/Clair/PycharmProjects/HKP_ML_DL/Preprocessing/raw/ibes')

    # prepare()
    # filter_date()
    main()
