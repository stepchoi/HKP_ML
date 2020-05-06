import datetime as dt
import os
from collections import Counter

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import f1_score, r2_score, fbeta_score, precision_score, recall_score, \
    accuracy_score, cohen_kappa_score, hamming_loss, jaccard_score
from tqdm import tqdm


def prepare_act():

    gvkey = pd.read_csv('gvkey_sic', header=None)[0].to_list()  # our gvkey used
    name_map = pd.read_csv('name_map2.csv', usecols=['gvkey','datadate','cusip']).drop_duplicates()  # use to map cusip to gvkey
    name_map['cusip'] = name_map['cusip'].astype(str).apply(lambda x: x.zfill(9)[:-1])  # convert cusip9 -> cusip8

    # ibes = pd.read_csv('ibes_summary.csv', usecols=['CUSIP', 'STATPERS', 'FPEDATS', 'MEASURE', 'FISCALP', 'MEDEST',
    #                                                 'MEANEST','ACTUAL'])  # ibes consensus data

    ibes = pd.read_csv('ibes_summary3.csv')
    ibes = ibes.dropna(subset=['INT0DATS','CUSIP'])
    ibes['INT0DATS'] =  ibes['INT0DATS'].astype(int)
    print(ibes.describe())


    ibes.columns = [x.lower() for x in ibes.columns]
    ibes['cusip'] = ibes['cusip'].astype(str).apply(lambda x: x.zfill(8))

    # ibes = pd.merge(ibes, name_map, left_on=['fpedats', 'cusip'], right_on=['datadate','cusip'], how='inner')  # map cusip to gvkey
    ibes = pd.merge(ibes, name_map, left_on=['int0dats', 'cusip'], right_on=['datadate','cusip'], how='inner')  # map cusip to gvkey

    ibes = ibes.loc[ibes['gvkey'].isin(gvkey)]  # filter gvkey we used
    # ibes = ibes.loc[ibes['measure']=='NET']  # filter NET (net income) as measure

    # for name, g in ibes.groupby(['measure']):
    #     print(name, len(g))
    # exit(0)


    niq = pd.read_csv('/Users/Clair/PycharmProjects/HKP_ML_DL/Hyperopt_LightGBM/niq_main.csv', usecols=['gvkey', 'datacqtr', 'niq'])  # read actual niq
    niq['datacqtr'] = pd.to_datetime(niq['datacqtr'], format='%Y-%m-%d').dt.strftime('%Y%m%d').astype(int)
    print('niq shape', niq.shape)
    ibes = pd.merge(ibes, niq, left_on=['int0dats','gvkey'], right_on=['datacqtr','gvkey'], how='inner')

    print('all', ibes.shape)
    ibes = ibes.loc[ibes['int0a']==ibes['niq']]
    print('same', ibes.shape)

    # ibes.to_csv('ibes_act.csv', index=False)

    # ibes_dict = {}
    # for name, g in ibes.groupby('fiscalp'):
    #     # g.to_csv('ibes_{}.csv'.format(name))
    #     ibes_dict[name] = g

    # return ibes_dict['ANN'], ibes_dict['QTR']


def prepare():
    '''map gvkey, datacqtr to consensus data'''

    # read raw tables
    gvkey = pd.read_csv('gvkey_sic', header=None)[0].to_list()  # our gvkey used
    name_map = pd.read_csv('name_map3.csv', usecols=['gvkey','datadate','cusip']).drop_duplicates()  # use to map cusip to gvkey
    name_map['cusip'] = name_map['cusip'].astype(str).apply(lambda x: x.zfill(9)[:-1])  # convert cusip9 -> cusip8

    ibes = pd.read_csv('ibes_summary3.csv', usecols=['CUSIP', 'STATPERS', 'FPEDATS', 'MEASURE', 'FISCALP', 'MEDEST',
                                                    'MEANEST', 'ACTUAL'])  # ibes consensus data

    ibes.columns = [x.lower() for x in ibes.columns]
    ibes['cusip'] = ibes['cusip'].astype(str).apply(lambda x: x.zfill(8))

    ibes = pd.merge(ibes, name_map, left_on=['fpedats', 'cusip'], right_on=['datadate','cusip'], how='inner')  # map cusip to gvkey
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
        print('filter date (ann, qtr): ', ann.shape, qtr.shape)
        # ann, qtr = prepare_act()

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

        df = df.filter(['gvkey','fpedats', 'medest','meanest', 'actual'])
        # print(df)
        df.columns = ['gvkey','datacqtr', 'medest','meanest','actual'] # filter and rename

        df = self.drop_nonseq(df)
        self.df = pd.merge(atq, df, on=['gvkey','datacqtr'], how='right') # add 'atq'
        self.num_col = ['medest', 'meanest','actual']

    def drop_nonseq(self, df):

        ''' fill in NaN for no recording period '''

        df['datacqtr'] = pd.to_datetime(df['datacqtr'])

        # all_quarter = pd.DataFrame(index=sort(set(df['datacqtr'])), columns=set(df['gvkey']))
        # print(all_quarter)
        # all_quarter_unstack = all_quarter.unstack(0).reset_index(drop=False)
        # print(all_quarter_unstack)
        # exit(0)

        df_med = df.pivot(index='datacqtr',columns='gvkey',values='medest').unstack(0).reset_index(drop=False)
        df_mean = df.pivot(index='datacqtr',columns='gvkey',values='meanest').unstack(0).reset_index(drop=False)
        df_act = df.pivot(index='datacqtr',columns='gvkey',values='actual').unstack(0).reset_index(drop=False)

        df = pd.merge(df_med, df_mean, on=['gvkey','datacqtr'], how='outer')
        df = pd.merge(df, df_act, on=['gvkey','datacqtr'], how='outer')

        df.columns = ['gvkey','datacqtr','medest','meanest','actual']
        print('after drop non seq: ', df.shape)
        return df

    def qoq(self):
        ''' for QTR estimation -> convert to qoq format '''
        print('start qoq')
        for col in self.num_col:
            self.df[col] = self.df[col].shift(-1).sub(self.df['actual']).div(self.df['atq']) # convert to qoq

        self.df.iloc[self.df.groupby('gvkey').tail(1).index, 2:] = np.nan   # remove effect due to cross gvkey operation
        return self.df

    def yoy(self):
        ''' for ANN estimation -> convert to yoy format '''
        print('start yoy')

        '''
        ANN consensus: 2008Y -> 2009Y-2008Y/2008Q4-atq
        QTR actual y from Compustat: 2008Q4 -> 2008Q4 -> (2009Q1..2009Q4) - (2008Q1..2008Q4)/2008Q4-atq
        '''
        for col in self.num_col:
            self.df[col] = self.df[col].shift(-4).sub(self.df['actual']).div(self.df['atq']) # convert to qoq

        self.df.iloc[self.df.groupby('gvkey').tail(4).index, 2:] = np.nan
        # self.df['m'] = [str(x)[5] for x in self.df['datacqtr']]
        # print(self.df['datacqtr'].dt.strftime('%m'))
        self.df = self.df.loc[self.df['datacqtr'].dt.strftime('%m')=='12']
        # print(self.df.head())
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
        exist = pd.read_csv('/Users/Clair/PycharmProjects/HKP_ML_DL/Hyperopt_LightGBM/exist.csv')
        self.niq['k']=self.niq['gvkey'].astype(str) + self.niq['datacqtr'].astype(str)
        e=exist['gvkey'].astype(str) + exist['datacqtr'].astype(str)
        self.niq = self.niq.loc[self.niq['k'].isin(e)]

        print(self.niq.shape)

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
            # exit(0)

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
        # print(self.ibes_df)
        # self.y_type = 'actual'

        consensus_details = []

        for i in tqdm(set(self.ibes_df['datacqtr'])):  # evaluate all testing period

            if (i <= pd.Timestamp(2018, 1, 1, 1)) & (i >= pd.Timestamp(2008, 1, 1, 1)):

                testing_period = self.ibes_df.loc[self.ibes_df['datacqtr'] == i]

                # print('266', self.y_type)
                cut_bins = self.all_bins[self.y_type][i.strftime('%Y-%m-%d')]['cut_bins']

                self.all_bins[self.y_type][i.strftime('%Y-%m-%d')]['test_len_act'] = testing_period[self.y_type].notnull().sum()
                self.all_bins[self.y_type][i.strftime('%Y-%m-%d')]['test_len_est'] = testing_period['medest'].notnull().sum()

                testing_period = testing_period.dropna(how='any')

                # df_print = testing_period.copy()

                # r2[i] = r2_score(testing_period[self.y_type], testing_period['medest'])


                ''' 1. cut using bins from training set'''
                for col in [self.y_type, 'medest','meanest','actual']:    # qcut testing period (actual & consensus)
                    testing_period[col] = pd.cut(testing_period[col], bins=cut_bins, labels=range(self.qcut_q))
                    self.all_bins[self.y_type][i.strftime('%Y-%m-%d')]['test_count_{}'.format(col)] = list(
                        dict(Counter(testing_period[col].to_list())).values())

                result_df = testing_period.filter(['gvkey','datacqtr','medest','meanest','actual'])
                result_df.columns = ['gvkey','datacqtr','medest','meanest','actual_ibes']
                print(result_df)
                consensus_details.append(result_df)

                # if i == pd.Timestamp(2017, 9, 30, 0, 0, 0):
                #     check_print([df_print, testing_period], sort=False)

                ''' 2. evaluation for medest and meanest -> accuracy score...'''
                med_records[i.strftime('%Y-%m-%d')] = eval(testing_period['actual'], testing_period['medest'])
                mean_records[i.strftime('%Y-%m-%d')] = eval(testing_period['actual'], testing_period['meanest'])


        # print(pd.DataFrame.from_records(r2, index=[0]).transpose())
        df_full = pd.concat([self.dict_to_df(med_records, 'medest'), self.dict_to_df(mean_records, 'meanest')], axis=0)
        consensus_details_df = pd.concat(consensus_details, axis=0)

        pd.DataFrame(self.all_bins[self.y_type]).transpose().to_csv('cutbins_{}{}_ibes_test_act_est.csv'.format(self.y_type, self.qcut_q))

        return df_full, consensus_details_df

def main():
    try:
        ann = pd.read_csv('ibes_ANN_last.csv')  # for annual forecast
        qtr = pd.read_csv('ibes_QTR_last.csv')  # for quarterly forecast
        print('local version run - ann_last, qtr_last')
    except:
        ann, qtr = filter_date()

    q = 9


    # convert QTR estimation to qoq and evaluate
    qtr = convert(qtr).qoq()
    print(qtr.describe(), qtr.shape, qtr.columns)
    qtr.filter(['gvkey','datacqtr','medest','meanest','actual']).dropna(how='any').to_csv('consensus_qtr.csv', index=False)

    ann = convert(ann).yoy()
    print(ann.describe(), ann.shape,ann.columns)
    ann.filter(['gvkey','datacqtr','medest','meanest','actual']).dropna(how='any').to_csv('consensus_ann.csv', index=False)
    exit(0)

    df_full, consensus_details_df = evaluate(ibes_df=qtr, y_type='qoq', q=q).eval_all()
    # df_full.sort_index().to_csv('consensus_qoq{}_ibes.csv'.format(q))
    consensus_details_df['qcut'] = q
    consensus_details_df['y_type'] = 'qoq'
    print(consensus_details_df.shape, consensus_details_df)
    consensus_details_df.to_csv('consensus_detail_qoq{}_ibes.csv'.format(q), index=False)


    # convert ANN estimation to yoy and evaluate
    ann = convert(ann).yoy()
    df_full_ann, consensus_details_df = evaluate(ibes_df=ann, y_type='yoyr', q=q).eval_all()
    # df_full_ann.sort_index().to_csv('consensus_yoyr{}_ibes.csv'.format(q))
    consensus_details_df['qcut'] = q
    consensus_details_df['y_type'] = 'yoyr'
    print(consensus_details_df.shape, consensus_details_df)
    consensus_details_df.to_csv('consensus_detail_yoyr{}_ibes.csv'.format(q), index=False)

if __name__ == '__main__':
    os.chdir('/Users/Clair/PycharmProjects/HKP_ML_DL/Preprocessing/raw/ibes/ibes_new')

    # prepare_act()

    # prepare()
    # filter_date()
    main()
