import datetime as dt
import os

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
        df = df.filter(['gvkey','fpedats', 'medest','meanest'])
        df.columns = ['gvkey','datacqtr', 'medest','meanest']

        self.num_col = ['medest', 'meanest']
        df[self.num_col] = df[self.num_col].mask(df[self.num_col] < 0, float(1e-6))
        self.df = df
        self.pmax = {'qoq': 2.912091, 'yoy': 3.894995, 'yoy_rolling': 1.392811}

    def drop_nonseq(self, df, interval):
        '''drop non-sequential records'''

        df['datacqtr'] = pd.to_datetime(df['datacqtr'])
        if interval == 'q':
            i = relativedelta(months=3)
        elif interval == 'y':
            i = relativedelta(months=12)
        drop_idx = df.loc[df['datacqtr'] != df['datacqtr'].shift(1).apply(lambda x: x + relativedelta(days=1)
                                                                    + i - relativedelta(days=1))].index.to_list()
        first_idx = df.groupby('gvkey').head().index.to_list()
        drop_idx = list(set(drop_idx) - set(first_idx))
        df.iloc[drop_idx, -2:] = np.nan
        self.drop_idx = drop_idx
        return df

    def qoq(self):
        self.df = self.drop_nonseq(self.df,'q')
        self.df[self.num_col] = self.df.groupby('gvkey').apply(lambda x: x[self.num_col].shift(-1).div(x[self.num_col]).sub(1))
        self.df[self.num_col] = self.df[self.num_col].mask(self.df[self.num_col] > self.pmax['qoq'], self.pmax['qoq'])
        return self.df

    def yoy(self):
        self.df = self.drop_nonseq(self.df,'y')
        self.df[self.num_col] = self.df.groupby('gvkey').apply(lambda x: x[self.num_col].shift(-1).div(x[self.num_col]).sub(1))
        self.df[self.num_col] = self.df[self.num_col].mask(self.df[self.num_col] > self.pmax['yoy_rolling'], self.pmax['yoy_rolling'])
        return self.df

def conver_qoq_yoy():
    try:
        ann = pd.read_csv('ibes_ANN_last.csv')  # for annual forecast
        # qtr = pd.read_csv('ibes_QTR_last.csv')  # for quarterly forecast
        print('local version run - ann_last, qtr_last')
    except:
        ann, qtr = filter_date()

    # qtr = convert(qtr).qoq()
    ann = convert(ann).yoy()
    # df_full = evaluate(qtr).eval_all('qoq')
    df_full_ann = evaluate(ann, 'yoy_rolling').eval_all()
    # df_full.to_csv('consensus_qoq.csv')
    df_full_ann.to_csv('consensus_yoyr.csv')



def eval(Y_test, Y_test_pred):
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
    def __init__(self, ibes_df, y_type):
        self.all_bins = self.get_all_bins()
        self.ibes_df = pd.merge(self.niq[['gvkey', 'datacqtr', y_type]], ibes_df, on=['gvkey', 'datacqtr'], how='inner')
        self.ibes_df = self.ibes_df.dropna(how='any')
        self.y_type = y_type

    def qcut_y(self, df, col):  # qcut y with train_y cut_bins
        bins = {}
        period_1 = dt.datetime(2008, 3, 31)
        for i in tqdm(range(40)):  # change to 40 for full 40 sets, change to False to stop saving csv
            end = period_1 + i * relativedelta(months=3)  # define testing period
            start = end - relativedelta(years=20)  # define training period
            train = df.loc[(start <= df['datacqtr']) & (df['datacqtr'] < end)]  # train df = 80 quarters
            train[col], cut_bins = pd.qcut(train[col], q=3, labels=range(3), retbins=True, duplicates='drop')
            bins[end.strftime('%Y-%m-%d')] = cut_bins
        # d = pd.DataFrame.from_dict(bins, orient='index',columns=[0,1,2,3])
        return bins

    def get_all_bins(self):
        self.niq = pd.read_csv('/Users/Clair/PycharmProjects/HKP_ML_DL/Hyperopt_LightGBM/niq_main.csv')
        self.niq['datacqtr'] = pd.to_datetime(self.niq['datacqtr'])
        all_bins = {}
        for y in ['qoq', 'yoy', 'yoy_rolling']:
            all_bins[y] = self.qcut_y(self.niq, y)
        return all_bins

    def dict_to_df(self, dict, measure):
        df = pd.DataFrame()
        for date in dict.keys():
            for score in dict[date].keys():
                df.loc[date, score] = dict[date][score]
        df['type'] = measure
        return df

    def eval_all(self):
        med_records = {}
        mean_records = {}
        for i in tqdm(set(self.ibes_df['datacqtr'])):  # evaluate all testing period
            if (i <= pd.Timestamp(2018, 1, 1, 1)) & (i >= pd.Timestamp(2008, 1, 1, 1)):
                testing_period = self.ibes_df.loc[self.ibes_df['datacqtr'] == i]
                cut_bins = self.all_bins[self.y_type][i.strftime('%Y-%m-%d')]

                for col in [self.y_type, 'medest','meanest']:    # qcut testing period (actual & consensus)
                    testing_period[col] = pd.cut(testing_period[col], bins=cut_bins, labels=range(3))

                testing_period = testing_period.fillna(0)   # some '-1' will be qcut into NaN -> manual replace by 0 category
                med_records[i.strftime('%Y-%m-%d')] = eval(testing_period[self.y_type], testing_period['medest'])
                mean_records[i.strftime('%Y-%m-%d')] = eval(testing_period[self.y_type], testing_period['meanest'])
        df_full = pd.concat([self.dict_to_df(med_records, 'medest'), self.dict_to_df(mean_records, 'meanest')], axis=0)
        return df_full

if __name__ == '__main__':
    os.chdir('/Users/Clair/PycharmProjects/HKP_ML_DL/Preprocessing/raw')

    conver_qoq_yoy()
    # filter_quarters()
    # main()