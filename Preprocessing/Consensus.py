import datetime as dt
import os

import pandas as pd
from LoadData import convert_to_float32
from PrepareDatabase import drop_nonseq
from dateutil.relativedelta import relativedelta
from sklearn.metrics import f1_score, r2_score, fbeta_score, precision_score, recall_score, \
    accuracy_score, cohen_kappa_score, hamming_loss, jaccard_score
from tqdm import tqdm


def prepare():
    '''map gvkey, datacqtr to consensus data'''

    col = ['m_ticker', 'ticker', 'comp_name', 'exchange', 'currency_code', 'per_end_date', 'per_type',
           'per_fisc_year', 'per_fisc_qtr', 'per_cal_year', 'per_cal_qtr', 'obs_date', 'eps_mean_est',
           'eps_median_est', 'eps_cnt_est', 'eps_high_est', 'eps_low_est', 'eps_std_dev_est', 'eps_cnt_est_rev_up',
           'eps_cnt_est_rev_down']

    # identify companies(gvkey) used in our model
    gvkey_1 = pd.read_csv('gvkey_sic',header=None)[0].to_list()

    # use linktable of (mticker, ticker -> cik) & (cik -> gvkey) to map (mticker, ticker -> gvkey)
    zacklink = pd.read_excel('ZACKS_Company List.xlsx', sheet_name = 'sheet1',usecols= ['m_ticker','ticker','comp_cik'])
    ciqlink = pd.read_excel('ciqlink.xlsx', sheet_name = 'ciqlink', usecols=['GVKEY','ENDDATE','CIK'])

    print(ciqlink.shape)

    ciqlink['year'] = pd.to_datetime(ciqlink['ENDDATE']).dt.strftime('%Y').astype(int)
    ciqlink = ciqlink.loc[ciqlink['year']>=2013]
    print(ciqlink.shape)
    print(ciqlink)

    id = pd.merge(zacklink, ciqlink, left_on='comp_cik',right_on='CIK',how='inner')

    id = id.loc[id['GVKEY'].isin(gvkey_1)] # filter gvkey used in our model
    id = id.drop_duplicates(subset=['GVKEY'], keep='first')
    id.to_csv('ticker_cik_gvkey.csv', index=False)

    # gvkey_2 = set(id['GVKEY'])
    # # print([x for x in gvkey_1 if x not in gvkey_2])

    # read consensus table
    consensus = pd.read_csv('postgres_public_zz_zacks_import.csv', header=None)
    consensus.columns = col
    consensus = consensus.filter(['m_ticker', 'ticker', 'per_end_date', 'per_type', 'obs_date', 'eps_mean_est'])
    consensus = pd.merge(id, consensus, on=['m_ticker','ticker'],how='inner') # map gvkey to each observation

    consensus = consensus.loc[consensus['per_type']=='Q'] # we only use quarter eps estimation corresponding with niq(Y)

    consensus = consensus.filter(['GVKEY','per_end_date','obs_date','eps_mean_est'])   # filter & rename columns
    consensus.columns = ['gvkey', 'datacqtr','obs_date','eps_mean_est']

    consensus = consensus.sort_values(by=['gvkey','datacqtr','obs_date']) # sorting

    du = consensus[consensus.duplicated(subset=['gvkey','datacqtr','obs_date'])]
    print(du)

    convert_to_float32(consensus)

    print(consensus.info())
    consensus.to_csv('consensus_raw.csv', index=False)

def filter_quarters():
    '''filter the last observation as concensus input'''

    raw = pd.read_csv('consensus_raw.csv')
    raw['datacqtr'] = pd.to_datetime(raw['datacqtr'])
    raw['obs_date'] = pd.to_datetime(raw['obs_date'])

    raw['last_obs_qoq'] = raw['datacqtr'].apply(lambda x: x - relativedelta(months=3))
    raw['last_obs_yoy'] = raw['datacqtr'].apply(lambda x: x - relativedelta(months=12))

    qoq_obs = raw.loc[raw['obs_date']<=raw['last_obs_qoq']].groupby(['gvkey','datacqtr']).last().reset_index(drop=False)
    qoq_obs.to_csv('qoq_obs.csv')
    print(qoq_obs.shape)

    yoy_obs = raw.loc[raw['obs_date'] <= raw['last_obs_yoy']].groupby(['gvkey','datacqtr']).last().reset_index(drop=False)
    yoy_obs.to_csv('yoy_obs.csv')
    print(yoy_obs.shape)

def qoq(df):
    '''convert to qoq format'''

    df = df.reset_index(drop=True)
    df['datacqtr'] = pd.to_datetime(df['datacqtr'])

    df = drop_nonseq(df)
    df['eps_mean_est'] = df['eps_mean_est'].mask(df['eps_mean_est']<0,float(1e-6))

    # convert to qoq, yoy
    df['next1_abs'] = df.groupby('gvkey').apply(lambda x: x['eps_mean_est'].shift(-1)).to_list()
    df['qoq'] = df['next1_abs'].div(df['eps_mean_est']).sub(1)  # T1/T0

    df = df.dropna(how='any')

    df = df.filter(['gvkey', 'datacqtr', 'qoq'])

    pmax = {'qoq':2.912091,
            'yoy':3.894995,
            'yoy_rolling':1.392811}

    df['qoq'] = df['qoq'].mask(df['qoq'] > pmax['qoq'], pmax['qoq'])
    df.columns = ['gvkey','datacqtr','qoq_est']

    return df

def yoy(df):
    '''convert to yoy, rolling yoy format'''

    df = df.reset_index(drop=True)
    df['datacqtr'] = pd.to_datetime(df['datacqtr'])

    df = drop_nonseq(df)
    df['eps_mean_est'] = df['eps_mean_est'].mask(df['eps_mean_est']<0,float(1e-6))

    df['next4'] = df.groupby('gvkey').apply(lambda x: x['eps_mean_est'].shift(-4)).to_list()
    df['yoy'] = df['next4'].div(df['eps_mean_est']).sub(1)  # T4/T0

    df['past4_abs'] = df.groupby('gvkey').apply(
        lambda x: x['eps_mean_est'].rolling(4, min_periods=4).sum()).to_list()  # rolling past 4 quarter
    df['next4_abs'] = df.groupby('gvkey').apply(lambda x: x['past4_abs'].shift(-4)).to_list()  # rolling next 4 quarter

    df['yoy_rolling'] = df['next4_abs'].div(df['past4_abs']).sub(1)  # T4/T0

    df = df.filter(['gvkey', 'datacqtr', 'yoy', 'yoy_rolling'])
    df = df.dropna(how='any')


    pmax = {'qoq':2.912093,
            'yoy':3.894996,
            'yoy_rolling':1.392812}
    num_list = ['yoy', 'yoy_rolling']
    for n in num_list:
        df[n] = df[n].mask(df[n] > pmax[n], pmax[n])
    df.columns = ['gvkey','datacqtr','yoy_est', 'yoy_rolling_est']

    return df

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

def qcut_y(df, col): # qcut y with train_y cut_bins
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

def main():
    '''define qcut for each y_type, testing set'''
    niq = pd.read_csv('niq_main.csv')
    niq['datacqtr'] = pd.to_datetime(niq['datacqtr'])
    all_bins = {}
    for y in ['qoq','yoy','yoy_rolling']:
        all_bins[y] = qcut_y(niq,y)

    '''evaluate qoq'''
    qoq_obs = pd.read_csv('qoq_obs.csv', index_col='Unnamed: 0')
    qoq_obs = qoq(qoq_obs)
    qoq_obs = pd.merge(niq[['gvkey','datacqtr','qoq']], qoq_obs, on=['gvkey','datacqtr'], how='inner')

    records = {}
    for i in tqdm(set(qoq_obs['datacqtr'])):  # change to 40 for full 40 sets, change to False to stop saving csv
        if i <= pd.Timestamp(2018, 1, 1, 1):
            qoq_period = qoq_obs.loc[qoq_obs['datacqtr']==i]
            cut_bins = all_bins['qoq'][i.strftime('%Y-%m-%d')]

            for col in ['qoq', 'qoq_est']:
                qoq_period[col] = pd.cut(qoq_period[col], bins=cut_bins, labels=range(3))
            qoq_period = qoq_period.fillna(0)

            records[i.strftime('%Y-%m-%d')] = eval(qoq_period['qoq'], qoq_period['qoq_est'])

    d_list = []
    d = pd.DataFrame()
    for date in records.keys():
        for score in records[date].keys():
            d.loc[date, score] = records[date][score]
    d['type'] = 'qoq'
    d_list.append(d)

    '''evaluate yoy, rolling_yoy'''
    yoy_obs = pd.read_csv('yoy_obs.csv', index_col='Unnamed: 0')
    yoy_obs = yoy(yoy_obs)
    print(yoy_obs.describe())
    yoy_obs = pd.merge(niq[['gvkey','datacqtr','yoy', 'yoy_rolling']], yoy_obs, on=['gvkey','datacqtr'], how='inner')

    for y_type in ['yoy', 'yoy_rolling']:
        records = {}
        for i in tqdm(set(yoy_obs['datacqtr'])):  # change to 40 for full 40 sets, change to False to stop saving csv
            if i <= pd.Timestamp(2018, 1, 1, 1):
                qoq_period = yoy_obs.loc[yoy_obs['datacqtr']==i]
                cut_bins = all_bins['qoq'][i.strftime('%Y-%m-%d')]

                for col in [y_type, y_type+'_est']:
                    qoq_period[col] = pd.cut(qoq_period[col], bins=cut_bins, labels=range(3))
                qoq_period = qoq_period.fillna(0)

                records[i.strftime('%Y-%m-%d')] = eval(qoq_period[y_type], qoq_period[y_type+'_est'])

        d = pd.DataFrame()
        for date in records.keys():
            for score in records[date].keys():
                d.loc[date, score] = records[date][score]

        d['type'] = y_type
        d_list.append(d)

    full_df = pd.concat(d_list, axis=0)
    full_df.to_csv('consensus.csv')
    # print(comp_list)
if __name__ == '__main__':
    os.chdir('/Users/Clair/PycharmProjects/HKP_ML_DL/Preprocessing/raw')

    # prepare()
    # filter_quarters()
    main()