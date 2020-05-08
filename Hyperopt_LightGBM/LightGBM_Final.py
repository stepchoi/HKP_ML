import argparse
import datetime as dt

import lightgbm as lgb
import numpy as np
import pandas as pd
from LoadData import (load_data, sample_from_datacqtr)
from dateutil.relativedelta import relativedelta
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, r2_score, fbeta_score, precision_score, recall_score, \
    accuracy_score, cohen_kappa_score, hamming_loss, jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, Boolean, TIMESTAMP, TEXT, BIGINT, NUMERIC
from tqdm import tqdm

# define parser use for server running
parser = argparse.ArgumentParser()
parser.add_argument('--bins', type=int, default=3)
parser.add_argument('--sample_no', type=int, default=40)
parser.add_argument('--lag', type=int, default=5)
parser.add_argument('--sql_version', default=False, action='store_true')
parser.add_argument('--resume', default=False, action='store_true')
parser.add_argument('--add_ibes', default=False, action='store_true') # CHANGE FOR DEBUG
parser.add_argument('--non_gaap', default=False, action='store_true') # CHANGE FOR DEBUG
parser.add_argument('--y_type', default='qoq')
args = parser.parse_args()

space = {
    # better accuracy
    'learning_rate': hp.choice('learning_rate', np.arange(0.6, 1.0, 0.05, dtype='d')),
    # 'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']), # CHANGE FOR IBES
    'max_bin': hp.choice('max_bin', [127, 255]),
    'num_leaves': hp.choice('num_leaves', np.arange(50, 200, 30, dtype=int)),

    # avoid overfit
    'min_data_in_leaf': hp.choice('min_data_in_leaf', np.arange(500, 1400, 300, dtype=int)),
    'feature_fraction': hp.choice('feature_fraction', np.arange(0.3, 0.8, 0.1, dtype='d')),
    'bagging_fraction': hp.choice('bagging_fraction', np.arange(0.4, 0.8, 0.1, dtype='d')),
    'bagging_freq': hp.choice('bagging_freq', [2, 4, 8]),
    'min_gain_to_split': hp.choice('min_gain_to_split', np.arange(0.5, 0.72, 0.02, dtype='d')),
    'lambda_l1': hp.choice('lambda_l1', np.arange(1, 20, 5, dtype=int)),
    'lambda_l2': hp.choice('lambda_l2', np.arange(350, 450, 20, dtype=int)),

    # parameters won't change
    'boosting_type': 'gbdt',  # past:  hp.choice('boosting_type', ['gbdt', 'dart']
    'objective': 'multiclass',
    'num_class': 3,
    'verbose': -1,
    'metric': 'multi_error',
    'num_threads': 16  # for the best speed, set this to the number of real CPU cores
}

def myPCA(n_components, train_x, test_x):
    ''' PCA for given n_components on train_x, test_x'''

    pca = PCA(n_components=n_components)  # Threshold for dimension reduction, float or integer
    pca.fit(train_x)
    new_train_x = pca.transform(train_x)
    new_test_x = pca.transform(test_x)
    sql_result['pca_components'] = new_train_x.shape[1]
    print('train_x size after PCA {}: '.format(n_components), train_x.shape)

    return new_train_x, new_test_x

class convert_main:
    ''' split train, valid, test from main dataframe
        for given testing_period, y_type, valid_method, valid_no '''

    def __init__(self, main, y_type, testing_period):
        self.testing_period = testing_period

        # 1. define end, start of training period
        end = testing_period
        start = testing_period - relativedelta(years=20)

        # 2. split [gvkey, datacqtr] columns in given the training period for later chronological split of valid set
        label_df = main.iloc[:, :2]
        self.label_df = label_df.loc[(start <= label_df['datacqtr']) & (label_df['datacqtr'] < end)].reset_index(drop=True)
        self.label_df_test = label_df.loc[label_df['datacqtr']==end].reset_index(drop=True)

        # 3. extract array for X, Y for given y_type, testing_period
        X_train_valid, X_test, self.Y_train_valid, self.Y_test = sample_from_datacqtr(main, y_type=y_type,
                                                                                      testing_period=testing_period,
                                                                                      q=space['num_class'])
        sql_result.update({'train_valid_length': len(X_train_valid)})

        # 4. use PCA on X arrays
        self.X_train_valid_PCA, self.X_test_PCA = myPCA(n_components=sql_result['reduced_dimension'],
                                                        train_x=X_train_valid, test_x=X_test)
        print('x_after_PCA shape(train, test): ', self.X_train_valid_PCA.shape, self.X_test_PCA.shape)

        # args.add_ibes = True  # CHANGE FOR DEBUG
        if args.add_ibes == True:
            print(' ------------------------------ add_ibes ------------------------------ ')
            self.add_ibes_func()
            print('x_with ibes shape(train, test): ', self.X_train_valid_PCA.shape, self.X_test_PCA.shape)

    def add_ibes_func(self):  # arr_x_train, arr_x_test

        # 1. read ibes from csv/sql
        if y_type == 'yoyr':
            try:
                ibes = pd.read_csv(
                    '/Users/Clair/PycharmProjects/HKP_ML_DL/Preprocessing/raw/ibes/ibes_new/consensus_ann.csv')
            except:
                ibes = pd.read_sql('SELECT * FROM consensus_ann', engine)
        elif y_type == 'qoq':
            try:
                ibes = pd.read_csv(
                    '/Users/Clair/PycharmProjects/HKP_ML_DL/Preprocessing/raw/ibes/ibes_new/consensus_qtr.csv')
            except:
                ibes = pd.read_sql('SELECT * FROM consensus_qtr', engine)

        # 1.1. change datacqtr to datetime
        ibes['datacqtr'] = pd.to_datetime(ibes['datacqtr'], format='%Y-%m-%d')

        # 1.2. select used datacqtr samples
        ibes_train = pd.merge(ibes, self.label_df, on=['gvkey', 'datacqtr'], how='inner')
        ibes_test = pd.merge(ibes, self.label_df_test, on=['gvkey', 'datacqtr'], how='inner')
        label_df_ibes = ibes_train.iloc[:,:2]
        label_df_test_ibes = ibes_test.iloc[:,:2]

        # 2. standardize + qcut ibes dataframes
        scaler = StandardScaler().fit(ibes_train.iloc[:, 2:4])
        ibes_train_x = scaler.transform(ibes_train.iloc[:, 2:4])
        ibes_test_x = scaler.transform(ibes_test.iloc[:, 2:4])
        print(ibes_test['actual'])

        y_train, cut_bins = pd.qcut(ibes_train['actual'], q=qcut_q, labels=range(qcut_q), retbins=True)
        y_test = pd.cut(ibes_test['actual'], bins=cut_bins, labels=range(qcut_q))  # can work without test set

        from collections import Counter # CHANGE FOR DEBUG
        print(Counter(y_train))

        # 4. merge ibes with after pca array
        # 4.1. label ibes
        ibes_train.iloc[:,2:4] = ibes_train_x
        ibes_train.iloc[:, 4] = y_train
        ibes_test.iloc[:,2:4] = ibes_test_x
        ibes_test.iloc[:, 4] = y_test
        print('4: ', ibes_train.shape, ibes_train)

        # 4.2. label original X after PCA
        x_train = pd.concat([self.label_df, pd.DataFrame(self.X_train_valid_PCA)], axis=1)
        x_test = pd.concat([self.label_df_test, pd.DataFrame(self.X_test_PCA)], axis=1)

        # 4.3. add ibes to X
        self.X_train_valid_PCA = pd.merge(x_train, ibes_train[['gvkey', 'datacqtr','medest','meanest']], on=['gvkey', 'datacqtr'], how='inner').iloc[:, 2:].to_numpy()
        self.X_test_PCA = pd.merge(x_test, ibes_test[['gvkey', 'datacqtr','medest','meanest']], on=['gvkey', 'datacqtr'], how='inner').iloc[:, 2:].to_numpy()
        # print('x_train, x_test after add ibes: ', self.X_train_valid_PCA.shape, self.X_test_PCA.shape)
        # print(x_train.describe())

        print('x_train shape: ', self.X_train_valid_PCA.shape, x_train.columns.to_list())

        # 4.4. label original Y
        y_train = pd.concat([self.label_df, pd.DataFrame(self.Y_train_valid)], axis=1)
        y_test = pd.concat([self.label_df_test, pd.DataFrame(self.Y_test)], axis=1)
        # print(y_train)

        # 4.5. label original Y
        # args.non_gaap = True # CHANGE FOR DEBUG
        if args.non_gaap == True:
            print(' ------------------------------ non_gaap ------------------------------ ')
            self.Y_train_valid = pd.merge(y_train, ibes_train[['gvkey', 'datacqtr', 'actual']], on=['gvkey', 'datacqtr'], how='inner').iloc[:, 2]
            self.Y_test = pd.merge(y_test, ibes_test[['gvkey', 'datacqtr', 'actual']], on=['gvkey', 'datacqtr'], how='inner').iloc[:, 2]
            print(self.Y_train_valid)
            print('non_gaap')
        else:
            self.Y_train_valid = pd.merge(y_train, label_df_ibes, on=['gvkey', 'datacqtr'], how='inner').iloc[:, 2]
            self.Y_test = pd.merge(y_test, label_df_test_ibes, on=['gvkey', 'datacqtr'], how='inner').iloc[:, 2]

        print('y_train, y_test after add ibes: ', len(self.Y_train_valid), len(self.Y_test), type(self.Y_train_valid))


    def split_chron(self, df, valid_no):  # chron split of valid set
        date_df = pd.concat([self.label_df, pd.DataFrame(df)], axis=1)
        valid_period = self.testing_period - valid_no * relativedelta(months=3)
        train = date_df.loc[(date_df['datacqtr'] < valid_period), date_df.columns[2:]].values
        valid = date_df.loc[(date_df['datacqtr'] >= valid_period), date_df.columns[2:]].values
        return train, valid

    def split_valid(self, valid_method, valid_no):  # split validation set from training set

        if valid_method == 'shuffle':  # split validation set by random shuffle
            test_size = valid_no / 80
            X_train, X_valid, Y_train, Y_valid = train_test_split(self.X_train_valid_PCA, self.Y_train_valid,
                                                                  test_size=test_size)

        elif valid_method == 'chron':  # split validation set by chron
            X_train, X_valid = self.split_chron(self.X_train_valid_PCA, valid_no)
            Y_train, Y_valid = self.split_chron(self.Y_train_valid, valid_no)
            Y_train = np.reshape(Y_train, -1)  # transpose array for y
            Y_valid = np.reshape(Y_valid, -1)

        return X_train, X_valid, self.X_test_PCA, Y_train, Y_valid, self.Y_test

def myLightGBM(space, valid_method, valid_no):
    ''' X_train, X_valid, X_test, Y_train, Y_valid
    -> lightgbm for with params from hyperopt
    -> predict Y_pred using trained gbm model
    '''

    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = converted_main.split_valid(valid_method, valid_no)


    params = space.copy()

    '''Training'''
    lgb_train = lgb.Dataset(X_train, label=Y_train, free_raw_data=False)
    lgb_eval = lgb.Dataset(X_valid, label=Y_valid, reference=lgb_train, free_raw_data=False)

    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=lgb_eval,
                    num_boost_round=1000, # CHANGE FOR DEBUG
                    early_stopping_rounds=150,
                    )

    '''Evaluation on Test Set'''
    Y_train_pred_softmax = gbm.predict(X_train, num_iteration=gbm.best_iteration)
    Y_train_pred = [list(i).index(max(i)) for i in Y_train_pred_softmax]
    Y_valid_pred_softmax = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
    Y_valid_pred = [list(i).index(max(i)) for i in Y_valid_pred_softmax]
    Y_test_pred_softmax = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    Y_test_pred = [list(i).index(max(i)) for i in Y_test_pred_softmax]

    return Y_train, Y_valid, Y_test, Y_train_pred, Y_valid_pred, Y_test_pred, gbm, X_train

def f(space):
    ''' train & evaluate LightGBM on given space by hyperopt trails '''

    Y_train, Y_valid, Y_test, Y_train_pred, Y_valid_pred, Y_test_pred, gbm, X_train = myLightGBM(space, sql_result['valid_method'],
                                                                                   sql_result['valid_no'])

    result = {'loss': 1 - accuracy_score(Y_valid, Y_valid_pred),
              'accuracy_score_train': accuracy_score(Y_train, Y_train_pred),
              'accuracy_score_valid': accuracy_score(Y_valid, Y_valid_pred),
              'accuracy_score_test': accuracy_score(Y_test, Y_test_pred),
              'precision_score_test': precision_score(Y_test, Y_test_pred, average='micro'),
              'recall_score_test': recall_score(Y_test, Y_test_pred, average='micro'),
              'f1_score_test': f1_score(Y_test, Y_test_pred, average='micro'),
              'f0.5_score_test': fbeta_score(Y_test, Y_test_pred, beta=0.5, average='micro'),
              'f2_score_test': fbeta_score(Y_test, Y_test_pred, beta=2, average='micro'),
              'r2_score_test': r2_score(Y_test, Y_test_pred),
              "cohen_kappa_score": cohen_kappa_score(Y_test, Y_test_pred, labels=None),
              "hamming_loss": hamming_loss(Y_test, Y_test_pred),
              "jaccard_score": jaccard_score(Y_test, Y_test_pred, labels=None, average='macro'),
              'status': STATUS_OK}

    # f = plt.figure()
    # shap_values = shap.TreeExplainer(gbm).shap_values(X_train)
    # shap.summary_plot(shap_values, X_train)
    # file_name = '{}{}{}'.format(sql_result['testing_period'], sql_result['y_type'], sql_result['qcut'])
    # gbm.save_model('model{}.txt'.format(file_name))
    # f.savefig("summary_plot_{}.png".format(file_name), bbox_inches='tight', dpi=300)
    # best_model_acc = result['accuracy_score_test']

    sql_result.update(space)
    sql_result.update(result)
    sql_result.pop('is_unbalance')
    sql_result['finish_timing'] = dt.datetime.now()

    pt = pd.DataFrame.from_records([sql_result], index=[0])

    pt = pt.astype(str)
    print('sql_result_before writing: ', sql_result)

    pt.to_sql('lightgbm_results_ibes', con=engine, index=False, if_exists='append', dtype=types)

    return result

def HPOT(space, max_evals):
    ''' use hyperopt on each set '''
    trials = Trials()
    best = fmin(fn=f, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    print(best)

if __name__ == "__main__":

    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)
    sql_result = {}

    db_last = pd.read_sql("SELECT * FROM lightgbm_results WHERE y_type='{}' "
                          "order by finish_timing desc LIMIT 1".format(args.y_type), engine)  # identify current # trials from past execution
    db_last_klass = db_last[['y_type', 'valid_method',
                             'valid_no', 'testing_period', 'reduced_dimension']].to_dict('records')[0]
    print(args)

    # define columns types for db
    types = {'non_gaap': Boolean(), 'add_ibes': Boolean(), 'trial': BIGINT(), 'qcut': BIGINT(), 'name': TEXT(), 'max_evals': BIGINT(), 'y_type': TEXT(), 'valid_method': TEXT(), 'valid_no': BIGINT(), 'testing_period': TIMESTAMP(), 'train_valid_length': BIGINT(), 'pca_components': BIGINT(), 'bagging_fraction': NUMERIC(), 'bagging_freq': BIGINT(), 'boosting_type': TEXT(), 'early_stopping_rounds': BIGINT(), 'feature_fraction': NUMERIC(), 'lambda_l1': BIGINT(), 'lambda_l2': BIGINT(), 'learning_rate': NUMERIC(), 'max_bin': BIGINT(), 'metric': TEXT(), 'min_data_in_leaf': BIGINT(), 'min_gain_to_split': NUMERIC(), 'num_boost_round': BIGINT(), 'num_class': BIGINT(), 'num_leaves': BIGINT(), 'num_threads': BIGINT(), 'objective': TEXT(), 'reduced_dimension': NUMERIC(), 'verbose': BIGINT(), 'loss': NUMERIC(), 'accuracy_score_train': NUMERIC(), 'accuracy_score_valid': NUMERIC(), 'accuracy_score_test': NUMERIC(), 'precision_score_test': NUMERIC(), 'recall_score_test': NUMERIC(), 'f1_score_test': NUMERIC(), 'f0.5_score_test': NUMERIC(), 'f2_score_test': NUMERIC(), 'r2_score_test': NUMERIC(), 'cohen_kappa_score': NUMERIC(), 'hamming_loss': NUMERIC(), 'jaccard_score': NUMERIC(), 'status': TEXT(), 'finish_timing': TIMESTAMP()}

    # parser
    qcut_q = int(args.bins)
    y_type = args.y_type  # 'yoyr','qoq','yoy'
    resume = args.resume
    sample_no = args.sample_no

    # load data for entire period
    main = load_data(lag_year=args.lag, sql_version=args.sql_version)  # CHANGE FOR DEBUG
    label_df = main.iloc[:,:2]

    space['num_class'] = qcut_q
    space['is_unbalance'] = True

    sql_result = {'qcut': qcut_q, 'non_gaap': args.non_gaap, 'add_ibes': args.add_ibes}
    # sql_result['name'] = 'try add ibes as X'
    # sql_result['trial'] = db_last['trial'] + 1

    # roll over each round
    period_1 = dt.datetime(2017, 12, 31)  # CHANGE FOR DEBUG

    for i in tqdm(range(sample_no)):  # divide sets and return
        testing_period = period_1 + i * relativedelta(months=3)  # set sets in chronological order

        # PCA dimension

        for max_evals in [30]:  # 40, 50  # CHANGE FOR DEBUG

            for reduced_dimension in [0.75]:  # 0.66, 0.7
                sql_result['reduced_dimension'] = reduced_dimension

                for valid_method in ['shuffle','chron']:  # 'chron'

                    for valid_no in [1, 5]:  # 1,5

                        klass = {'y_type': y_type,
                                 'valid_method': valid_method,
                                 'valid_no': valid_no,
                                 'testing_period': testing_period,
                                 'reduced_dimension': reduced_dimension}

                        if db_last_klass == klass:
                            resume = False
                            print('resume from params', klass)
                        elif resume == False:
                            print(resume)
                        else:
                            continue

                        print(klass, type(klass))

                        sql_result.update({'max_evals': max_evals})
                        sql_result.update(klass)

                        converted_main = convert_main(main, y_type, testing_period)

                        HPOT(space, max_evals=max_evals)

    # print('x shape before PCA:', x.shape)
