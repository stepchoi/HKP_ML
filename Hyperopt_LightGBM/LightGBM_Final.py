import datetime as dt
import gc

import lightgbm as lgb
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, r2_score, fbeta_score, precision_score, recall_score, \
    accuracy_score, cohen_kappa_score, hamming_loss, jaccard_score
from sklearn.model_selection import train_test_split
from sqlalchemy import Column, MetaData, Table, Integer, create_engine, DateTime
from tqdm import tqdm

from Preprocessing.LoadData import (load_data, sample_from_datacqtr)

space = {
    # dimension
    'reduced_dimension' : hp.choice('reduced_dimension', np.arange(0.66, 0.75, 0.01)), # past: [508, 624, 757]

    # better accuracy
    'learning_rate': hp.choice('learning_rate', np.arange(0.6, 1.0, 0.05)),
    'boosting_type': 'gbdt', # past:  hp.choice('boosting_type', ['gbdt', 'dart']
    'max_bin': hp.choice('max_bin', [127, 255]),
    'num_leaves': hp.choice('num_leaves', np.arange(50,200,30,dtype=int)),

    # avoid overfit
    'min_data_in_leaf': hp.choice('min_data_in_leaf', np.arange(500, 1400, 300, dtype=int)),
    'feature_fraction': hp.choice('feature_fraction', np.arange(0.3, 0.8, 0.1)),
    'bagging_fraction': hp.choice('bagging_fraction', np.arange(0.4, 0.8, 0.1)),
    'bagging_freq': hp.choice('bagging_freq', [2,4,8]),
    'min_gain_to_split': hp.choice('min_gain_to_split', np.arange(0.5, 0.72, 0.02)),
    'lambda_l1': hp.choice('lambda_l1', np.arange(1, 20, 5)),
    'lambda_l2': hp.choice('lambda_l2', np.arange(350, 450, 20)),

    # Voting Parallel
    # 'tree_learner': 'voting'
    # 'top_k': 2
    # multi_error_top_k

    # parameters won't change
    'objective': 'multiclass',
    'num_class': 3,
    'verbose': -1,
    'metric': 'multi_error',
    'num_boost_round': 1000,
    'num_threads': 12  # for the best speed, set this to the number of real CPU cores
    }

def myPCA(n_components, train_x, test_x):

    ''' PCA for given n_components on train_x, test_x'''

    pca = PCA(n_components=n_components)  # Threshold for dimension reduction, float or integer
    pca.fit(train_x)
    new_train_x = pca.transform(train_x)
    new_test_x = pca.transform(test_x)
    sql_result['pca_components'] = new_train_x.shape[1]
    return new_train_x, new_test_x

class convert_main:

    ''' split train, valid, test from main dataframe
        for given testing_period, y_type, valid_method, valid_no '''

    def __init__(self, main, space, y_type, testing_period):
        self.testing_period=testing_period

        # 1. define end, start of training period
        end = testing_period
        start = testing_period - relativedelta(years=20)

        # 2. split [gvkey, datacqtr] columns in given the training period for later chronological split of valid set
        label_df = main.iloc[:,:2]
        self.label_df = label_df.loc[(start <= label_df['datacqtr']) & (label_df['datacqtr'] < end)].reset_index(drop=True)

        # 3. extract array for X, Y for given y_type, testing_period
        X_train_valid, X_test, self.Y_train_valid, self.Y_test = sample_from_datacqtr(main, y_type=y_type,
                                                                            testing_period=testing_period, q=3)
        sql_result.update({'train_valid_length': len(X_train_valid)})

        # 4. use PCA on X arrays
        self.X_train_valid_PCA, self.X_test_PCA = myPCA(n_components=space['reduced_dimension'],
                                                        train_x=X_train_valid, test_x=X_test)

    def split_chron(self, df, valid_no): # chron split of valid set
        date_df = pd.concat([self.label_df, pd.DataFrame(df)], axis=1)
        valid_period = self.testing_period - valid_no * relativedelta(months=3)
        train = date_df.loc[(date_df['datacqtr'] < valid_period), date_df.columns[2:]].values
        valid = date_df.loc[(date_df['datacqtr'] >= valid_period), date_df.columns[2:]].values
        return train, valid

    def split_valid(self, valid_method, valid_no):  # split validation set from training set

        if valid_method == 'shuffle':   # split validation set by random shuffle
            test_size = valid_no / 80
            X_train, X_valid, Y_train, Y_valid = train_test_split(self.X_train_valid_PCA, self.Y_train_valid,
                                                                  test_size=test_size, random_state=666)

        elif valid_method == 'chron':   # split validation set by chron
            X_train, X_valid = self.split_chron(self.X_train_valid_PCA, valid_no)
            Y_train, Y_valid = self.split_chron(self.Y_train_valid, valid_no)
            Y_train = np.reshape(Y_train, -1)   # transpose array for y
            Y_valid = np.reshape(Y_valid, -1)

        return X_train, X_valid, self.X_test_PCA, Y_train, Y_valid, self.Y_test

def myLightGBM(main, space, y_type, testing_period, valid_method, valid_no):

    ''' X_train, X_valid, X_test, Y_train, Y_valid
    -> lightgbm for with params from hyperopt
    -> predict Y_pred using trained gbm model
    '''

    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = convert_main(main, space, y_type,
                                                                      testing_period).split_valid(valid_method, valid_no)

    params = space.copy()
    params.pop('reduced_dimension')

    '''Training'''
    lgb_train = lgb.Dataset(X_train, label=Y_train, free_raw_data=False)
    lgb_eval = lgb.Dataset(X_valid, label=Y_valid, reference=lgb_train, free_raw_data=False)

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=lgb_eval,  # eval training data
                    early_stopping_rounds=150)


    '''print and save feature importance for model'''
    def feature_importance():
        importance = gbm.feature_importance(importance_type='split')
        name = gbm.feature_name()
        feature_importance = pd.DataFrame({'feature_name': name, 'importance': importance})
        print(feature_importance)
        feature_importance.to_csv('feature_importance.csv', index=False)


    '''Evaluation on Test Set'''

    Y_train_pred_softmax = gbm.predict(X_train, num_iteration=gbm.best_iteration)
    Y_train_pred = [list(i).index(max(i)) for i in Y_train_pred_softmax]
    Y_valid_pred_softmax = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
    Y_valid_pred = [list(i).index(max(i)) for i in Y_valid_pred_softmax]
    Y_test_pred_softmax = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    Y_test_pred = [list(i).index(max(i)) for i in Y_test_pred_softmax]

    return Y_train, Y_valid, Y_test, Y_train_pred, Y_valid_pred, Y_test_pred

def f(space):

    ''' train & evaluate LightGBM on given space by hyperopt trails '''

    Y_train, Y_valid, Y_test, Y_train_pred, Y_valid_pred, Y_test_pred = myLightGBM(main=main, space=space, **klass)

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

    sql_result.update(space)
    sql_result.update(result)
    sql_result['finish_timing'] = dt.datetime.now()

    pd.DataFrame.from_records([sql_result],index='trial').to_sql('lightgbm_results', con=engine, if_exists='append')
    sql_result['trial']+=1

    return result

def HPOT(space, sample_no, max_evals):

    # roll over each round
    period_1 = dt.datetime(2008, 3, 31)

    for i in tqdm(range(sample_no)):    # divide sets and return
        testing_period = period_1 + i * relativedelta(months=3) # set sets in chronological order

        klass['testing_period'] = testing_period
        sql_result.update(klass)

        ''' use hyperopt on each set '''
        trials = Trials()
        best = fmin(fn=f, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        print(best)
        '''??????????'''

if __name__ == "__main__":

    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    main = load_data(lag_year=5, sql_version=False)  # main = entire dataset before standardization/qcut

    sample_no = 40
    qcut_q = 3

    sql_result = {'qcut': qcut_q}
    sql_result['name']='only yoyr'
    # space['is_unbalance'] = True

    if qcut_q > 3:  # for 6, 9 qcut test
        space['num_class'] = qcut_q

    for y_type in ['yoyr']: # 'qoq','yoy'
        for max_evals in [10, 20, 30, 40, 50]:
            for valid_method in ['chron', 'shuffle']:
                for valid_no in [1,5,10]:

                    klass = {'y_type': y_type,
                             'valid_method': valid_method,
                             'valid_no': valid_no}
                    sql_result['trial']=0
                    sql_result.update({'max_evals':max_evals})

                    HPOT(space, sample_no=sample_no, max_evals=max_evals)

    # print('x shape before PCA:', x.shape)
