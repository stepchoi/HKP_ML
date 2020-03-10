import pandas as pd
import numpy as np

from PCA_for_LightGBM import PCA_fitting, PCA_predict
from Autoencoder_for_LightGBM import AE_fitting, AE_predict

from sklearn.model_selection import train_test_split

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import lightgbm as lgb

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from Preprocessing.LoadData import load_data, sample_from_main


space = {
    # dimension
    'reduced_dimension' : hp.choice('reduced_dimension', [508, 624, 757]),

    # better accuracy
    'learning_rate': hp.choice('learning_rate', [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]),
    'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
    'max_bin': hp.choice('max_bin', [200, 255, 300]),
    'num_leaves': hp.choice('num_leaves', [200, 300, 400]),

    # avoid overfit
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [250, 500, 750]),
    'feature_fraction': hp.choice('feature_fraction', [0.6, 0.8, 0.9]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.6, 0.8, 0.95]),
    'bagging_freq': hp.choice('bagging_freq', [2, 5, 8]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.05, 0.2, 0.4]),
    'lambda_l1': hp.choice('lambda_l1', [0, 0.4, 1]),
    'lambda_l2': hp.choice('lambda_l2', [0, 0.5, 1]),

    # parameters won't change
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_error',
    'num_threads': 2  # for the best speed, set this to the number of real CPU cores
    }

x =
y =

def Dimension_reduction(reduced_dimensions, method='PCA'):

    x_lgbm, x_test, y_lgbm, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
    x_train, x_valid, y_train, y_valid = train_test_split(x_lgbm, y_lgbm, test_size=0.25, stratify=y_lgbm)

    if (method == 'AE'):
        AE_model = AE_fitting(x_train, reduced_dimensions)
        compressed_x_train = AE_predict(x_train, AE_model)
        compressed_x_valid = AE_predict(x_valid, AE_model)
        compressed_x_test = AE_predict(x_test, AE_model)

    if (method == 'PCA'):
        PCA_model = PCA_fitting(x_train, reduced_dimensions)
        compressed_x_train = PCA_predict(x_train, PCA_model)
        compressed_x_valid = PCA_predict(x_valid, PCA_model)
        compressed_x_test = PCA_predict(x_test, PCA_model)

    return compressed_x_train, compressed_x_valid, compressed_x_test, y_train, y_valid, y_test

def LightGBM(space):

    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = Dimension_reduction(space['reduced_dimension'])

    params = space.copy()
    params.pop('reduced_dimension')
    
    lgb_train = lgb.Dataset(X_train, Y_train,  free_raw_data=False)
    lgb_valid = lgb.Dataset(X_valid, Y_valid,  reference=lgb_train, free_raw_data=False)

    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=lgb_valid,
                    num_boost_round=1000,
                    verbose_eval=1,
                    early_stopping_rounds=150)

    return gbm, X_test, Y_test

def f(space):

    gbm, X_test, Y_test = LightGBM(space)

    Y_train_pred_softmax = gbm.predict(X_train, num_iteration=gbm.best_iteration)
    Y_train_pred = [list(i).index(max(i)) for i in Y_train_pred_softmax]
    Y_valid_pred_softmax = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
    Y_valid_pred = [list(i).index(max(i)) for i in Y_valid_pred_softmax]
    Y_test_pred_softmax = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    Y_test_pred = [list(i).index(max(i)) for i in Y_test_pred_softmax]



    result = {'loss': - accuracy_score(Y_test, Y_test_pred),
            'accuracy_score_train': accuracy_score(Y_train, Y_train_pred),
            'accuracy_score_valid': accuracy_score(Y_valid, Y_valid_pred),
            'accuracy_score_test': accuracy_score(Y_test, Y_test_pred),
            'precision_score_test': precision_score(Y_test, Y_test_pred, average='micro'),
            'recall_score_test': recall_score(Y_test, Y_test_pred, average='micro'),
            'f1_score_test': f1_score(Y_test, Y_test_pred, average='micro'),
            'space': space,
            'status': STATUS_OK}

    print(space)
    print(result)

    return result

if __name__ == "__main__":


    method = 'PCA'
    reduced_dimensions = [508, 624, 757]



    trials = Trials()
    best = fmin(f, space, algo=tpe.suggest, max_evals=100, trials=trials)

    records = pd.DataFrame()
    row = 0
    for record in trials.trials:
        print(record)
        for i in record['result']['space'].keys():
            records.loc[row, i] = record['result']['space'][i]
        record['result'].pop('space')
        for i in record['result'].keys():
            records.loc[row, i] = record['result'][i]
        row = row + 1
    records.to_csv('records.csv')


    print(best)

