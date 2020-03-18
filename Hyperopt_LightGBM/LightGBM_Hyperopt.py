import datetime as dt

import lightgbm as lgb
import numpy as np
import pandas as pd
from Autoencoder_for_LightGBM import AE_fitting, AE_predict
from PCA_for_LightGBM import PCA_fitting, PCA_predict
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

from Preprocessing.LoadData import load_data, sample_from_main

space = {
    # dimension
    'reduced_dimension' : hp.choice('reduced_dimension', np.arrange(0.66, 0.75, 0.01)), # past: [508, 624, 757]

    # better accuracy
    'learning_rate': hp.choice('learning_rate', np.arrange(0.75, 1.0, 0.05)),
    'boosting_type': 'gbdt', # past:  hp.choice('boosting_type', ['gbdt', 'dart']
    'max_bin': hp.choice('max_bin', [31, 63, 127, 255]),
    'num_leaves': hp.choice('num_leaves', [20, 30, 50, 100, 115, 125]),

    # avoid overfit
    'min_data_in_leaf': hp.choice('min_data_in_leaf', np.arrange(750, 1800, 150, dtype=int)),
    'feature_fraction': hp.choice('feature_fraction', [0.3, 0.4, 0.5, 0.6]),
    'bagging_fraction': hp.choice('bagging_fraction', np.arange(0.3, 0.4, 0.02)),
    'bagging_freq': hp.choice('bagging_freq', [3, 4]),
    'min_gain_to_split': hp.choice('min_gain_to_split', np.arange(0.5, 0.7, 0.04)),
    'lambda_l1': hp.choice('lambda_l1', [2,3,4, 13, 15, 17]),
    'lambda_l2': hp.choice('lambda_l2', np.arange(270, 540, 30, dtype=int)),

    # parameters won't change
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_error',
    'num_boost_round': 1000,
    'num_threads': 12  # for the best speed, set this to the number of real CPU cores
    }

space_check = {
    # check
    'num_boost_round': hp.choice('num_boost_round', [100, 1000]),
    'learning_rate': hp.choice('learning_rate', [1, 5]),

    # dimension
    'reduced_dimension' : 0.7,

    # better accuracy
    'boosting_type': 'gbdt',
    'max_bin': 255,
    'num_leaves': 400,

    # avoid overfit
    'min_data_in_leaf': 250,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 8,
    'min_gain_to_split': 0.05,
    'lambda_l1': 0,
    'lambda_l2': 1,

    # parameters won't change
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_error',
    'num_threads': 12  # for the best speed, set this to the number of real CPU cores
    }

space_check_full = {
    'num_boost_round': 1000,
    'reduced_dimension': 0.7,
    'learning_rate': 0.1,
    'boosting_type': 'gbdt',
    'max_bin': 255,
    'num_leaves': 400,
    'min_data_in_leaf': 250,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 8,
    'min_gain_to_split': 0.05,
    'lambda_l1': 0,
    'lambda_l2': 1,
    'objective': 'multiclass',
    'num_class': 6,
    'metric': 'multi_error',
    'num_threads': 12  # for the best speed, set this to the number of real CPU cores
    }

def load(q):
    main = load_data(lag_year=5, sql_version = False)    # main = entire dataset before standardization/qcut
    col = main.columns[2:-2]
    dfs = sample_from_main(main, y_type='yoy', part=1, q=q)  # part=1: i.e. test over entire 150k records
    x, y = dfs[0]
    return x, y, col

def Dimension_reduction(reduced_dimensions, method='PCA'):

    method = 'PCA'

    x_lgbm, x_test, y_lgbm, y_test = train_test_split(x, y, test_size=0.2)
    x_train, x_valid, y_train, y_valid = train_test_split(x_lgbm, y_lgbm, test_size=0.25)

    if (method == 'AE'):
        AE_model = AE_fitting(x_train, reduced_dimensions)
        compressed_x_train = AE_predict(x_train, AE_model)
        compressed_x_valid = AE_predict(x_valid, AE_model)
        compressed_x_test = AE_predict(x_test, AE_model)

    elif (method == 'PCA'):
        PCA_model = PCA_fitting(x_train, reduced_dimensions)
        compressed_x_train = PCA_predict(x_train, PCA_model)
        compressed_x_valid = PCA_predict(x_valid, PCA_model)
        compressed_x_test = PCA_predict(x_test, PCA_model)
        print('reduced_dimensions:', reduced_dimensions)
        print('x_train shape after PCA:', compressed_x_train.shape)

    else:
        compressed_x_train = x_train
        compressed_x_valid = x_valid
        compressed_x_test = x_test

    return compressed_x_train, compressed_x_valid, compressed_x_test, y_train, y_valid, y_test

def LightGBM(space):

    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = Dimension_reduction(space['reduced_dimension'])

    params = space.copy()
    params.pop('reduced_dimension')
    params['num_leaves'] = int(params['num_leaves'])

    lgb_train = lgb.Dataset(X_train, Y_train,  free_raw_data=False)
    lgb_valid = lgb.Dataset(X_valid, Y_valid,  reference=lgb_train, free_raw_data=False)

    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=lgb_valid,
                    verbose_eval=1,
                    early_stopping_rounds=150)

    # print and save feature importance for model
    # importance = gbm.feature_importance(importance_type='split')
    # print(col)
    # feature_importance = pd.DataFrame({'feature_name': col, 'importance': importance})
    # print(feature_importance)
    # feature_importance.to_csv('feature_importance.csv', index=False)

    # predict Y
    Y_train_pred_softmax = gbm.predict(X_train, num_iteration=gbm.best_iteration)
    Y_train_pred = [list(i).index(max(i)) for i in Y_train_pred_softmax]
    Y_valid_pred_softmax = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
    Y_valid_pred = [list(i).index(max(i)) for i in Y_valid_pred_softmax]
    Y_test_pred_softmax = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    Y_test_pred = [list(i).index(max(i)) for i in Y_test_pred_softmax]

    return Y_train, Y_train_pred, Y_valid, Y_valid_pred, Y_test, Y_test_pred

def f(space):

    Y_train, Y_train_pred, Y_valid, Y_valid_pred, Y_test, Y_test_pred = LightGBM(space)

    # why result is dictionary???????
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

def main(space, max_evals, name=''):
    print('-------------------- start hyperopt for lightgbm {} --------------------'.format(name))

    d = dt.datetime.today().strftime('%Y%m%d')
    save_name = 'records_{}_{}.csv'.format(d, name)

    trials = Trials()
    best = fmin(fn=f, space=space, algo=tpe.suggest, max_evals=max_evals,
                trials=trials)  # space = space for normal run; max_evals = 50

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

    records.to_csv(save_name)
    print(best)

if __name__ == "__main__":
    qcut_q = 3
    x, y, col = load(q=qcut_q)
    space['num_class'] = qcut_q

    # main(space=space_check_full, max_evals=1)
    main(space=space, max_evals=200, name='qcut{}_200'.format(qcut_q))

    print('x shape before PCA:', x.shape)
