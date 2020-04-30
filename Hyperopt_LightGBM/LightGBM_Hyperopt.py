import datetime as dt

import lightgbm as lgb
import numpy as np
import pandas as pd
from LoadData import (load_data, sample_from_main)
# from Autoencoder_for_LightGBM import AE_fitting, AE_predict
from PCA_for_LightGBM import PCA_fitting, PCA_predict
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

space = {
    # dimension
    'reduced_dimension' : hp.choice('reduced_dimension', np.arange(0.66, 0.7, 0.75)), # past: [508, 624, 757]

    # better accuracy
    'learning_rate': hp.choice('learning_rate', np.arange(0.4, 2, 0.4)),
    'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
    'max_bin': hp.choice('max_bin', [127, 255]),
    'num_leaves': hp.choice('num_leaves', np.arange(50,400,50, dtype=int)),

    # avoid overfit
    'min_data_in_leaf': hp.choice('min_data_in_leaf', np.arange(100, 500, 100, dtype=int)),
    'feature_fraction': hp.choice('feature_fraction', np.arange(0.6, 1, 0.1)),
    'bagging_fraction': hp.choice('bagging_fraction', np.arange(0.6, 1, 0.1)),
    'bagging_freq': hp.choice('bagging_freq', [2,4,8]),
    'min_gain_to_split': hp.choice('min_gain_to_split', np.arange(0.2, 0.8, 0.2)),
    'lambda_l1': hp.choice('lambda_l1', np.arange(0.1, 1, 0.3)),
    'lambda_l2': hp.choice('lambda_l2', np.arange(0.1, 401, 100)),

    # Voting Parallel
    # 'tree_learner': 'voting'
    # 'top_k': 2

    # unbalanced qcut(>6)
    'is_unbalance': True,
    # 'scale_pos_weight':

    # parameters won't change
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_error',
    'num_threads': 16  # for the best speed, set this to the number of real CPU cores
    }

def load(q, y_typq):
    main = load_data(lag_year=5, sql_version = False)    # main = entire dataset before standardization/qcut
    col = main.columns[2:-2]
    dfs = sample_from_main(main, y_type=y_typq, part=1, q=q)  # part=1: i.e. test over entire 150k records
    x, y = dfs[0]
    return x, y

def Dimension_reduction(reduced_dimensions):

    dimension_reduction_method='PCA'
    print('x shape before PCA:', x.shape)

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)
    
    if (dimension_reduction_method == 'AE'):
        from Autoencoder_for_LightGBM import AE_fitting, AE_predict
        AE_model = AE_fitting(x_train, reduced_dimensions)
        compressed_x_train = AE_predict(x_train, AE_model)
        compressed_x_valid = AE_predict(x_valid, AE_model)

    elif (dimension_reduction_method == 'PCA'):
        PCA_model = PCA_fitting(x_train, reduced_dimensions)
        compressed_x_train = PCA_predict(x_train, PCA_model)
        compressed_x_valid = PCA_predict(x_valid, PCA_model)
        print('reduced_dimensions:', reduced_dimensions)
        print('x_train shape after PCA:', compressed_x_train.shape)

    return compressed_x_train, compressed_x_valid, y_train, y_valid

def LightGBM(space):

    X_train, X_valid, Y_train, Y_valid = Dimension_reduction(space['reduced_dimension'])

    params = space.copy()
    params.pop('reduced_dimension')
    params['num_leaves'] = int(params['num_leaves'])

    lgb_train = lgb.Dataset(X_train, Y_train,  free_raw_data=False)
    lgb_valid = lgb.Dataset(X_valid, Y_valid,  reference=lgb_train, free_raw_data=False)

    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=lgb_valid,
                    verbose_eval=-1,
                    num_boost_round=1000,
                    early_stopping_rounds=150)

    # predict Y
    Y_train_pred_softmax = gbm.predict(X_train, num_iteration=gbm.best_iteration)
    Y_train_pred = [list(i).index(max(i)) for i in Y_train_pred_softmax]
    Y_valid_pred_softmax = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
    Y_valid_pred = [list(i).index(max(i)) for i in Y_valid_pred_softmax]

    return Y_train, Y_train_pred, Y_valid, Y_valid_pred

def f(space):

    Y_train, Y_train_pred, Y_valid, Y_valid_pred = LightGBM(space)

    result = {'loss': 1 - accuracy_score(Y_valid, Y_valid_pred),
            'accuracy_score_train': accuracy_score(Y_train, Y_train_pred),
            'accuracy_score_valid': accuracy_score(Y_valid, Y_valid_pred),
            # 'accuracy_score_test': accuracy_score(Y_test, Y_test_pred),
            # 'precision_score_test': precision_score(Y_test, Y_test_pred, average='micro'),
            # 'recall_score_test': recall_score(Y_test, Y_test_pred, average='micro'),
            # 'f1_score_test': f1_score(Y_test, Y_test_pred, average='micro'),
            # 'space': space,
            'status': STATUS_OK}

    pt_dict = {'y_typq': y_typq, 'qcut': qcut_q, 'finish_timing': dt.datetime.now()}
    pt_dict.update(space)
    pt_dict.update(result)

    pt = pd.DataFrame.from_records([pt_dict], index=[0])
    print(pt)
    pt.to_sql('lightgbm_results_hyperopt', con=engine, index=False, if_exists='append')

    return result


if __name__ == "__main__":

    # 1. testing subset for HPOT
    qcut_q = 3
    y_typq = 'qoq'

    # 2. prepare sql location
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    # 3. load data
    x, y = load(q=qcut_q, y_typq=y_typq)
    space['num_class'] = qcut_q

    # 4. HPOT
    trials = Trials()
    best = fmin(fn=f, space=space, algo=tpe.suggest, max_evals=30,
                trials=trials)
    print(best)


