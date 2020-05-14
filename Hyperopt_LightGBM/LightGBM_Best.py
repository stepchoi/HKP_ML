import argparse

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from LightGBM_Final import convert_main
from LoadData import (load_data)
from hyperopt import hp
from sqlalchemy import create_engine, INTEGER, TIMESTAMP, TEXT, BIGINT

# define parser use for server running
parser = argparse.ArgumentParser()
parser.add_argument('--bins', type=int, default=3)
parser.add_argument('--sample_no', type=int, default=40)
parser.add_argument('--sql_version', default=False, action='store_true')
parser.add_argument('--y_type', default='qoq')
args = parser.parse_args()

# args.y_type = 'yoyr'  # CHANGE FOR HKPC

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

class best_model_rerun:

    def __init__(self):
        self.types = {'gvkey': INTEGER(), 'datacqtr': TIMESTAMP(), 'actual': BIGINT(), 'lightgbm_result': BIGINT(),
                 'y_type': TEXT(), 'qcut': BIGINT()}

        for qcut in [3]: # , 6, 9
            self.db_max = self.best_iteration(y_type=args.y_type, qcut=qcut)

            for i in [39]: #range(len(self.db_max))
                sql_result.update(self.db_max.iloc[i, :].to_dict())
                space.update(self.db_max.iloc[i, 6:].to_dict())
                space.update({'num_class': qcut, 'is_unbalance': True})
                print('sql_result1: ', sql_result)
                self.step_load_data()
                self.step_lightgbm()

    def best_iteration(self, qcut, y_type):
        max_sql_string = "select y_type, testing_period, qcut, reduced_dimension, valid_method, valid_no, " \
                         "bagging_fraction, bagging_freq, feature_fraction, lambda_l1, lambda_l2, learning_rate, max_bin,\
                          min_data_in_leaf, min_gain_to_split, num_leaves \
                            from ( select *, max(accuracy_score_test) over (partition by testing_period) as max_thing\
                                   from lightgbm_results\
                                 where (trial IS NOT NULL) AND name='after update y to /atq' AND qcut={} AND y_type='{}') t\
                            where accuracy_score_test = max_thing\
                            Order By testing_period ASC".format(qcut, y_type)

        db_max = pd.read_sql(max_sql_string, engine).drop_duplicates(subset=['testing_period'], keep='first')
        return db_max

    def step_load_data(self):


        convert_main_class = convert_main(main, sql_result['y_type'], sql_result['testing_period'], sql_result)

        self.X_train, self.X_valid, self.X_test, self.Y_train, \
        self.Y_valid, self.Y_test = convert_main_class.split_valid(sql_result['valid_method'], sql_result['valid_no'])

        label_df = main.iloc[:, :2]
        self.label_df = label_df.loc[label_df['datacqtr'] == sql_result['testing_period']].reset_index(drop=True)

    def step_lightgbm(self):
        params = space.copy()

        '''Training'''
        lgb_train = lgb.Dataset(self.X_train, label=self.Y_train, free_raw_data=False)
        lgb_eval = lgb.Dataset(self.X_valid, label=self.Y_valid, reference=lgb_train, free_raw_data=False)

        gbm = lgb.train(params,
                        lgb_train,
                        valid_sets=lgb_eval,
                        num_boost_round=1000, # change to 1000!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        early_stopping_rounds=150,
                        )

        f = plt.figure()
        shap_values = shap.TreeExplainer(gbm).shap_values(self.X_train)
        shap.summary_plot(shap_values, self.X_train)
        file_name = '{}{}{}'.format(sql_result['testing_period'], sql_result['y_type'], sql_result['qcut'])
        # gbm.save_model('model{}.txt'.format(file_name))
        f.savefig("summary_plot_{}1.png".format(file_name), bbox_inches='tight', dpi=600)

        '''Evaluation on Test Set'''
        Y_test_pred_softmax = gbm.predict(self.X_test, num_iteration=gbm.best_iteration)
        Y_test_pred = [list(i).index(max(i)) for i in Y_test_pred_softmax]

        self.label_df['actual'] = self.Y_test
        self.label_df['lightgbm_result'] = Y_test_pred
        self.label_df['y_type'] = sql_result['y_type']
        self.label_df['qcut'] = sql_result['qcut']

        # self.label_df.to_sql('lightgbm_results_best', con=engine, index=False, if_exists='append', dtype=types)
        # print('finish:', sql_result['testing_period'])

if __name__ == "__main__":

    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    # parser
    print(args)
    qcut_q = int(args.bins)
    y_type = args.y_type  # 'yoyr','qoq','yoy'

    sql_result = {'qcut': qcut_q, 'y_type': y_type}

    # load data for entire period
    main = load_data(lag_year=5, sql_version=args.sql_version)  # CHANGE FOR DEBUG
    label_df = main.iloc[:,:2]

    space['num_class'] = qcut_q
    if qcut_q ==2:
        space['is_unbalance'] = True

    best_model_rerun() # run best model

