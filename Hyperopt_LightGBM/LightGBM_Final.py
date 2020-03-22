import datetime as dt
import gc

import lightgbm as lgb
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, r2_score, fbeta_score, precision_score, recall_score, \
    accuracy_score, cohen_kappa_score, hamming_loss, jaccard_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from Preprocessing.LoadData import (load_data, sample_from_datacqtr)

params_over = '0.8	8	gbdt	0.8	2	5	0.075	255	multi_error	500	0.2	1000	3	200	12	multiclass	0.66'
params_1 = '0.6	5	gbdt	0.6	2	15	0.060295296	255	multi_error	750	2.4	1000	3	50	12	multiclass	0.728340625'
params_2 = '0.8	8	gbdt	0.8	2	1	0.1	255	multi_error	500	0.2	1000	3	200	12	multiclass	0.66'
params_3 = '0.6	2	gbdt	0.3	2	45	0.085002592	128	multi_error	1200	0.6	1000	3	75	12	multiclass	0.710608306'
params_qoq = '0.7	8	gbdt	0.4	11	430	0.6	255	multi_error	800	0.64	1000	3	155	12	multiclass	0.72'

def set_params(params_fig):
    params_name = 'bagging_fraction	bagging_freq	boosting_type	feature_fraction	lambda_l1	lambda_l2	learning_rate	max_bin	metric	min_data_in_leaf	min_gain_to_split	num_boost_round	num_class	num_leaves	num_threads	objective	reduced_dimension'
    params_name = params_name.split()
    params_fig = params_fig.split()

    params = {}
    for i in range(len(params_name)):
        if params_name[i] in ['bagging_fraction', 'feature_fraction', 'lambda_l1', 'lambda_l2', 'learning_rate',
                              'min_gain_to_split']:
            params_fig[i] = float(params_fig[i])
        if params_name[i] in ['bagging_freq', 'max_bin', 'min_data_in_leaf', 'num_boost_round', 'num_class',
                              'num_leaves', 'num_threads']:
            params_fig[i] = int(params_fig[i])
        params[params_name[i]] = params_fig[i]

    params.pop('reduced_dimension')
    print(params)
    return params

params = set_params(params_qoq)

def myPCA(n_components, train_x, test_x):

    '''PCA 此处应添加直接引用PCA指定阈值ratio数量的参数'''
    pca = PCA(n_components=n_components)  # Threshold for dimension reduction, float or integer
    pca.fit(train_x)
    new_train_x = pca.transform(train_x)
    new_test_x = pca.transform(test_x)
    print('PCA components:', n_components)
    print('after PCA train shape', new_train_x.shape)
    return new_train_x, new_test_x

def myLightGBM(X_train, X_valid, X_test, Y_train, Y_valid):

    '''                                  X_train -> Y_train_pred
    X_train + Y_train -> gbm (model) ->  X_valid -> Y_valid_pred
    X_valid + Y_valid ->                 X_test  -> Y_test_pred
    '''

    '''Training'''
    lgb_train = lgb.Dataset(X_train, label=Y_train, free_raw_data=False)
    lgb_eval = lgb.Dataset(X_valid, label=Y_valid, reference=lgb_train, free_raw_data=False)

    print('Starting training...')
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=lgb_eval,  # eval training data
                    # verbose=1,
                    early_stopping_rounds=150)

    '''save model'''
    def save_model():
        print('Saving model...')
        # save model to file
        gbm.save_model('model.txt')

    '''print and save feature importance for model'''
    def feature_importance():
        importance = gbm.feature_importance(importance_type='split')
        name = gbm.feature_name()
        feature_importance = pd.DataFrame({'feature_name': name, 'importance': importance})
        print(feature_importance)
        feature_importance.to_csv('feature_importance.csv', index=False)

    '''Evaluation on Test Set'''
    print('Loading model to predict...')

    # load model to predict
    # bst = lgb.Booster(model_file='model.txt')

    Y_train_pred_softmax = gbm.predict(X_train, num_iteration=gbm.best_iteration)
    Y_train_pred = [list(i).index(max(i)) for i in Y_train_pred_softmax]
    Y_valid_pred_softmax = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
    Y_valid_pred = [list(i).index(max(i)) for i in Y_valid_pred_softmax]
    Y_test_pred_softmax = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    Y_test_pred = [list(i).index(max(i)) for i in Y_test_pred_softmax]

    return Y_train_pred, Y_valid_pred, Y_test_pred

def eval(X_train, X_valid, X_test, Y_train, Y_valid, Y_test):

    Y_train_pred, Y_valid_pred, Y_test_pred = myLightGBM(X_train, X_valid, X_test, Y_train, Y_valid)

    result = {'loss': - accuracy_score(Y_test, Y_test_pred),
              'accuracy_score_train': accuracy_score(Y_train, Y_train_pred),
              'accuracy_score_valid': accuracy_score(Y_valid, Y_valid_pred),
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
    print(result)

    return result

def each_round(main, y_type, testing_period, n_components, valid_method, valid_no):

    # retrieve gvkey, datacqtr columns for train_valid set from main
    end = testing_period
    start = testing_period - relativedelta(years=20) # define training period
    label_df = main.iloc[:,:2]
    label_df = label_df.loc[(start <= label_df['datacqtr']) & (label_df['datacqtr'] < end)].reset_index(drop=True)

    X_train_valid, X_test, Y_train_valid, Y_test = sample_from_datacqtr(main, y_type=y_type,
                                                                        testing_period=testing_period, q=3)

    '''1. PCA on train_x, test_x'''
    X_train_valid_PCA, X_test_PCA = myPCA(n_components, X_train_valid, X_test)

    '''2. train_test_split'''
    if valid_method == 'shuffle':
        test_size = valid_no/80
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_valid_PCA, Y_train_valid, test_size=test_size, random_state=666)
    elif valid_method == 'chron':

        def split_chron(df):
            date_df = pd.concat([label_df, pd.DataFrame(df)], axis=1)
            valid_period = testing_period - valid_no * relativedelta(months=3)
            print('validation period start:', valid_period)
            train = date_df.loc[(date_df['datacqtr'] < valid_period), date_df.columns[2:]].values
            valid = date_df.loc[(date_df['datacqtr'] >= valid_period), date_df.columns[2:]].values
            return train, valid

        X_train, X_valid = split_chron(X_train_valid_PCA)

        Y_train, Y_valid = split_chron(Y_train_valid)
        Y_train = np.reshape(Y_train, -1)
        Y_valid = np.reshape(Y_valid, -1)

    '''3. train & evaluate LightGBM'''
    return eval(X_train, X_valid, X_test_PCA, Y_train, Y_valid, Y_test)

def main(y_type, sample_no, n_components, valid_method, valid_no=None):

    main = load_data(lag_year=5, sql_version=False)  # main = entire dataset before standardization/qcut

    results = {}

    # roll over each round
    period_1 = dt.datetime(2008, 3, 31)
    for i in tqdm(range(sample_no)):
        testing_period = period_1 + i * relativedelta(months=3)
        results[i] = each_round(main, y_type, testing_period, n_components, valid_method, valid_no)
    del main
    gc.collect()

    records = pd.DataFrame()
    for date in results.keys():
        for col in results[date].keys():
            records.loc[date, col] = results[date][col]

    records.to_csv('final_result_{}{}.csv'.format(valid_method, valid_no))

if __name__ == "__main__":
    y_type = 'qoq'
    sample_no = 40
    n_components = 0.72
    for valid_method in ['chron', 'shuffle']:
        for valid_no in [1,5,10]:
            print(valid_method, valid_no)
            main(y_type, sample_no, n_components, valid_method, valid_no=valid_no)