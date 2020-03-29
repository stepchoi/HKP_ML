import datetime as dt
import gc

import lightgbm as lgb
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score, hamming_loss, jaccard_score
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from Preprocessing.LoadData import (load_data, clean_set)

'''
part_dict = sample_from_main(part=5)  # part_dict[0], part_dict[1], ... would be arrays after standardization
print(part_dict)

for i in part_dict.keys():
    pass
'''

main = load_data(sql_version=False)  # change sql_version -> True if trying to run this code through Postgres Database
period_1 = dt.datetime(2008, 3, 31)
main_period = clean_set(main, period_1)
train_x, test_x = main_period.standardize_x(return_test_x = True)
train_yoy, test_yoy = main_period.yoy()

'''PCA'''
pca = PCA(n_components=508)  # Threshold for dimension reduction,float or integer
#此处应添加直接引用PCA指定阈值ratio数量的参数
pca.fit(train_x)
new_train_x = pca.transform(train_x)
pca.fit(test_x)
new_test_x = pca.transform(test_x)

del main_period  # delete this train_x and collect garbage -> release memory
gc.collect()

'''Converting np-array to dataframe'''
X_tr=pd.DataFrame(new_train_x)
y_tr=pd.DataFrame(train_yoy)
#X_test=pd.DataFrame(new_test_x)
#y_test=pd.DataFrame(test_yoy)
X_train, X_test, y_train, y_test = train_test_split(X_tr, y_tr,test_size=0.25, random_state=666)

'''Training'''
lgb_train = lgb.Dataset(X_train, y_train,
                         free_raw_data=False)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train,
                        free_raw_data=False)

#Could add para 'weight=W_train/weight=W_test' after lr

params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_error',
    'num_leaves': 300, #调小防止过拟合
    'max_bin': 255, #调小防止过拟合
   #'min_data_in_leaf': 500, #使用 min_data_in_leaf 和 min_sum_hessian_in_leaf防止过拟合
    'learning_rate': 0.001,
    #'feature_fraction': 0.8, #特征子抽样
    #'bagging_fraction': 0.8, #bagging防止过拟合
    'bagging_freq': 5,
    'is_provide_training_metric':'true',
    #'lambda_l1': 0.4, #正则化参数
    #'lambda_l2': 0.5, #正则化参数
    #'min_gain_to_split': 0.2, #正则化参数
    #'verbose': -1, #一个整数，表示是否输出中间信息。默认值为1。如果小于0，则仅仅输出critical 信息；如果等于0，则还会输出error,warning 信息； 如果大于0，则还会输出info 信息。
    'num_threads':10,
}


print('Starting training...')
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_eval,  # eval training data
                early_stopping_rounds=100)

print('Saving model...')
# save model to file
gbm.save_model('model.txt')

# feature names
print('Feature names:', gbm.feature_name())
# feature importances
print('Feature importances:', list(gbm.feature_importance()))

'''Evaluation on Test Set'''

print('Loading model to predict...')
# load model to predict
bst = lgb.Booster(model_file='model.txt')
# can only predict with the best iteration (or the saving iteration)
y_pred_initial = bst.predict(X_test)
y_pred=[list(x).index(max(x)) for x in y_pred_initial]
# eval with loaded model
print(y_pred_initial)
print(y_pred)
print(y_test)
print("accuracy_score:", accuracy_score(y_test, y_pred))
print("precision_score：", precision_score(y_test, y_pred,average='macro'))
print("recall_score：", recall_score(y_test, y_pred,average='macro'))
print("f1_score_macro:",f1_score(y_test, y_pred,average='macro') )
print("f0.5_score:",fbeta_score(y_test, y_pred,beta=0.5,average='macro') )
print("f2_score:",fbeta_score(y_test, y_pred,beta=2.0,average='macro') )

"cohen_kappa_score:": cohen_kappa_score(y_test, y_pred,labels=None)),
"hamming_loss:": hamming_loss(y_test, y_pred)),
"jaccard_score:": jaccard_score(y_test, y_pred,labels=None,average='macro')),
print("The rmse of loaded model's prediction is:", mean_squared_error(y_test, y_pred) ** 0.5)