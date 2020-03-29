import gc
import datetime as dt
import pandas as pd
from Preprocessing.LoadData import (load_data, clean_set)
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score,r2_score,fbeta_score,roc_auc_score,precision_score,recall_score,accuracy_score
from sklearn.metrics import cohen_kappa_score,hamming_loss,jaccard_score,hinge_loss
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential,load_model
from keras.layers import Dense , BatchNormalization , Dropout , Activation
from keras.layers import LSTM, GRU
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import adam , SGD , RMSprop

'''
part_dict = sample_from_main(part=5)  # part_dict[0], part_dict[1], ... would be arrays after standardization
print(part_dict)

for i in part_dict.keys():
    pass
'''

main = load_data(lag_year = 1,sql_version=False)  # change sql_version -> True if trying to run this code through Postgres Database
period_1 = dt.datetime(2008, 3, 31)
main_period = clean_set(main, period_1)
train_x, test_x = main_period.standardize_x()
train_yoy, test_yoy = main_period.yoy(q=3)

'''PCA'''
pca = PCA(n_components=0.66)  # Threshold for dimension reduction,float or integer
#此处应添加直接引用PCA指定阈值ratio数量的参数
#n_features=
train_x=pd.DataFrame(train_x)
test_x=pd.DataFrame(test_x)
pca.fit(train_x)
new_train_x = pca.transform(train_x)
pca.fit(test_x)
new_test_x = pca.transform(test_x)
print(new_train_x.shape)
print(new_test_x.shape)

del main_period  # delete this train_x and collect garbage -> release memory
gc.collect()

'''Converting data scale'''
#X_train=pd.DataFrame(new_train_x)
#y_train=pd.DataFrame(train_yoy)
#X_test=pd.DataFrame(new_test_x)
#y_test=pd.DataFrame(test_yoy)

trainX = new_train_x.reshape(new_train_x.shape[0], 20, new_train_x.shape[1])
testX = new_test_x.reshape(new_test_x.shape[0], 20, new_test_x.shape[1])

print(trainX.shape)
print(testX.shape)

'''Model Construction'''

model_name = 'Earnings_prediction_GRU'
model = Sequential()
model.add(GRU(256, input_shape=(20, n_features), return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(256))
model.add(Dropout(0.2))
model.add(GRU(256))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

'''Model Training'''

model.fit(trainX, y_train, batch_size=250, epochs=500, validation_split=0.2, verbose=1)
#callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)]
model.save("{}.h5".format(model_name))
print('MODEL-SAVED')

'''Model Evaluation'''

#model = load_model("{}.h5".format(model_name))
#print("MODEL-LOADED")

score = model.evaluate(testX, y_test)
print('Score: {}'.format(score))
#yhat = model.predict(testX)
#yhat = MinMaxScaler().inverse_transform(yhat)
#y_test = MinMaxScaler().inverse_transform(test_yoy)
#plt.plot(yhat[-100:], label='Predicted')
#plt.plot(y_test[-100:], label='Ground Truth')
#plt.legend()
#plt.show()