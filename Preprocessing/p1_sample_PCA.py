#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

os.chdir('/Users/Clair/Desktop/FYP/data')


# In[4]:


#1: import all sample dataset
main_abs = pd.read_csv("main_abs.csv")
main_log = pd.read_csv("main_log.csv")
main_yoy = pd.read_csv("main_yoy.csv")
main_qoq = pd.read_csv("main_qoq.csv")
main_dep = pd.read_csv("main_dependent.csv")


# In[25]:


#2: import feature selection dataset -> dictionary separate feature by needed formats (type_dict)
bytype = pd.read_csv('pca1_select_type.csv',header = None)
ie = pd.read_csv('ie_70.csv')
bytype = pd.merge(bytype, ie, left_on = 0, right_on = ['long'], how = 'left')

grouped = bytype.groupby(1)
type_dict = {}
for name, g in grouped:
    type_dict[name] = g['short'].dropna().to_list()

num_col = list(set(bytype['short'].dropna())) # all feature columns

label_col = ['gvkey', 'datacqtr', 'datafqtr', 'gsector', 'cquarter',
                 'cyear', 'cyeargvkey', 'gvkeydatafqtr']


# In[63]:


# 3: select from each individual database and concat together -> dataframe in needed format (main)
main_label = main_abs.filter(label_col)
main_abs_select = main_abs.filter(type_dict['Absolute'])
main_log_select = main_log.filter(type_dict['Log'])
main_qoq_select = main_qoq.filter(type_dict['QoQ'])
main_yoy_select = main_yoy.filter(type_dict['YoY'])
main = pd.concat([main_label, main_abs_select, main_log_select,
                  main_qoq_select, main_yoy_select], axis=1)


# In[49]:


'''starting point 1 - main.shape = (309036, 78)'''
main = pd.read_csv('main_selected.csv')


# In[51]:


#3.1: fillna
'''
1. fill YoY, QoQ -> 0
2. fill abs, log -> last observation
3. fill abs, log -> next observation
4. fill rest -> 0
'''
for i in ['YoY', 'QoQ']:
    main[type_dict[i]] = main[type_dict[i]].fillna(0)
for i in ['Absolute', 'Log']:
    main[type_dict[i]] = main.groupby('gvkey').apply(lambda x: x.fillna(method = 'ffill'))[type_dict[i]]
    main[type_dict[i]] = main.groupby('gvkey').apply(lambda x: x.fillna(method = 'bfill'))[type_dict[i]]
main = main.fillna(0)


# In[59]:


# 4: construct PCA dataframe -> whole dataframe (df_pca) for 1980 -2020
gvkey = list(set(main['gvkey']))        # gvkey list = 4592
datacqtr = list(set(main['datacqtr']))  # cqtr list = 158 (1980Q2 - 2020Q1)

df_pca = pd.DataFrame()
df_pca['gvkey'] = gvkey

grouped = main.groupby('datafqtr')
for cqtr, g in grouped:
    print(cqtr)
    g_num = g.filter(num_col + ['gvkey'])
    g_num.columns = [col + '_' + cqtr for col in num_col] + ['gvkey']
    df_pca = pd.merge(df_pca, g_num, on = ['gvkey'], how = 'left')


# In[60]:


'''starting point 2 - df_pca.shape = (4592, 11271)'''
#df_pca.to_csv('pca_full.csv')
df_pca = pd.read_csv('pca_full.csv')


# In[65]:


# 5: extract dataframe for selected window[3, 5] -> dataframe (pca_df_5) + (pca_df_5)
def filter_by_window(window, last_year = 2019, df_pca = df_pca):
    cqtr_window =[]
    for yr in np.arange(last_year - window,last_year,1):
        cqtr_window.extend([str(yr)+'Q'+str(i+1) for i in range(4)])

    num_col_cqtr = []
    for t in cqtr_window:
        num_col_cqtr.extend([col + '_' + t for col in num_col])
    print(cqtr_window[0], cqtr_window[-1], len(num_col_cqtr))
    
    df_pca_sub = df_pca.filter(['gvkey'] + num_col_cqtr)
    df_pca_sub = df_pca_sub.dropna(subset = num_col_cqtr)
    df_pca_sub.to_csv('df_pca_'+str(window)+'.csv', index = False)
    return df_pca_sub

df_pca_3 = filter_by_window(3)
df_pca_5 = filter_by_window(5)


# In[ ]:




