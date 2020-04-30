import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


def result_boxplot(qcut, y_type, only_test = False):

    df = pd.read_csv('/Users/Clair/PycharmProjects/HKP_ML_DL/Preprocessing/raw/ibes/ibes_new/boxplot/records_full.csv')    # PCA_LightGBM_Hyperopt

    subset_name = '{}{}'.format(y_type, qcut)
    print('Subset analysed: ', subset_name)
    df = df.loc[(df['y_type']==y_type) & (df['qcut']==qcut)]
    print('Result df shape: ', df.shape)

    # df_num = df.iloc[:, 1:-8]
    # print(df.columns)
    # print(df.columns)
    # print(df[''])
    # exit(0)

    option = {}
    for col in df.columns[5:28]:
        # print(col)
        if len(set(df[col])) in np.arange(2,20,1):
            option[col] = set(df[col])
            # print('success')

    print('Params to be analysed: ', option.keys())

    n = round((len(option.keys()))/3,0)+1
    fig = plt.figure(figsize=(4*n,12), dpi=120)
    fig.suptitle(subset_name, fontsize=14)

    k = 1
    for i in option.keys():
        # print(i, option[i])
        data = []
        data2 = []
        label = []
        for name, g in df.groupby([i]):
            # print(name)
            if type(name) == str:
                label.append(name)
            else:
                label.append(round(name,2))

            data.append(g['accuracy_score_test'])
            data2.append(g['accuracy_score_train'])

        ax = fig.add_subplot(3, n, k)

        def draw_plot(ax, data, label, edge_color, fill_color):
            bp = ax.boxplot(data, labels = label, patch_artist=True)

            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color=edge_color)

            for patch in bp['boxes']:
                patch.set(facecolor=fill_color)

        draw_plot(ax, data, label, 'red', 'tan')
        if only_test == False:
            draw_plot(ax, data2, label, 'blue', 'cyan')
        else:
            subset_name = subset_name+'_test'
        ax.set_title(i)
        k += 1

    fig.savefig('/Users/Clair/PycharmProjects/HKP_ML_DL/Preprocessing/raw/ibes/ibes_new/boxplot/{}.png'.format(subset_name))

def eta_accuracy(csv_name):
    df = pd.read_csv(csv_name + '.csv')    # PCA_LightGBM_Hyperopt

    x = np.log(df['learning_rate']).values.reshape(-1, 1)
    y = df['accuracy_score_train'].values.reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)

    plt.scatter(x,y)
    plt.suptitle(csv_name + ' (train_R2 = {})'.format(r_sq),fontsize=14)
    plt.savefig(csv_name + '_train_R2.png')

def same_para(csv_name):
    df = pd.read_csv(os.getcwd()+'/records/' + csv_name + '.csv')    # PCA_LightGBM_Hyperopt
    para = df.columns[1:18].to_list()
    print(para)

    df1 = df.groupby(para).mean().reset_index(drop=False)
    df2 = df.groupby(para).count().reset_index(drop=True)['status']
    df3 = pd.concat([df1, df2], axis=1)
    print(df3)
    df3.to_csv(os.getcwd()+'/records/' + csv_name+'_mean.csv')

def round_eta_accuracy(csv_name):
    df = pd.read_csv(csv_name + '.csv')    # PCA_LightGBM_Hyperopt
    df_mean = df.groupby(['learning_rate','num_boost_round']).mean().reset_index(drop=False)
    df_pivot = df_mean.pivot(index='learning_rate', columns = 'num_boost_round', values='accuracy_score_train')
    df_pivot.to_csv('eta_round.csv')
    exit(0)
    l = np.log(df['learning_rate'])
    n = np.log(df['num_boost_round'])
    acc = round(df['accuracy_score_train'],2)
    plt.scatter(l, n, c=acc, s=30, cmap='gray')

    for i, txt in enumerate(n):
        plt.annotate(acc[i], (l[i], n[i]))

    plt.legend()
    plt.show()

def final_plot(dict_name):

    results = {}
    aggre = []
    accuracy = []
    for valid_method in ['chron', 'shuffle']:
        for valid_no in [1,5,10]:
            csv_name = '{}{}'.format(valid_method, valid_no)
            results[csv_name] = pd.read_csv('{}/final_results/{}/final_result_{}.csv'.format(os.getcwd(), dict_name, csv_name))
            avg = results[csv_name].describe()
            acc = results[csv_name][['accuracy_score_train', 'accuracy_score_test']]
            acc.columns = [csv_name+'_train', csv_name +'_test']
            avg['Unnamed: 0'] = csv_name
            aggre.append(avg)
            accuracy.append(acc)

    result_acc = pd.concat(accuracy,axis=1)
    result_acc['max_test'] = result_acc.iloc[:,np.arange(1,12,2)].max(axis=1)
    result_acc['max_test_idx'] = result_acc.iloc[:, np.arange(1, 12, 2)].idxmax(axis=1)

    result_average = pd.concat(aggre,axis=0)
    result_acc.to_csv('{}/final_results/{}/final_result_acc.csv'.format(os.getcwd(), dict_name))
    result_average.to_csv('{}/final_results/{}/final_result_average.csv'.format(os.getcwd(), dict_name))

    result_acc.iloc[:,np.arange(1,12,2)].plot(figsize=(10,5), grid=True)
    plt.savefig('{}/final_results/{}/test_acc.png'.format(os.getcwd(), dict_name))

    result_acc.iloc[:, np.arange(0, 11, 2)].plot(figsize=(10, 5), grid=True)
    plt.savefig('{}/final_results/{}/train_acc.png'.format(os.getcwd(), dict_name))

if __name__ == '__main__':

    # 1. analyze full records -> Boxplot
    result_boxplot(qcut=3, y_type='yoyr',only_test=False)

    # 2.



    # df = pd.read_csv('best.csv').drop_duplicates()
    # df['testing_period'] = pd.to_datetime(df['testing_period'])
    #
    # # print(df.iloc[0])
    # plt.scatter(df['trial'],df[['accuracy_score_test', 'accuracy_score_valid', 'accuracy_score_train']])
    # plt.show()

    # for col in ['max_evals','valid_method','valid_no']
    # for col in df.columns:
    #     print(col,len(set(df[col])-set([np.nan])))