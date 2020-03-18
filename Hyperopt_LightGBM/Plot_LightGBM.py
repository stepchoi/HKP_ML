import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


def result_boxplot(csv_name, only_test = False):
    df = pd.read_csv(os.getcwd()+'/records/' + csv_name + '.csv')    # PCA_LightGBM_Hyperopt
    df_num = df.iloc[:, 1:-8]

    # remove underfit
    # df = df.loc[df['learning_rate']==0.1]

    option = {}
    scat = {}
    for col in df_num:
        if len(set(df[col])) in np.arange(2,10,1):
            option[col] = set(df[col])
        elif len(set(df[col])) > 10:
            scat[col] = df[col]


    fig = plt.figure(figsize=(20, 16), dpi=120)
    n = round((len(option.keys())+len(scat.keys()))**0.5,0)+1
    k = 1
    for i in option.keys():
        print(i, option[i])
        data = []
        data2 = []
        label = []
        for name, g in df.groupby([i]):
            label.append(name)
            data.append(g['accuracy_score_test'])
            data2.append(g['accuracy_score_train'])

        ax = fig.add_subplot(n, n, k)
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
            csv_name = csv_name+'_test'
        ax.set_title(i)
        k += 1

    for i in scat.keys():
        ax = fig.add_subplot(n, n, k)
        ax.scatter(scat[i], df['accuracy_score_test'], df['accuracy_score_train'])
        ax.set_title(i)
        k+= 1


    fig.suptitle(csv_name, fontsize=14)
    fig.savefig(os.getcwd()+'/visuals/' + csv_name+'.png')

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
    df = pd.read_csv(csv_name + '.csv')    # PCA_LightGBM_Hyperopt
    para = df.columns[1:17].to_list()
    print(df.groupby(para).filter(lambda x: len(x)>1))

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

if __name__ == '__main__':

    # os.chdir(os.getcwd()+'/records')


    # same_para('records_20200313')
    # eta_accuracy('records_20200313')
    result_boxplot('records_20200318_qcut3_200', only_test=True)
    # round_eta_accuracy('records_eta_round_acc')

