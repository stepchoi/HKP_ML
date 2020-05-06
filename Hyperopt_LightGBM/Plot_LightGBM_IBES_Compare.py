import os

import pandas as pd
from sklearn.metrics import accuracy_score


# organize max result
def lightgbm_result_max():

    writer_org = pd.ExcelWriter('results_original.xlsx', engine='xlsxwriter')

    result_dict = {}
    result_dict['consensus'] = {}
    result_dict['results'] = {}

    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            if name.rsplit('.')[1] == 'csv':
                sheet_name = name.rsplit('.')[0]
                print(sheet_name)
                df = pd.read_csv(name)
                result_dict[sheet_name.rsplit('_')[0]][sheet_name.rsplit('_')[1]] = df
                df.reset_index(drop=False).to_excel(writer_org, sheet_name=sheet_name, index=None, header=True)

    # print(result_dict)
    writer_org.save()

    writer = pd.ExcelWriter('results_new_graph_divide.xlsx', engine='xlsxwriter')

    des = {}
    for k in result_dict['consensus'].keys():
        df1 = result_dict['consensus'][k].filter(['Unnamed: 0', 'accuracy_score_test', 'type'])
        df1.columns = ['date', 'consensus', 'type']
        df1 = df1.pivot_table(index='date', columns='type', values='consensus')
        df2 = result_dict['results'][k].filter(['testing_period', 'accuracy_score_test']).drop_duplicates()
        df2.columns = ['date', 'our model']
        df = pd.merge(df1, df2, on=['date'], how='left').set_index('date')

        df = df.append(df.corr()['our model'])
        df = df.append(df.mean(axis=0).rename('average'))

        df.reset_index(drop=False).to_excel(writer, sheet_name=k, index=None, header=True)

    print('finish')
    writer.save()

# organize part accuracy result

class part_accuracy:

    def __init__(self, consensus_qoq, lightgbm_qoq):
        # num_col = ['medest', 'meanest', 'actual_ibes', 'actual', 'lightgbm_result']
        self.result_qoq = pd.merge(consensus_qoq, lightgbm_qoq, on=['datacqtr', 'gvkey', 'qcut', 'y_type']) #.filter(num_col)
        self.result_qoq['same_{}'.format(consensus)] = self.result_qoq[consensus] == self.result_qoq['lightgbm_result']

    def accu_calculation(self, df, col, cr):
        cd_accu = {}
        sub_result = df.loc[df[col] == cr]
        cd_accu['{} = {}_ibes'.format(col, cr)] = accuracy_score(sub_result[consensus], sub_result['actual_ibes'])
        cd_accu['{} = {}_lgb'.format(col, cr)] = accuracy_score(sub_result['lightgbm_result'], sub_result['actual'])
        cd_accu['{} = {}_len'.format(col, cr)] = len(sub_result)
        return cd_accu

    def eval_conditions(self, df):
        qoq_results = {}

        for TF in [True, False]:
            qoq_results.update(self.accu_calculation(df, 'same_{}'.format(consensus), TF))

        for no_group in range(q):
            qoq_results.update(self.accu_calculation(df, 'lightgbm_result', no_group))

        return qoq_results


    def eval_all_qtr(self):

        date_concat = []

        for date in set(self.result_qoq['datacqtr']):

            df_qtr = self.result_qoq.loc[self.result_qoq['datacqtr']==date]
            eval_result = self.eval_conditions(df_qtr)

            df = pd.DataFrame.from_records(eval_result, index=[0]).transpose()[0].reset_index(drop=False)
            df['accuracy_type'] = [x.rsplit('_', 1)[1] for x in df['index']]
            df['criteria'] = [x.rsplit('_', 1)[0] for x in df['index']]

            df_pivot = df.pivot_table(index='criteria', columns='accuracy_type', values=0).reset_index(drop=False)
            df_pivot['testing_period'] = date
            df_pivot['y_type'] = list(set(df_qtr['y_type']))[0]
            df_pivot = df_pivot.filter(['y_type', 'testing_period', 'criteria', 'ibes', 'lgb', 'len'])
            df_pivot.columns = ['y_type', 'testing_period', 'criteria', 'consensus_accuracy', 'lightgbm_accuracy', 'length']

            df_pivot['consensus_overall'] = accuracy_score(df_qtr[consensus], df_qtr['actual_ibes'])
            df_pivot['lightgbm_overall'] = accuracy_score(df_qtr['lightgbm_result'], df_qtr['actual'])
            df_pivot = df_pivot.sort_values(by=['testing_period'])

            date_concat.append(df_pivot)

        result_df_csv = pd.concat(date_concat, axis=0)

        return result_df_csv


def main_part_accuracy():

    for q in [3,6,9]:

        lightgbm_details = pd.read_csv('lightgbm_details_{}.csv'.format(q))
        consensus_qoq = pd.read_csv('consensus_detail_qoq{}_ibes.csv'.format(q))
        consensus_yoyr = pd.read_csv('consensus_detail_yoyr{}_ibes.csv'.format(q))
        print(lightgbm_details.shape, consensus_qoq.shape, consensus_yoyr.shape)

        lightgbm_qoq = lightgbm_details.loc[lightgbm_details['y_type']=='qoq']
        lightgbm_yoyr = lightgbm_details.loc[lightgbm_details['y_type']=='yoyr']

        # lightgbm_qoq.to_csv('comb_results_qoq{}.csv'.format(q), index=False)
        # lightgbm_yoyr.to_csv('comb_results_yoyr{}.csv'.format(q), index=False)

    for consensus in ['medest', 'meanest']:
        result_df = []
        result_df.append(part_accuracy(consensus_qoq, lightgbm_qoq))
        result_df.append(part_accuracy(consensus_yoyr, lightgbm_yoyr))
        result_df_csv = pd.concat(result_df, axis=0)
        print(result_df_csv)
        result_df_csv.to_csv('conditional_accuracy_result{}_{}.csv'.format(q, consensus), index=False)

def label_industry(csv_name):
    # raw = pd.read_csv('/Users/Clair/PycharmProjects/HKP_ML_DL/Preprocessing/raw/raw.csv')
    # print(raw.columns)
    # print(raw.columns.to_list())
    # exit(0)

    lgb = pd.read_csv('comb_results_{}.csv'.format(csv_name))
    ibes = pd.read_csv('consensus_detail_{}_ibes.csv'.format(csv_name))
    df = pd.merge(lgb, ibes, on=['gvkey', 'datacqtr']).filter(['gvkey', 'datacqtr', 'actual', 'lightgbm_result',
                                                               'medest', 'meanest', 'actual_ibes'])
    df['datacqtr'] = pd.to_datetime(df['datacqtr'],format='%Y-%m-%d')

    info = pd.read_csv('/Users/Clair/PycharmProjects/HKP_ML_DL/Preprocessing/raw/raw.csv',
                       usecols=['gvkey', 'datadate', 'sic', 'conm'])

    gvkey_comn = info[['gvkey', 'conm']].drop_duplicates()
    # print(gvkey_comn)

    # print(info['datadate'])
    info.columns = ['gvkey', 'datacqtr', 'conm','sic']

    info['datacqtr'] = pd.to_datetime(info['datacqtr'],format='%Y%m%d')

    # print(info['datacqtr'])
    df = pd.merge(df, info, on=['gvkey', 'datacqtr'])
    df['year'] = df['datacqtr'].dt.strftime('%Y')
    df['month'] = df['datacqtr'].dt.strftime('%m')

    def accu_calculation(df):
        cd_accu = {}
        cd_accu['consensus_accuracy'] = accuracy_score(df['medest'], df['actual_ibes'])
        cd_accu['lightgbm_accuracy'] = accuracy_score(df['lightgbm_result'], df['actual'])
        cd_accu['length'] = len(df)
        return cd_accu

    df['sic'] = df['sic'].astype(int)
    sic_name = pd.read_csv('sic_name1.csv')
    df = pd.merge(df, sic_name, on=['sic'], how='left')

    total_accuracy = {}

    for name, g in df.groupby(['division']):
        total_accuracy[name] = accu_calculation(g)
    total_accuracy['total'] = accu_calculation(df)


    df_result = pd.DataFrame.from_dict(total_accuracy).transpose().reset_index(drop=False)
    # df_result.columns = ['sic','consensus_accuracy','lightgbm_accuracy','length']
    df_result['diff'] = df_result['lightgbm_accuracy'] - df_result['consensus_accuracy']
    # df_result = df_result.sort_values(by='diff',ascending=False)
    df_result.to_csv('result_by_type_{}_sicd.csv'.format(csv_name), index=False)

    # conm_accuracy = {}
    # for name, g in df.groupby('gvkey'):
    #     conm_accuracy[name] = accu_calculation(g)
    #
    # df_result_conm = pd.DataFrame.from_dict(conm_accuracy).transpose().reset_index(drop=False)
    # df_result_conm.columns = ['gvkey','consensus_accuracy','lightgbm_accuracy','length']
    # df_result_conm = pd.merge(df_result_conm, gvkey_comn, on=['gvkey'], how='left')
    # df_result_conm['diff'] = df_result_conm['lightgbm_accuracy'] - df_result_conm['consensus_accuracy']
    # df_result_conm = df_result_conm.sort_values(by='diff',ascending=False)
    # df_result_conm.to_csv('result_by_type_{}_conm.csv'.format(csv_name), index=False)

if __name__ == '__main__':

    os.chdir('/Users/Clair/PycharmProjects/HKP_ML_DL/Preprocessing/raw/ibes/ibes_new/details')

    for i in ['qoq3','qoq6','qoq9','yoyr3','yoyr6','yoyr9']:
        print(i)
        label_industry(i)



    # # lbs
    #
    # q = 3
    #
    # lightgbm_details = pd.read_csv('lightgbm_details_{}.csv'.format(q))
    # consensus_qoq = pd.read_csv('consensus_detail_qoq{}_ibes.csv'.format(q))
    # consensus_yoyr = pd.read_csv('consensus_detail_yoyr{}_ibes.csv'.format(q))
    # print(lightgbm_details.shape, consensus_qoq.shape, consensus_yoyr.shape)
    #
    # lightgbm_qoq = lightgbm_details.loc[lightgbm_details['y_type']=='qoq']
    # lightgbm_yoyr = lightgbm_details.loc[lightgbm_details['y_type']=='yoyr']
    #
    # # lightgbm_qoq.to_csv('comb_results_qoq{}.csv'.format(q), index=False)
    # # lightgbm_yoyr.to_csv('comb_results_yoyr{}.csv'.format(q), index=False)
    #
    # for consensus in ['medest', 'meanest']:
    #     result_df = []
    #     result_df.append(part_accuracy(consensus_qoq, lightgbm_qoq).eval_all_qtr())
    #     result_df.append(part_accuracy(consensus_yoyr, lightgbm_yoyr).eval_all_qtr())
    #     result_df_csv = pd.concat(result_df, axis=0)
    #     print(result_df_csv)
    #     result_df_csv.to_csv('conditional_date_result{}_{}.csv'.format(q, consensus), index=False)


