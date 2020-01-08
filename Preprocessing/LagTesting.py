def check_correlation(df, threshold=0.9):
    # find high correlated items -> excel

    ex_corr_col = []

    def high_corr(df, threshold=0.9):
        corr = df.corr().abs()
        so = corr.unstack().reset_index()
        print(so)
        # so = so.sort_values(kind="quicksort", ascending=False).to_frame().reset_index()
        so.columns = ['v1', 'v2', 'corr']
        so = so.loc[(so['v1'] != so['v2']) & (so['corr'] > threshold)].drop_duplicates(subset=['v1', 'v2'])
        return so

    def del_corr():
        # Set of all the names of deleted columns
        from collections import Counter
        items = high_corr_df['v1'].to_list() + high_corr_df['v2'].to_list()
        items_occurence = Counter(items).keys()

        col_corr = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                    colname = corr_matrix.columns[i]  # getting the name of column
                    col_corr.add(colname)
        del_corr = col_corr - set(ex_corr_col)

        # DataFrame of high correlated, description, and values
        del_value = high_corr_df.loc[(high_corr_df['v1'].isin(del_corr)) | (high_corr_df['v2'].isin(del_corr))]
        del_col_ie = pd.merge(del_value, format_map, left_on=['v1'], right_on=['abbreviation'], how='left')
        del_col_ie = pd.merge(del_col_ie, format_map, left_on=['v2'], right_on=['abbreviation'], how='left')
        del_col_ie = del_col_ie.loc[((del_col_ie['abbreviation_x'].isin(del_corr))) &
                                    (~(del_col_ie['abbreviation_y'].isin(del_corr)))].sort_values(
            by='abbreviation_x').iloc[:, 2:]
        return del_corr, del_col_ie

    corr_matrix = df.corr()
    high_corr_df = high_corr(df)
    print(high_corr_df)


def merge_dep_macro(df, dependent_variable='epspxq_nextq_abs'):
    macro = pd.read_sql("SELECT * FROM macro_clean", engine)
    macro_lst = macro.columns[2:]
    dep = pd.read_sql('SELECT gvkeydatafqtr, ' + dependent_variable + ', selected FROM main_dependent', engine)

    df_1 = pd.merge(df, dep, on=['gvkeydatafqtr'], how='left')
    df_1 = pd.merge(df_1, macro, on=['cyear', 'cquarter'], how='left')

    return df_1, macro_lst


def add_lag(df, col):
    def namelag(lst, i):
        return [(k + '_lag' + str(i).zfill(2)) for k in lst]

    for i in range(19):
        print(i)
        df[namelag(col, i + 1)] = df.groupby('gvkey').shift(i + 1)[col]
    return df


def cut_train_test(df, testing_cqtr):
    df_train = df.loc[df['datacqtr_no'].isin(range(testing_cqtr - 20, testing_cqtr, 1))]
    df_test = df.loc[df['datacqtr_no'] == testing_cqtr]
    return df_train, df_test


if __name__ == "__main__":

    import pandas as pd

    # import engine, select variables, import raw database
    try:
        main = pd.read_csv('main_forward.csv')
        engine = None
        print('local version running')
    except:
        from sqlalchemy import create_engine

        db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
        engine = create_engine(db_string)
        main = pd.read_sql('SELECT * FROM main_forward', engine)

    # check_correlation(main.iloc[:,2:])
    del_corr = ['xsgaq_qoq', 'gdwlq_log', 'cogsq_qoq']  # same for both forward & rolling version
    main = main.drop(main[del_corr], axis=1)

    # main = drop_nonseq(main)
    # main, macro_lst = merge_dep_macro(raw)  # add dependent variable & macro variables to main
    #
    # num_col = numeric_dict['YoY'] + numeric_dict['QoQ'] + numeric_dict['Abs'] + numeric_dict['Log']
    # main_lag = add_lag(df, num_col).dropna(how='any', axis=0)
    #
    # pca_dict = {}
    # pca_dict['train'] = {}
    # pca_dict['test'] = {}
    #
    #     for i in range(111):
    #         pass