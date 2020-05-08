import pandas as pd
from sqlalchemy import create_engine

def check_print(df_list, sort=True):
    import os
    df = pd.concat(df_list, axis=1)
    if sort==True:
        col = ['gvkey','datacqtr'] + [x for x in sorted(df.columns) if x not in ['gvkey','datacqtr']]
        df = df.reindex(col, axis=1)
    df.to_csv('check.csv')

    os.system("open -a '/Applications/Microsoft Excel.app' 'check.csv'")
    exit(0)

def remove_outliers(y_type, dep, by):  # remove outlier from both 'yoy' & 'qoq' y_type
    y_series = dep[y_type].dropna()

    if by == 'stv': # remove outlier by standard deviation
        y_clean = y_series.where(np.abs(stats.zscore(y_series)) < 5).dropna()
        idx = y_clean.index

    elif by == 'quantile': # remove outlier by top/bottom percentage
        Q1 = y_series.quantile(0.01)
        Q3 = y_series.quantile(0.99)
        y_clean = y_series.mask((y_series < Q1) | (y_series > Q3)).dropna()
        idx = y_clean.index
    else:
        print("Error: 'by' can only be 'stv' or 'quantile'.")
    return dep.loc[idx]

def delete_high_zero_row(missing_dict):
    df_zero_series = pd.concat(missing_dict, axis = 1).sum(axis=1)
    print(df_zero_series)

    def print_csv(df_zero_series):
        from collections import Counter

        c = Counter(df_zero_series)
        csum = 0
        missing_dict = {}
        for missing, count in sorted(c.items()):
            csum += count
            missing_dict[missing] = csum

        df = pd.DataFrame.from_dict(missing_dict, orient='index', columns=['count'])
        df['%_count'] = df['count'] / 333325
        df['changes'] = df['%_count'].sub(df['%_count'].shift(1))

        def double_plot(df, axis1, axis2):

            # plot dataframe by two columns, axis1 = 'red', axis2 = 'blue'
            import matplotlib.pyplot as plt

            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            ax1.plot(df[axis1], color='tab:red')
            ax1.tick_params(axis='y', labelcolor='tab:red')

            ax2.plot(df[axis2], color='tab:blue')
            ax2.tick_params(axis='y', labelcolor='tab:red')

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.show()
            plt.savefig('double_plot_delete_row.png')

        double_plot(df, '%_count', 'changes')

        df.to_csv('missing_below_threshold.csv')
        id = df.loc[df['%_count'] < 0.9].sort_values(by=['%_count']).tail(1).index
        print(id)
        return id

    df = df.loc[df_zero_series < print_csv(df_zero_series)]

    return df

def whole_print(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)

def identify_types():
    meta = MetaData()
    table = Table('lightgbm_results', meta, autoload=True, autoload_with=engine)
    columns = table.c
    types = {}
    for c in columns:
        types[c.name] = c.type
    # types.pop('early_stopping_rounds')
    # types.pop('num_boost_round')
    return types

def eda_hist(df, name='temp'):
    from matplotlib import pyplot as plt
    import math
    fig = plt.figure(figsize=(50, 50), dpi=80)
    plt.rcParams.update({'font.size': 6})
    k = 1
    for col in df.columns.to_list():
        n = math.ceil(len(df.columns) ** 0.5)
        axis1 = fig.add_subplot(n, n, k)
        axis1.hist(df.loc[df[col].notnull(), col], density=True, bins=50)
        axis1.set_title(col, fontsize=60)
        print(col, k)
        k += 1
    fig.tight_layout()
    fig.savefig(name + '.png')


def unstack_selection():
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    format_map = pd.read_sql('SELECT name, yoy, qoq, log, nom FROM format_map', engine, index_col=['name'])
    format_map_df = format_map.unstack().reset_index()
    format_map_df.columns = ['format', 'name', 'selection']
    format_map_df.to_csv('format_map_selection.csv')

def Timestamp(df):
    df['datacqtr'] = df['datacqtr'].apply(lambda x: pd.Period(x, freq='Q-DEC').to_timestamp(how='end').strftime('%Y-%m-%d'))
    return df

def save_load_dict(type, dict = None, name = 'temp'):
    import json
    if type == 'save':
        with open(name + '.json', 'w') as fp:
            json.dump(dict, fp)

    elif type == 'load':
        with open(name + '.json', 'r') as fp:
            data = json.load(fp)
            return data
    else:
        print('ERROR: No dictionary found')

def check_correlation_delete(df, threshold=0.9):
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

if __name__ == "__main__":
    ipo = pd.read_csv('ipo.csv').groupby('gvkey').head(1).filter(['gvkey','fyear'])
    ipo.columns = ['gvkey','start_year']

    df = pd.read_csv('main_forward.csv')
    df = pd.merge(df, ipo, on = ['gvkey'], how = 'left')
    df['fyear'] = [x[:4] for x in df['datacqtr']]
    df['age'] = df['fyear'] - df['start_year']
    print(df)