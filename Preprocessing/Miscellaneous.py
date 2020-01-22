import pandas as pd
from sqlalchemy import create_engine


def unstack_selection():


    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)
    format_map = pd.read_sql('SELECT name, yoy, qoq, log, nom FROM format_map', engine, index_col=['name'])
    format_map_df = format_map.unstack().reset_index()
    format_map_df.columns = ['format', 'name', 'selection']
    format_map_df.to_csv('format_map_selection.csv')

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
