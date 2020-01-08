def unstack_selection():
    import pandas as pd
    from sqlalchemy import create_engine

    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)
    format_map = pd.read_sql('SELECT name, yoy, qoq, log, nom FROM format_map', engine, index_col=['name'])
    format_map_df = format_map.unstack().reset_index()
    format_map_df.columns = ['format', 'name', 'selection']
    format_map_df.to_csv('format_map_selection.csv')

if __name__ == "__main__":
