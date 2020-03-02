import pandas as pd
from sqlalchemy import create_engine



def load_main_data():

    # load the main dataset, return a Python DataFrame

    engine = create_engine('postgresql://postgres:DLvalue123@hkpolyu-dl-value.c1lrltigx0e7.us-east-1.rds.amazonaws.com/postgres')

    with engine.connect() as db_connection:
        main_data = db_connection.execute("SELECT * FROM public.main_abs")

    main_data_list = [[r[i] for i in range(len(main_data.keys()))] for r in main_data]
    main_data_df = pd.DataFrame(main_data_list, columns = main_data.keys())

    return main_data_df


def load_data(table_name):

    # load a SQL table, return as Python DataFrame

    engine = create_engine(
        'postgresql://postgres:DLvalue123@hkpolyu-dl-value.c1lrltigx0e7.us-east-1.rds.amazonaws.com/postgres')

    with engine.connect() as db_connection:
        data = db_connection.execute("SELECT * FROM "+ table_name)

    data_list = [[r[i] for i in range(len(data.keys()))] for r in data]
    data_df = pd.DataFrame(data_list, columns=data.keys())

    return data_df


def insert_row(table_name):

    # insert one row in a SQL table

    engine = create_engine('postgresql://postgres:DLvalue123@hkpolyu-dl-value.c1lrltigx0e7.us-east-1.rds.amazonaws.com/postgres')

    with engine.connect() as db_connection:
        db_connection.execute("INSERT INTO public.abbreviation (abbreviation) VALUES ('text')")

    return 0

if __name__ == "__main__":

    x = load_main_data()
    print(x)
