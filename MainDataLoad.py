import numpy as np
import pandas as pd
from sqlalchemy import Column, Date, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import text
from sqlalchemy import create_engine



def load_main_data():

    # load the SQL table, return a Python DataFrame

    engine = create_engine('postgresql://postgres:DLvalue123@hkpolyu-dl-value.c1lrltigx0e7.us-east-1.rds.amazonaws.com/postgres')

    with engine.connect() as db_connection:
        main_abs = db_connection.execute("SELECT * FROM public.main_abs LIMIT 100")

    main_abs_list = [[r[i] for i in range(len(main_abs.keys()))] for r in main_abs]
    main_abs_df = pd.DataFrame(main_abs_list, columns = main_abs.keys())

    return main_abs_df


def load_data():

    # load the SQL table, return a Python DataFrame

    engine = create_engine(
        'postgresql://postgres:DLvalue123@hkpolyu-dl-value.c1lrltigx0e7.us-east-1.rds.amazonaws.com/postgres')

    with engine.connect() as db_connection:
        main_abs = db_connection.execute("SELECT * FROM public.main_abs LIMIT 100")

    main_abs_list = [[r[i] for i in range(len(main_abs.keys()))] for r in main_abs]
    main_abs_df = pd.DataFrame(main_abs_list, columns=main_abs.keys())

    return main_abs_df


def insert_row():

    # insert one row

    engine = create_engine('postgresql://postgres:DLvalue123@hkpolyu-dl-value.c1lrltigx0e7.us-east-1.rds.amazonaws.com/postgres')

    with engine.connect() as db_connection:
        db_connection.execute("INSERT INTO public.abbreviation (abbreviation) VALUES ('text')")

    return 0

if __name__ == "__main__":

    x = load_data()
    print(x)

    #create_table()
