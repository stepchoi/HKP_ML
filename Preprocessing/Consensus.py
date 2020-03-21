import numpy as np
import pandas as pd
from LoadData import convert_to_float32
from Miscellaneous import Timestamp
from sqlalchemy import create_engine

def main():
    consensus = pd.read_csv('postgres_public_zz_zacks_import.csv')
    comp_list = pd.read_csv('ZACKS_Company List.csv')


if __name__ == '__main__':
    main()