import datetime as dt

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine


def main(sql_version=False):

    if sql_version is True:  # sql version read TABLE from Postgre SQL
        db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
        engine = create_engine(db_string)
        main = pd.read_sql('SELECT * FROM main', engine)
    else:  # local version read TABLE from local csv files -> faster
        main = pd.read_csv('Hyperopt_LightGBM/main.csv')
        engine = None
        print('local version running - main')

    main['datacqtr'] = pd.to_datetime(main['datacqtr'],format='%Y-%m-%d')
    main = main.dropna(how='any')
    main = main.loc[(main['datacqtr']<= dt.datetime(2019, 1, 1))&(main['datacqtr']>= dt.datetime(1983, 1, 1))].reset_index(drop=True)

    # standardize based on entire table
    scaler = StandardScaler().fit(main.iloc[:, 2:])
    main.iloc[:, 2:] = scaler.transform(main.iloc[:, 2:])

    # pca on entire table
    pca = PCA(n_components=0.75)
    main_pca = pca.fit_transform(main.iloc[:, 2:])
    main = pd.concat([main.iloc[:,:2],pd.DataFrame(main_pca)], axis=1)

    # start construct 3d array
    arr=[]
    for name, g in main.groupby('datacqtr'):
        print(name)
        g = g.set_index('gvkey')
        df = pd.DataFrame(index=list(set(main['gvkey'])))
        df0 = pd.merge(df, g, left_index=True, right_index=True,how='outer').filter(main.columns[2:])
        arr.append(np.array(df0))

    arr3d = np.array(arr)   # arr3d is the 3d array
    print(arr3d)
    print(arr3d.shape)
    return arr3d

if __name__ == '__main__':
    arr3d = main()
