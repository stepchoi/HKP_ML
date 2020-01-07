if __name__ == '__main__':

    from sqlalchemy import create_engine
    import pandas as pd
    from PrepareDatabase import drop_nonseq

    # import engine, select variables, import raw database
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    dep = pd.read_csv('raw.csv', usecols = ['gvkey', 'datacqtr', 'epspxq'])
    dep = drop_nonseq(dep)

    dep['next1_abs'] = dep.groupby('gvkey').apply(lambda x: x['epspxq'].shift(1)).to_list()
    dep['epspxq_qoq'] = dep['next1_abs'].div(dep['epspxq']).sub(1) # T1/T0
    dep['past4_abs'] = dep.groupby('gvkey').apply(lambda x: x['epspxq'].rolling(4, min_periods=4).sum()).to_list() # rolling past 4 quarter
    dep['next4_abs'] = dep.groupby('gvkey').apply(lambda x: x['past4_abs'].shift(-4)).to_list() # rolling next 4 quarter
    dep['epspxq_yoy'] = dep['next4_abs'].div(dep['past4_abs']).sub(1) # T4/T0
    del dep['datacqtr_no']
    print(dep)
    dep.to_csv('epspxq.csv', index=False)
