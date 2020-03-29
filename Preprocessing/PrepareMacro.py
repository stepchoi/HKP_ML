import pandas as pd
from sqlalchemy import create_engine

def main():
    try:
        macro = pd.read_csv('macro_raw.csv')
    except:
        db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
        engine = create_engine(db_string)
        macro = pd.read_sql('SELECT * FROM macro_raw', engine)

    macro.columns = ['Date', 'Long Term Rate', 'GDP', 'PCE', 'LEI', 'Short Term Market Rate',
                     'Central Bank Target Rate', 'Unemployment', 'Philly Fed', 'CPI', 'S&P']

    macro['Date'] = pd.to_datetime(macro['Date'])
    macro = macro.sort_values('Date', ascending=True)
    macro['datacqtr'] = macro['Date'].apply(
        lambda x: pd.Period(x, freq='Q-DEC').to_timestamp(how='end').strftime('%Y-%m-%d'))

    macro['GDP_qoq'] = macro['GDP'].pct_change(periods=1)
    macro['GDP_yoy'] = macro['GDP'].pct_change(periods=4)
    macro['PCE_qoq'] = macro['PCE'].pct_change(periods=1)
    macro['PCE_yoy'] = macro['PCE'].pct_change(periods=4)
    macro['S&P_qoq'] = macro['S&P'].pct_change(periods=1).shift(-1)
    macro['S&P_yoy'] = macro['S&P'].pct_change(periods=4).shift(-1)
    interest_rate = ['Central Bank Target Rate', 'Short Term Market Rate', 'Long Term Rate']
    print(macro[interest_rate])
    macro[interest_rate] = macro[interest_rate].shift(-1)
    print(macro[interest_rate])

    macro_main = macro[
        ['datacqtr', 'GDP_qoq', 'GDP_yoy', 'PCE_qoq', 'PCE_yoy', 'Central Bank Target Rate', 'Short Term Market Rate',
         'Long Term Rate', 'S&P_qoq', 'S&P_yoy']]

    macro_main = macro_main.reset_index()
    del macro_main['index']

    macro_main.to_csv("macro_main.csv", index=False)

if __name__ == "__main__":
    main()



