import pandas as pd

if __name__ == "__main__":
    import os
    os.chdir('/Users/Clair/PycharmProjects/HKP_ML_DL')

    macro = pd.read_csv('Macro_Data.csv')

    macro.columns = ['Date', 'Long Term Rate', 'GDP', 'PCE', 'LEI', 'Short Term Market Rate',
                     'Central Bank Target Rate', 'Unemployment', 'Philly Fed', 'CPI', 'S&P']

    macro['Date'] = pd.to_datetime(macro['Date'])
    macro = macro.sort_values('Date', ascending=True)
    macro['datacqtr'] = macro['Date'].apply(
        lambda x: pd.Period(x, freq='Q-DEC').to_timestamp(how='end').strftime('%Y-%m-%d'))
    macro['GDP_qoq'] = macro['GDP'] / macro['GDP'].shift(1) - 1
    macro['GDP_yoy'] = macro['GDP'] / macro['GDP'].shift(4) - 1
    macro['S&P_qoq'] = macro['S&P'] / macro['S&P'].shift(1) - 1
    macro['S&P_yoy'] = macro['S&P'] / macro['S&P'].shift(4) - 1
    macro['PCE_qoq'] = macro['PCE'] / macro['PCE'].shift(1) - 1
    macro['PCE_yoy'] = macro['PCE'] / macro['PCE'].shift(4) - 1
    macro_main = macro[
        ['datacqtr', 'GDP_qoq', 'GDP_yoy', 'PCE_qoq', 'PCE_yoy', 'Central Bank Target Rate', 'Short Term Market Rate',
         'Long Term Rate', 'S&P_qoq', 'S&P_yoy']]
    macro_main = macro_main.reset_index()
    del macro_main['index']

    macro_main.to_csv("macro_main.csv", index = False)

