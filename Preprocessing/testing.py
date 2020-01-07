import pandas as pd
a = pd.DataFrame(range(8))
print(a)
print(a.rolling(4, min_periods=1, axis=0).sum().shift(-4))