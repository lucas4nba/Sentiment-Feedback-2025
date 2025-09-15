import numpy as np, pandas as pd
from pathlib import Path
import statsmodels.api as sm

def fit_spec(df):
    y = df['RET']
    X = sm.add_constant(df[['UMCSENT', 'VIXCLS']])  # Using available columns
    return sm.OLS(y, X).fit().params

def time_block_boot(df, date_col='DATE', B=500, block_len=6, seed=42):
    rng = np.random.default_rng(seed)
    months = sorted(df[date_col].unique())
    n = len(months)
    params = []
    for _ in range(B):
        idx = []
        i = 0
        while i < n:
            start = rng.integers(0, n - block_len + 1)
            idx += months[start:start+block_len]
            i += block_len
        df_b = df[df[date_col].isin(idx[:n])]
        params.append(fit_spec(df_b))
    return pd.DataFrame(params)

df = pd.read_parquet('panel_ready.parquet')
DF = time_block_boot(df)
DF.to_csv('outputs/panel/timeblock_params.csv', index=False)
print(DF.describe().T[['mean','std']])
