import numpy as np, pandas as pd
from pathlib import Path
import statsmodels.api as sm

def fit_spec(df):
    y = df['RET']
    X = sm.add_constant(df[['UMCSENT', 'VIXCLS']])  # Using available columns
    mod = sm.OLS(y, X).fit(cov_type='HC1')
    return mod.params, mod.bse

def jackknife(df, id_col='PERMNO', bins=10):
    # assign bins deterministically
    ids = df[id_col].drop_duplicates().sort_values().to_numpy()
    rng = np.random.default_rng(1)
    rng.shuffle(ids)
    splits = np.array_split(ids, bins)
    params = []
    for i in range(bins):
        keep = ~df[id_col].isin(splits[i])
        p, _ = fit_spec(df[keep])
        params.append(p)
    P = pd.DataFrame(params)
    theta = P.mean()
    # jackknife se
    se = ((bins-1)/bins * ((P - theta)**2).sum())**0.5
    return theta, se

df = pd.read_parquet('panel_ready.parquet')
theta, se = jackknife(df)
out = pd.DataFrame({'coef': theta, 'se_jack': se})
out.to_csv('outputs/panel/jackknife_se.csv')
print(out)
