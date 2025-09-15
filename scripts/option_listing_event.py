import pandas as pd, numpy as np
import statsmodels.api as sm

events = pd.read_csv('data/events/first_option_listing.csv')  # permno, list_date (YYYY-MM)
panel = pd.read_parquet('panel_ready.parquet')   # includes outcome y and date ym

def event_window(df, permno, t0, k=12):
    sub = df[df.PERMNO==permno].copy()
    sub['relm'] = (sub['DATE'] - t0).dt.days/30.44
    return sub[(sub.relm>=-k)&(sub.relm<=k)]

rows=[]
for _,e in events.iterrows():
    w = event_window(panel, e.permno, pd.to_datetime(e.list_date))
    rows.append(w)
EW = pd.concat(rows, ignore_index=True)

EW['post'] = (EW['relm']>=0).astype(int)
X = sm.add_constant(EW[['relm','post','relm']].assign(relm_post=EW['relm']*EW['post']))
y = EW['RET']  # Using RET as outcome variable
fit = sm.OLS(y, X).fit(cov_type='HC1')
fit.save('outputs/events/opt_local_linear.pkl')
print(fit.summary())
