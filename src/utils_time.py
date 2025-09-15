import pandas as pd

def to_month_end(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, errors="coerce", utc=False).dt.tz_localize(None)
    return d.dt.to_period("M").dt.to_timestamp("M")

def monthly_avg(df, date_col, value_cols):
    g = (df.assign(M=to_month_end(df[date_col]))
           .groupby("M", as_index=False)[value_cols].mean(numeric_only=True))
    g.rename(columns={"M":"DATE"}, inplace=True)
    return g

def month_end_sample(df, date_col, keep_cols):
    """Pick last obs in each month (trading month if dates are trading days)."""
    x = df.copy()
    x["DATE"] = to_month_end(x[date_col])
    # keep last row per month by original timestamp order
    x["_idx"] = range(len(x))
    last = x.sort_values([ "DATE", "_idx"]).groupby("DATE").tail(1)
    return last[["DATE", *keep_cols]].reset_index(drop=True)
