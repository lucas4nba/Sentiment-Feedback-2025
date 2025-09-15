#!/usr/bin/env python3
"""
05b_portfolio_sorts.py
Monthly quintile sorts by Z_i, evaluate next-month returns (EW).

Usage:
  python 05b_portfolio_sorts.py --panel ./panel_ready.parquet --z-col retail_intensity --out ./tables_figures
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

ID_CAND = ["permno","permco","gvkey","secid","id","ticker","symbol"]
DATE_HINTS = ["date","month","caldt","yyyymm"]

def find_id(df):
    for c in ID_CAND:
        for col in df.columns:
            if col.lower() == c:
                return col
    counts = {c: df[c].nunique() for c in df.columns if df[c].dtype.kind in "iuOS"}
    return max(counts, key=counts.get)

def find_date(df):
    for c in df.columns:
        cl = c.lower()
        if any(h in cl for h in DATE_HINTS):
            try:
                return pd.to_datetime(df[c], errors="raise")
            except Exception:
                pass
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return pd.to_datetime(df[c])
    raise ValueError("No date column detected.")

def autodetect_ret(df):
    for c in df.columns:
        lc = c.lower()
        if "ret" in lc and "term" not in lc:
            return c
    raise ValueError("No return column detected. Use --ret-col.")

def newey_west_mean(y, lags):
    y = pd.Series(y).dropna()
    X = np.ones((len(y),1))
    fit = sm.OLS(y.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(y.mean()), float(fit.tvalues[0])

def main(panel_path: Path, z_col: str, ret_col: str, out_dir: Path, positive_only: bool = False, value_weighted: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(panel_path)

    id_col = find_id(df)
    date_parsed = find_date(df)
    df["_date"] = date_parsed.dt.to_period("M").dt.to_timestamp("M")
    if ret_col is None:
        ret_col = autodetect_ret(df)

    if z_col is None or z_col not in df.columns:
        raise ValueError("Please pass --z-col (e.g., retail_intensity, turnover, amihud, mcap).")

    # Build next-month return
    df["_y"] = df[ret_col].shift(-1)

    # Quintiles each month
    def tag_quintile(s):
        return pd.qcut(s, 5, labels=[1,2,3,4,5], duplicates="drop")
    df["_q"] = df.groupby("_date")[z_col].transform(tag_quintile)

    # Market cap for value weighting (if requested)
    if value_weighted:
        prc_col = None
        shr_col = None
        # Auto-detect price and shares columns
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ["prc", "price"] and prc_col is None:
                prc_col = col
            elif col_lower in ["shrout", "shares_out", "shrout"] and shr_col is None:
                shr_col = col
        
        if prc_col is None or shr_col is None:
            raise ValueError(f"Need price and shares columns for value weighting. Found: PRC={prc_col}, SHROUT={shr_col}")
        
        # Compute market cap
        sh_adj = 1000.0 if df[shr_col].median() < 1e8 else 1.0
        df["_mcap"] = df[prc_col].abs() * df[shr_col] * sh_adj
        print(f"[VW] Using {prc_col} and {shr_col} for market cap (adjustment factor: {sh_adj})")

    # After building df with _date and Z quintiles, bring in the shock column and restrict if flag is set:
    shock_candidates = [c for c in df.columns if "shock" in c.lower()]
    if "sentiment_shock" in df.columns:
        shock_col = "sentiment_shock"
    elif shock_candidates:
        shock_col = shock_candidates[0]
    else:
        # Need to merge shock data from time series dataset
        print("No shock columns found, merging from ts_dataset.parquet...")
        try:
            ts_data = pd.read_parquet("ts_dataset.parquet")
            shock_cols = [c for c in ts_data.columns if "shock" in c.lower()]
            if not shock_cols:
                raise ValueError("No shock columns found in ts_dataset.parquet")
            shock_col = shock_cols[0]
            print(f"Using shock column: {shock_col}")
            
            # Prepare time series data for merging
            if 'DATE' in ts_data.columns:
                ts_data['DATE'] = pd.to_datetime(ts_data['DATE'])
                ts_merge = ts_data[['DATE', shock_col]].copy()
                ts_merge['_date_merge'] = ts_merge['DATE'].dt.to_period("M").dt.to_timestamp("M")
            else:
                raise ValueError("No DATE column found in ts_dataset.parquet")
            
            # Merge shock data into panel
            df['_date_merge'] = df['_date']
            df = df.merge(ts_merge[['_date_merge', shock_col]], on='_date_merge', how='left')
            df = df.drop('_date_merge', axis=1)
            print(f"Successfully merged shock data: {df[shock_col].notna().sum()} / {len(df)} non-missing")
            
        except Exception as e:
            raise ValueError(f"Error merging shock data: {e}")
    if positive_only:
        # Keep months with positive or above-median shock
        month_med = df.groupby("_date")[shock_col].first().median()
        pos_months = df.groupby("_date")[shock_col].first()
        pos_months = pos_months[pos_months > month_med].index  # use >0 instead if preferred
        df = df[df["_date"].isin(pos_months)].copy()
        print(f"[PORT] Filtering to {len(pos_months)} months with shock > median ({month_med:.4f})")

    # Monthly return per quintile (EW or VW)
    def monthly_ret(g, vw=False):
        """Compute monthly portfolio return - EW or VW"""
        if vw and "_mcap" in g.columns:
            w = g["_mcap"].fillna(0)
            w = w / w.sum() if w.sum() > 0 else None
            return (g["_y"] * w).sum() if w is not None else g["_y"].mean()
        else:
            return g["_y"].mean()

    port = (df.dropna(subset=["_y","_q"])
              .groupby(["_date","_q"]).apply(lambda g: monthly_ret(g, vw=value_weighted))
              .unstack("_q").sort_index())

    # Compute time-series average and NW t-stats
    L = int(np.floor(4 * (len(port) / 100.0) ** (2.0 / 9.0)))
    L = max(L, 1)
    rows = []
    for q in port.columns:
        mean, t = newey_west_mean(port[q], L)
        rows.append({"port": f"Q{int(q)}", "mean": mean, "t": t})
    # Spread Q5 - Q1
    spread = port[5] - port[1]
    mean, t = newey_west_mean(spread, L)
    rows.append({"port": "Q5-Q1", "mean": mean, "t": t})
    res = pd.DataFrame(rows)

    # Set output filenames based on flags
    suffix = "_pos" if positive_only else ""
    weight_suffix = "_vw" if value_weighted else ""
    csv = out_dir / f"portfolio_sorts_q{suffix}{weight_suffix}.csv"
    tex = out_dir / f"portfolio_sorts_q{suffix}{weight_suffix}.tex"
    png = out_dir / f"portfolio_sorts_q{suffix}{weight_suffix}.png"
    
    res.to_csv(csv, index=False)
    try:
        tex.write_text(res.to_latex(index=False, float_format="%.4f"))
    except Exception:
        pass

    # Plot levels + spread
    ax = port.mean().plot(marker="o")
    pos_suffix = " (positive shock months only)" if positive_only else ""
    weight_type = "VW" if value_weighted else "EW"
    ax.set_title(f"{weight_type} next-month returns by Z quintile (avg across months){pos_suffix}")
    ax.set_xlabel("Quintile"); ax.set_ylabel("Mean return")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print(f"[PORT] wrote {csv}")
    print(res)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, default=Path("./panel_ready.parquet"))
    ap.add_argument("--z-col", type=str, default=None)
    ap.add_argument("--ret-col", type=str, default=None)
    ap.add_argument("--out", type=Path, default=Path("./tables_figures"))
    ap.add_argument("--positive-only", action="store_true", help="Use only months with positive Shock_t")
    ap.add_argument("--value-weighted", action="store_true", help="Use market cap weighting instead of equal weighting")
    args = ap.parse_args()
    main(args.panel, args.z_col, args.ret_col, args.out, args.positive_only, args.value_weighted)
