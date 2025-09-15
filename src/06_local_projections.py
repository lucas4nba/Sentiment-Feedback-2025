#!/usr/bin/env python3
"""
06_local_projections.py
Jordà-style local projections: R_{t→t+h} = a_h + b_h Shock_t + controls + e_{t+h}

Features:
- Computes overlapping cumulative excess returns R_{t→t+h}
- OLS with Newey-West HAC where lag=h-1
- --non_overlapping flag to sample every h months
- Scales coefficients per 1-SD of chosen shock
- Saves results with [h, beta, se, r2, shock_type, overlapping_flag, n_obs]
- Comparison plot for shock_ar1 vs shock_var_orth

Usage:
  python 06_local_projections.py --ts ./ts_dataset.parquet --out ./tables_figures --horizons 1 3 6 12
  python 06_local_projections.py --ts ./ts_dataset.parquet --non_overlapping --shock_type compare
  python 06_local_projections.py --horizons 1 2 4 8 12 24 --non_overlapping --shock_type compare
Requires: pandas, pyarrow, numpy, statsmodels, matplotlib
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def resolve_shock_cols(df, requested):
    """
    Resolve shock column names gracefully.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing shock columns
    requested : str
        Requested shock type: "ar1", "var_orth", "compare", or None
        
    Returns:
    --------
    dict
        Dictionary mapping shock types to column names
    """
    candidates = {
        "ar1": ["UMCSENT_shock_ar1", "shock_ar1"],
        "var_orth": ["UMCSENT_shock_var_orth", "shock_var_orth"],
    }
    
    if requested in ("compare", None):
        kinds = ["ar1", "var_orth"]
    else:
        kinds = [requested]
    
    out = {}
    for k in kinds:
        col = next((c for c in candidates[k] if c in df.columns), None)
        if col is None:
            raise ValueError(f"Missing shock column for {k}. Have: {df.columns.tolist()}")
        out[k] = col
    
    return out  # dict like {"ar1": "UMCSENT_shock_ar1", "var_orth": "UMCSENT_shock_var_orth"}

def newey_west(y, X, lags):
    """Run OLS with Newey-West HAC standard errors"""
    X1 = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X1, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": lags})

def compute_cumulative_returns(returns_series, horizon):
    """
    Compute overlapping cumulative returns R_{t→t+h} = sum(r_{t+1} to r_{t+h})
    
    Parameters:
    -----------
    returns_series : pd.Series
        Monthly excess returns
    horizon : int
        Number of months ahead
        
    Returns:
    --------
    pd.Series
        Cumulative returns from t to t+h
    """
    cumulative_returns = pd.Series(index=returns_series.index, dtype=float)
    
    for t in range(len(returns_series) - horizon):
        # Sum returns from t+1 to t+h (inclusive)
        cum_ret = returns_series.iloc[t+1:t+1+horizon].sum()
        cumulative_returns.iloc[t] = cum_ret
    
    return cumulative_returns

def autodetect_cols(ts: pd.DataFrame, shock_cols_dict=None):
    """Auto-detect sentiment shock and market returns columns"""
    
    # Use resolved shock columns if provided
    if shock_cols_dict:
        shocks = list(shock_cols_dict.values())
        print(f"Using resolved shock columns: {shocks}")
    else:
        # Find shock columns
        shock_cols = [c for c in ts.columns if "shock" in c.lower()]
        if not shock_cols:
            print(f"Available columns: {list(ts.columns)}")
            raise ValueError("No shock columns found in ts_dataset.")
        
        # Use both AR1 and VAR orthogonal shocks for comparison
        shocks = [c for c in shock_cols if 'UMCSENT' in c]  # Focus on UMCSENT shocks
        if not shocks:
            shocks = shock_cols[:2]  # Take first two if UMCSENT not available
    
    # Look for returns column
    ret = None
    for c in ts.columns:
        if "ret" in c.lower() and "term" not in c.lower():
            ret = c
            break
    
    if ret is None:
        print(f"Available columns: {list(ts.columns)}")
        raise ValueError("No returns column found in ts_dataset. Add a 'market_excess_ret' first.")
    
    print(f"Using returns column: {ret}")
    print(f"Using shock column(s): {shocks}")
    return shocks, ret

def create_comparison_plot(res: pd.DataFrame, out_dir: Path, ret_col: str):
    """Create comparison plot between shock_ar1 and shock_var_orth"""
    
    # Check that both "ar1" and "var_orth" exist
    shock_types = res['shock_type'].unique()
    has_ar1 = any('ar1' in s for s in shock_types)
    has_var_orth = any('var_orth' in s for s in shock_types)
    
    if has_ar1 and has_var_orth:
        # Filter for the two main shock types
        ar1_data = res[res['shock_type'].str.contains('ar1', na=False)]
        var_orth_data = res[res['shock_type'].str.contains('var_orth', na=False)]
        
        plt.figure(figsize=(10, 6))
        
        # Plot AR1 shock results
        valid_ar1 = ar1_data.dropna(subset=['beta'])
        if len(valid_ar1) > 0:
            plt.plot(valid_ar1["h"], valid_ar1["beta"], marker="o", label="AR(1) Shock", linewidth=2)
            # Add confidence bands
            plt.fill_between(valid_ar1["h"], 
                            valid_ar1["beta"] - 1.96 * valid_ar1["se"], 
                            valid_ar1["beta"] + 1.96 * valid_ar1["se"], 
                            alpha=0.2)
        
        # Plot VAR orthogonal shock results  
        valid_var = var_orth_data.dropna(subset=['beta'])
        if len(valid_var) > 0:
            plt.plot(valid_var["h"], valid_var["beta"], marker="s", label="VAR Orthogonal Shock", linewidth=2)
            # Add confidence bands
            plt.fill_between(valid_var["h"], 
                            valid_var["beta"] - 1.96 * valid_var["se"], 
                            valid_var["beta"] + 1.96 * valid_var["se"], 
                            alpha=0.2)
        
        plt.axhline(0.0, ls="--", lw=1, alpha=0.6, color='black')
        plt.title(f"Local Projection IRF Comparison: UMCSENT Shocks → {ret_col}")
        plt.xlabel("Horizon (months)")
        plt.ylabel("Beta (1-SD scaled)")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = out_dir / "lp_irf_plot_compare.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to: {plot_path}")
        plt.close()
    else:
        # Single shock type - still create plot but log warning
        print(f"Warning: Only one shock type available ({shock_types}). Creating single-line plot.")
        
        # Get the available shock type
        available_shock = shock_types[0]
        shock_data = res[res['shock_type'] == available_shock]
        
        plt.figure(figsize=(10, 6))
        valid_data = shock_data.dropna(subset=['beta'])
        
        if len(valid_data) > 0:
            plt.plot(valid_data["h"], valid_data["beta"], marker="o", linewidth=2)
            # Add confidence bands
            plt.fill_between(valid_data["h"], 
                            valid_data["beta"] - 1.96 * valid_data["se"], 
                            valid_data["beta"] + 1.96 * valid_data["se"], 
                            alpha=0.2)
        
        plt.axhline(0.0, ls="--", lw=1, alpha=0.6, color='black')
        plt.title(f"Local Projection IRF: {available_shock} → {ret_col}")
        plt.xlabel("Horizon (months)")
        plt.ylabel("Beta (1-SD scaled)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = out_dir / "lp_irf_plot_compare.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved single shock plot to: {plot_path}")
        plt.close()

def main(ts_path: Path, out_dir: Path, horizons, shock_type="compare", non_overlapping=False):
    """Main function to run local projections"""
    print(f"Loading data from: {ts_path}")
    print(f"Output directory: {out_dir}")
    print(f"Horizons: {horizons}")
    print(f"Shock type: {shock_type}")
    print(f"Non-overlapping sampling: {non_overlapping}")
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    try:
        ts = pd.read_parquet(ts_path)
        print(f"Loaded {len(ts)} observations")
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        return
    
    # Handle index
    if not isinstance(ts.index, pd.DatetimeIndex):
        # Try to coerce first datetime-like column into index
        dcols = [c for c in ts.columns if "date" in c.lower() or "month" in c.lower()]
        if dcols:
            print(f"Converting {dcols[0]} to datetime index")
            ts.index = pd.to_datetime(ts[dcols[0]])
        else:
            print("No datetime column found, using default index")
    
    ts = ts.sort_index()
    print(f"Data spans from {ts.index.min()} to {ts.index.max()}")
    
    # Resolve shock columns
    try:
        shock_cols_dict = resolve_shock_cols(ts, shock_type)
        shocks, ret = autodetect_cols(ts, shock_cols_dict)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Run local projections for each shock type
    all_results = []
    
    for shock in shocks:
        # Determine shock type label for output
        if 'ar1' in shock:
            shock_type_label = "ar1"
        elif 'var_orth' in shock:
            shock_type_label = "var_orth"
        else:
            shock_type_label = shock
        
        print(f"\n=== Running local projections for shock: {shock} ({shock_type_label}) -> returns: {ret} ===")
        
        # Check for missing values
        shock_missing = ts[shock].isna().sum()
        ret_missing = ts[ret].isna().sum()
        print(f"Missing values - {shock}: {shock_missing}, {ret}: {ret_missing}")
        
        # Calculate shock standard deviation for scaling
        shock_data = ts[shock].dropna()
        shock_std = shock_data.std()
        print(f"Shock standard deviation: {shock_std:.4f}")
        
        # Run local projections
        for h in horizons:
            print(f"\nRunning projection for horizon h={h}")
            
            # Compute cumulative returns R_{t→t+h}
            y = compute_cumulative_returns(ts[ret], h)
            X = ts[[shock]]
            
            # Apply non-overlapping sampling if requested
            if non_overlapping:
                # Sample every h months
                sample_indices = range(0, len(ts), h)
                y = y.iloc[sample_indices]
                X = X.iloc[sample_indices]
                print(f"Non-overlapping sampling: using every {h} months, {len(y)} observations")
            
            # Remove missing observations
            valid_obs = ~(y.isna() | X[shock].isna())
            y_clean = y[valid_obs]
            X_clean = X[valid_obs]
            
            if len(y_clean) < 10:
                print(f"Warning: Only {len(y_clean)} valid observations for horizon {h}")
                all_results.append({
                    "h": h, 
                    "beta": np.nan, 
                    "se": np.nan, 
                    "r2": np.nan, 
                    "shock_type": shock_type_label,
                    "overlapping_flag": not non_overlapping,
                    "n_obs": len(y_clean)
                })
                continue
            
            # Use h-1 lags for Newey-West HAC (as requested)
            nw_lags = max(h - 1, 0)
            print(f"Using {nw_lags} lags for Newey-West HAC standard errors")
            
            # Run regression with Newey-West HAC standard errors
            try:
                m = newey_west(y_clean, X_clean, lags=nw_lags)
                beta_raw = float(m.params.get(shock, np.nan))
                se_raw = float(m.bse.get(shock, np.nan))
                r2 = float(m.rsquared)
                
                # Scale coefficient per 1-SD of shock
                beta_scaled = beta_raw * shock_std
                se_scaled = se_raw * shock_std
                
                all_results.append({
                    "h": h,
                    "beta": beta_scaled,
                    "se": se_scaled,
                    "r2": r2,
                    "shock_type": shock_type_label,
                    "overlapping_flag": not non_overlapping,
                    "n_obs": len(y_clean)
                })
                
                print(f"  Beta (1-SD scaled): {beta_scaled:.4f}, SE: {se_scaled:.4f}, R²: {r2:.4f}, N: {len(y_clean)}")
                
            except Exception as e:
                print(f"Error running regression for horizon {h}: {e}")
                all_results.append({
                    "h": h, 
                    "beta": np.nan, 
                    "se": np.nan, 
                    "r2": np.nan, 
                    "shock_type": shock_type_label,
                    "overlapping_flag": not non_overlapping,
                    "n_obs": len(y_clean)
                })
    
    # Create results DataFrame
    res = pd.DataFrame(all_results)
    
    # Save results with specified columns
    csv_path = out_dir / "lp_irf_results.csv"
    res.to_csv(csv_path, index=False)
    print(f"\nSaved CSV results to: {csv_path}")
    
    # Create comparison plot
    create_comparison_plot(res, out_dir, ret)
    
    # Print post-run summary
    print(f"\n[LP] Results written to {csv_path}")
    print("\nPost-run summary:")
    print(res.groupby("shock_type")["h"].agg(["nunique", "min", "max", "count"]))
    print("\nLocal Projection Results:")
    print(res.to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Jordà-style Local Projections")
    ap.add_argument("--ts", type=Path, default=Path("./ts_dataset.parquet"),
                    help="Path to time series dataset (parquet file)")
    ap.add_argument("--out", type=Path, default=Path("./tables_figures"),
                    help="Output directory for results")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6, 12],
                    help="Forecast horizons for local projections")
    ap.add_argument("--shock_type", type=str, choices=["ar1", "var_orth", "compare"], default="compare",
                    help="Shock type to use: 'ar1', 'var_orth', or 'compare' (default: compare)")
    ap.add_argument("--non_overlapping", action="store_true",
                    help="Use non-overlapping sampling (sample every h months)")
    
    args = ap.parse_args()
    main(args.ts, args.out, args.horizons, args.shock_type, args.non_overlapping)
