"""
Construct Sentiment Shocks Script
Uses column_map.json to standardize variable selection and produces sentiment shocks via AR(1) residuals
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from scipy.linalg import cholesky
import warnings
warnings.filterwarnings('ignore')

def load_column_map(data_dir):
    """Load column mapping from JSON file"""
    map_path = Path(data_dir) / "build" / "column_map.json"
    if not map_path.exists():
        raise FileNotFoundError(f"Column map not found at {map_path}")
    
    with open(map_path, 'r') as f:
        return json.load(f)

def apply_gap_policy(df, column_map, dataset_type):
    """Apply gap policy based on column map configuration"""
    gap_policy = column_map.get("gap_policy", {})
    
    if dataset_type in ["sentiment", "flows"]:
        # No imputation for sentiment or flows
        policy = gap_policy.get(dataset_type, "no_imputation")
        if policy == "no_imputation":
            return df
    
    elif dataset_type == "controls":
        # Allow last-value-carry for macro controls
        policy = gap_policy.get("controls", "last_value_carry")
        if policy == "last_value_carry":
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            df[numeric_cols] = df[numeric_cols].ffill()
            
        # Option to drop months with missing controls
        drop_missing = gap_policy.get("drop_missing_controls", False)
        if drop_missing:
            df = df.dropna()
    
    return df

def get_analysis_window(column_map):
    """Get analysis window from column map"""
    window = column_map.get("analysis_window", {})
    start_date = pd.to_datetime(window.get("start_date", "1990-01-31"))
    end_date = pd.to_datetime(window.get("end_date", "2024-12-31"))
    return start_date, end_date

def filter_to_analysis_window(df, start_date, end_date, date_col='DATE'):
    """Filter dataframe to analysis window"""
    if date_col not in df.columns:
        return df
    
    mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
    return df[mask].copy()

def select_columns(df, column_list, dataset_name=""):
    """Select specified columns from dataframe"""
    available_cols = [col for col in column_list if col in df.columns]
    missing_cols = [col for col in column_list if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: {dataset_name} missing columns: {missing_cols}")
    
    if not available_cols:
        print(f"Warning: No columns found for {dataset_name}")
        return pd.DataFrame()
    
    # Always include DATE if it exists
    cols_to_select = ['DATE'] if 'DATE' in df.columns else []
    cols_to_select.extend(available_cols)
    
    return df[cols_to_select].copy()

def construct_ar1_shock(series, series_name="variable"):
    """
    Construct AR(1) shock from a time series
    Returns the residuals from an AR(1) model
    """
    # Remove missing values for AR estimation
    clean_series = series.dropna()
    
    if len(clean_series) < 10:  # Need at least 10 observations
        print(f"Warning: {series_name} has insufficient data for AR(1) estimation")
        return pd.Series(index=series.index, dtype=float)
    
    try:
        # Fit AR(1) model
        ar_model = AutoReg(clean_series, lags=1, trend='c')
        ar_fitted = ar_model.fit()
        
        # Get residuals
        residuals = ar_fitted.resid
        
        # Create full series with NaN for missing original values
        shock_series = pd.Series(index=series.index, dtype=float)
        shock_series.loc[clean_series.index] = residuals
        
        # Print diagnostics
        print(f"  {series_name} AR(1) results:")
        print(f"    Coefficient: {ar_fitted.params[1]:.4f}")
        try:
            print(f"    R-squared: {ar_fitted.rsquared:.4f}")
        except:
            # Some versions don't have rsquared attribute
            pass
        print(f"    Observations used: {len(clean_series)}")
        
        # Test for remaining serial correlation
        try:
            ljung_box = acorr_ljungbox(residuals, lags=min(5, len(residuals)//4), return_df=True)
            pval = ljung_box['lb_pvalue'].iloc[-1]  # Last lag test
            if pval < 0.05:
                print(f"    Warning: Residual serial correlation detected (p={pval:.4f})")
            else:
                print(f"    No residual serial correlation (p={pval:.4f})")
        except:
            print("    Could not test residual serial correlation")
        
        return shock_series
        
    except Exception as e:
        print(f"Error fitting AR(1) model for {series_name}: {e}")
        return pd.Series(index=series.index, dtype=float)

def build_var_orth_shock(df):
    """
    Build VAR-based orthogonalized sentiment shock
    
    Fits a VAR model on [UMCSENT, market_excess_return] with lag selection by AIC.
    Extracts reduced-form residuals and orthogonalizes UMCSENT residual to return residual.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing UMCSENT and market_excess_ret columns with DATE
        
    Returns:
    --------
    dict : Dictionary containing:
        - shock_var_orth: Standardized orthogonalized sentiment shock
        - shock_ar1: Standardized AR(1) sentiment shock (for comparison)
        - diagnostics: Model diagnostics and correlations
    """
    print("\nConstructing VAR-based orthogonalized sentiment shock...")
    
    # Check required columns
    required_cols = ['UMCSENT', 'market_excess_ret']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for VAR: {missing_cols}")
    
    # Prepare data - drop missing values for VAR estimation
    var_data = df[['DATE'] + required_cols].dropna()
    
    if len(var_data) < 20:
        raise ValueError(f"Insufficient data for VAR estimation: {len(var_data)} observations")
    
    print(f"  VAR estimation sample: {len(var_data)} observations")
    print(f"  Date range: {var_data['DATE'].min().strftime('%Y-%m')} to {var_data['DATE'].max().strftime('%Y-%m')}")
    
    # Test for stationarity
    diagnostics = {'stationarity_tests': {}}
    for col in required_cols:
        adf_result = adfuller(var_data[col].dropna(), autolag='AIC')
        diagnostics['stationarity_tests'][col] = {
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05
        }
        print(f"  {col} ADF test: statistic={adf_result[0]:.4f}, p-value={adf_result[1]:.4f}")
        if adf_result[1] < 0.05:
            print(f"    -> {col} is stationary")
        else:
            print(f"    -> Warning: {col} may be non-stationary (p={adf_result[1]:.4f})")
    
    # Prepare VAR data (just the endogenous variables)
    var_endog = var_data[required_cols].values
    
    # Select lag order using AIC (default max_lags=1, but allow up to 6)
    max_lags = min(6, len(var_data) // 10)  # Conservative lag selection
    var_model = VAR(var_endog)
    lag_order_results = var_model.select_order(maxlags=max_lags)
    
    # Use AIC-selected lag order (default to 1 if selection fails)
    try:
        optimal_lags = lag_order_results.aic
        if optimal_lags == 0:
            optimal_lags = 1  # Force at least 1 lag
    except:
        optimal_lags = 1
        
    print(f"  Optimal lag order (AIC): {optimal_lags}")
    
    # Fit VAR model
    var_fitted = var_model.fit(optimal_lags)
    
    # Extract reduced-form residuals
    residuals = var_fitted.resid  # Shape: (n_obs, n_vars)
    umcsent_resid = residuals[:, 0]  # UMCSENT residuals
    return_resid = residuals[:, 1]   # market_excess_ret residuals
    
    print(f"  VAR model summary:")
    print(f"    Lag order: {var_fitted.k_ar}")
    print(f"    Observations: {var_fitted.nobs}")
    print(f"    Variables: {var_fitted.names}")
    
    # Calculate correlation between reduced-form residuals
    resid_corr = np.corrcoef(umcsent_resid, return_resid)[0, 1]
    print(f"    Reduced-form residual correlation: {resid_corr:.4f}")
    
    # Orthogonalize UMCSENT residual to market return residual
    # Use simple regression-based orthogonalization: e_sentiment_orth = e_sentiment - β * e_return
    # where β is from regressing e_sentiment on e_return
    beta = np.cov(umcsent_resid, return_resid)[0, 1] / np.var(return_resid)
    umcsent_resid_orth = umcsent_resid - beta * return_resid
    
    print(f"    Orthogonalization coefficient (β): {beta:.4f}")
    
    # Verify orthogonalization
    orth_corr = np.corrcoef(umcsent_resid_orth, return_resid)[0, 1]
    print(f"    Post-orthogonalization correlation: {orth_corr:.6f} (should be ~0)")
    
    # Standardize shocks
    shock_var_orth_std = (umcsent_resid_orth - np.mean(umcsent_resid_orth)) / np.std(umcsent_resid_orth)
    
    # Also create standardized AR(1) shock for comparison
    ar1_shock = construct_ar1_shock(var_data['UMCSENT'], 'UMCSENT_AR1')
    ar1_shock_clean = ar1_shock.dropna()
    
    # Create full-length series (with NaNs for missing data)
    shock_var_orth_full = pd.Series(index=df.index, dtype=float)
    shock_ar1_full = pd.Series(index=df.index, dtype=float)
    
    # Map VAR orthogonalized shock back to original dataframe indices
    var_indices = var_data.index
    
    # Ensure we have the right length for VAR shock assignment
    if len(var_indices) == len(shock_var_orth_std):
        shock_var_orth_full.loc[var_indices] = shock_var_orth_std
    else:
        # Handle length mismatch due to VAR lags
        min_len = min(len(var_indices), len(shock_var_orth_std))
        shock_var_orth_full.loc[var_indices[:min_len]] = shock_var_orth_std[:min_len]
    
    # Map AR(1) shock back to original dataframe indices
    if len(ar1_shock_clean) > 0:
        shock_ar1_std = (ar1_shock_clean - np.mean(ar1_shock_clean)) / np.std(ar1_shock_clean)
        # Ensure consistent indexing
        if len(ar1_shock_clean.index) == len(shock_ar1_std):
            shock_ar1_full.loc[ar1_shock_clean.index] = shock_ar1_std
        else:
            # Handle any potential length mismatch
            min_len = min(len(ar1_shock_clean.index), len(shock_ar1_std))
            shock_ar1_full.loc[ar1_shock_clean.index[:min_len]] = shock_ar1_std[:min_len]
    
    # Store diagnostics
    diagnostics.update({
        'var_model': {
            'lag_order': optimal_lags,
            'nobs': var_fitted.nobs,
            'variables': var_fitted.names,
            'aic': var_fitted.aic,
            'bic': var_fitted.bic,
            'hqic': var_fitted.hqic
        },
        'residual_correlations': {
            'reduced_form': resid_corr,
            'post_orthogonalization': orth_corr,
            'orthogonalization_coeff': beta
        },
        'shock_correlations': {
            'var_orth_vs_ar1': None  # Will calculate after successful completion
        }
    })
    
    # Calculate correlation between the two shock series (aligned)
    aligned_shocks = pd.DataFrame({
        'var_orth': shock_var_orth_full,
        'ar1': shock_ar1_full
    }).dropna()
    
    if len(aligned_shocks) > 1:
        shock_corr = aligned_shocks['var_orth'].corr(aligned_shocks['ar1'])
        diagnostics['shock_correlations']['var_orth_vs_ar1'] = shock_corr
    else:
        shock_corr = np.nan
        diagnostics['shock_correlations']['var_orth_vs_ar1'] = np.nan
    
    print(f"  Shock comparison:")
    print(f"    VAR orth shock: mean={np.mean(shock_var_orth_std):.4f}, std={np.std(shock_var_orth_std):.4f}")
    print(f"    AR(1) shock: mean={np.mean(shock_ar1_std):.4f}, std={np.std(shock_ar1_std):.4f}")
    print(f"    Correlation (VAR orth vs AR1): {shock_corr:.4f}")
    
    return {
        'shock_var_orth': shock_var_orth_full,
        'shock_ar1': shock_ar1_full, 
        'diagnostics': diagnostics
    }

def save_shock_catalog(diagnostics, output_dir):
    """
    Save shock construction catalog with diagnostics and correlations
    
    Parameters:
    -----------
    diagnostics : dict
        Dictionary containing VAR model diagnostics
    output_dir : Path
        Directory to save catalog file
    """
    catalog_path = output_dir / "shock_construction_catalog.json"
    
    # Create serializable diagnostics (convert numpy types)
    serializable_diagnostics = {}
    
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    serializable_diagnostics = convert_numpy(diagnostics)
    
    # Add metadata
    catalog = {
        'metadata': {
            'creation_date': pd.Timestamp.now().isoformat(),
            'description': 'Sentiment shock construction diagnostics and correlations',
            'methods': ['AR(1) residuals', 'VAR orthogonalized residuals']
        },
        'diagnostics': serializable_diagnostics,
        'interpretation': {
            'shock_var_orth': 'UMCSENT residual orthogonalized to market return residual from VAR model',
            'shock_ar1': 'UMCSENT residual from AR(1) model',
            'stationarity_note': 'ADF test p-values < 0.05 indicate stationarity',
            'orthogonalization_note': 'Post-orthogonalization correlation should be near zero'
        }
    }
    
    with open(catalog_path, 'w') as f:
        json.dump(catalog, f, indent=2)
    
    print(f"Shock catalog saved: {catalog_path}")
    return catalog_path

def drop_missing_columns(df, threshold=1.0):
    """Drop columns that are missing above threshold (default 100%)"""
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct >= threshold].index.tolist()
    
    if cols_to_drop:
        print(f"Dropping 100% missing columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    return df


def construct_individual_shock(shock_name, df, ar1=True, standardize=True):
    """
    Construct individual shock based on shock name and parameters.
    
    Parameters:
    -----------
    shock_name : str
        Name of the shock ('umcsent', 'news', 'twitter', 'reddit', 'bw')
    df : pd.DataFrame
        DataFrame containing the raw data
    ar1 : bool
        Whether to use AR(1) residualization
    standardize : bool
        Whether to standardize the shock
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns [date, shock]
    """
    # Map shock names to column names
    shock_mapping = {
        'umcsent': 'UMCSENT',
        'news': 'NEWS_SENTIMENT',  # Adjust based on actual column names
        'twitter': 'TWITTER_SENTIMENT',
        'reddit': 'REDDIT_SENTIMENT',
        'bw': 'BW_SENTIMENT'
    }
    
    raw_col = shock_mapping.get(shock_name.lower())
    if raw_col not in df.columns:
        # Try to find similar column names
        possible_cols = [col for col in df.columns if shock_name.lower() in col.lower()]
        if possible_cols:
            raw_col = possible_cols[0]
            print(f"Using column '{raw_col}' for shock '{shock_name}'")
        else:
            raise ValueError(f"Could not find column for shock '{shock_name}'. Available columns: {list(df.columns)}")
    
    print(f"Constructing {shock_name} shock from column '{raw_col}'")
    
    # Extract the series
    series = df[raw_col].copy()
    
    if ar1:
        print(f"  Using AR(1) residualization")
        # Use existing AR(1) function
        shock_series = construct_ar1_shock(series, series_name=raw_col)
        
        if standardize:
            print(f"  Standardizing AR(1) residuals")
            shock_clean = shock_series.dropna()
            if len(shock_clean) > 0:
                shock_std = (shock_clean - shock_clean.mean()) / shock_clean.std()
                shock_series.loc[shock_clean.index] = shock_std
    else:
        print(f"  Using raw series (no AR(1))")
        shock_series = series.copy()
        
        if standardize:
            print(f"  Standardizing raw series")
            shock_clean = shock_series.dropna()
            if len(shock_clean) > 0:
                shock_std = (shock_clean - shock_clean.mean()) / shock_clean.std()
                shock_series.loc[shock_clean.index] = shock_std
    
    # Create output DataFrame
    result_df = pd.DataFrame({
        'date': df['DATE'],
        'shock': shock_series
    })
    
    # Remove rows with missing shocks
    result_df = result_df.dropna()
    
    print(f"  Final shock: {len(result_df)} observations")
    if len(result_df) > 0:
        print(f"  Mean: {result_df['shock'].mean():.4f}, Std: {result_df['shock'].std():.4f}")
    
    return result_df

def main():
    """Main function to construct sentiment shocks"""
    parser = argparse.ArgumentParser(description='Construct sentiment shocks via AR(1) and VAR orthogonalized residuals')
    parser.add_argument('--data-dir', default='.', help='Data directory path')
    parser.add_argument('--out', default='./ts_dataset.parquet', help='Output file path')
    parser.add_argument('--shock', choices=['umcsent', 'news', 'twitter', 'reddit', 'bw'], 
                       default='umcsent', help='Shock type to construct')
    parser.add_argument('--standardize', choices=['yes', 'no'], default='yes', 
                       help='Whether to standardize the shock (default: yes)')
    parser.add_argument('--ar1', choices=['yes', 'no'], default='yes', 
                       help='Whether to use AR(1) residualization (default: yes)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    
    print("=" * 60)
    print(f"CONSTRUCTING {args.shock.upper()} SHOCK")
    print("=" * 60)
    
    # Load column mapping
    print("Loading column mapping...")
    column_map = load_column_map(data_dir)
    
    # Get analysis window
    start_date, end_date = get_analysis_window(column_map)
    print(f"Analysis window: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
    
    # Load datasets
    build_dir = data_dir / "build"
    
    print("\nLoading datasets...")
    
    # Load sentiment data
    sentiment_df = pd.read_parquet(build_dir / "sentiment_monthly.parquet")
    sentiment_cols = column_map.get("sentiment", [])
    sentiment_selected = select_columns(sentiment_df, sentiment_cols, "sentiment")
    sentiment_windowed = filter_to_analysis_window(sentiment_selected, start_date, end_date)
    sentiment_final = apply_gap_policy(sentiment_windowed, column_map, "sentiment")
    print(f"Sentiment: {len(sentiment_final)} obs, columns: {list(sentiment_final.columns)}")
    
    # Load controls data
    controls_df = pd.read_parquet(build_dir / "controls_monthly.parquet")
    controls_cols = column_map.get("controls_keep", [])
    controls_selected = select_columns(controls_df, controls_cols, "controls")
    controls_windowed = filter_to_analysis_window(controls_selected, start_date, end_date)
    controls_final = apply_gap_policy(controls_windowed, column_map, "controls")
    print(f"Controls: {len(controls_final)} obs, columns: {list(controls_final.columns)}")
    
    # Load flows data if available (but we know it's not useful)
    flows_path = build_dir / "flows_market_monthly.parquet"
    flows_final = pd.DataFrame()
    if flows_path.exists():
        flows_df = pd.read_parquet(flows_path)
        flows_cols = column_map.get("flows", [])
        flows_selected = select_columns(flows_df, flows_cols, "flows")
        flows_windowed = filter_to_analysis_window(flows_selected, start_date, end_date)
        flows_final = apply_gap_policy(flows_windowed, column_map, "flows")
        
        if len(flows_final) < 12:
            print(f"Flows: {len(flows_final)} obs - insufficient for analysis")
            flows_final = pd.DataFrame()
        else:
            print(f"Flows: {len(flows_final)} obs, columns: {list(flows_final.columns)}")
    
    # Merge datasets first to get combined data for VAR
    print("\nMerging datasets for shock construction...")
    
    # Start with sentiment data
    combined_data = sentiment_final.copy()
    
    # Merge controls to get market_excess_ret
    if not controls_final.empty:
        combined_data = combined_data.merge(controls_final, on='DATE', how='left')
        print(f"After controls merge: {combined_data.shape}")
    
    # Construct both types of sentiment shocks
    print("\nConstructing sentiment shocks...")
    
    # Initialize shocks dataframe
    sentiment_shocks = sentiment_final[['DATE']].copy()
    
    # 1. Traditional AR(1) shocks for all sentiment variables
    print("\n1. AR(1) residual shocks:")
    for col in sentiment_final.columns:
        if col == 'DATE':
            continue
            
        shock_col = f"{col}_shock_ar1"
        sentiment_shocks[shock_col] = construct_ar1_shock(
            sentiment_final[col], 
            series_name=col
        )
    
    # 2. VAR-based orthogonalized shock for UMCSENT (if data available)
    var_diagnostics = None
    if 'UMCSENT' in combined_data.columns and 'market_excess_ret' in combined_data.columns:
        try:
            print("\n2. VAR-based orthogonalized shock:")
            var_results = build_var_orth_shock(combined_data)
            
            # Add VAR-based shocks to dataset
            sentiment_shocks['UMCSENT_shock_var_orth'] = var_results['shock_var_orth']
            
            # Store diagnostics for catalog
            var_diagnostics = var_results['diagnostics']
            
        except Exception as e:
            print(f"Warning: VAR shock construction failed: {e}")
            print("Continuing with AR(1) shocks only...")
    else:
        print("\n2. VAR-based shock: Skipped (missing UMCSENT or market_excess_ret)")
        missing_vars = []
        if 'UMCSENT' not in combined_data.columns:
            missing_vars.append('UMCSENT')
        if 'market_excess_ret' not in combined_data.columns:
            missing_vars.append('market_excess_ret')
        print(f"   Missing variables: {missing_vars}")
    
    # Merge all datasets
    print("\nFinalizing dataset...")
    
    # Start with sentiment shocks
    ts_dataset = sentiment_shocks.copy()
    
    # Add original sentiment levels
    for col in sentiment_final.columns:
        if col != 'DATE':
            ts_dataset[col] = sentiment_final[col]
    
    # Merge controls
    if not controls_final.empty:
        ts_dataset = ts_dataset.merge(controls_final, on='DATE', how='left')
        print(f"After controls merge: {ts_dataset.shape}")
    
    # Merge flows if available
    if not flows_final.empty:
        ts_dataset = ts_dataset.merge(flows_final, on='DATE', how='left')
        print(f"After flows merge: {ts_dataset.shape}")
    
    # Drop 100% missing columns
    ts_dataset = drop_missing_columns(ts_dataset, threshold=1.0)
    
    # Sort by date
    ts_dataset = ts_dataset.sort_values('DATE').reset_index(drop=True)
    
    print(f"\nFinal dataset: {ts_dataset.shape}")
    print(f"Columns: {list(ts_dataset.columns)}")
    
    # Save outputs
    print(f"\nSaving outputs...")
    
    # Save parquet
    ts_dataset.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")
    
    # Save CSV for eyeballing
    csv_path = out_path.with_suffix('.csv')
    ts_dataset.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Save individual shock file
    ar1_flag = args.ar1 == 'yes'
    standardize_flag = args.standardize == 'yes'
    
    print(f"\nConstructing individual {args.shock} shock...")
    individual_shock = construct_individual_shock(args.shock, sentiment_final, ar1_flag, standardize_flag)
    
    # Save individual shock
    shock_filename = f"shocks_{args.shock}.parquet"
    shock_path = data_dir / shock_filename
    individual_shock.to_parquet(shock_path, index=False)
    print(f"Saved individual shock: {shock_path}")
    
    # Save shock construction catalog if VAR diagnostics available
    if var_diagnostics is not None:
        catalog_path = save_shock_catalog(var_diagnostics, data_dir)
        print(f"Saved: {catalog_path}")
    else:
        print("No VAR diagnostics to save (AR(1) shocks only)")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"Time series dataset: {ts_dataset.shape[0]} observations × {ts_dataset.shape[1]} variables")
    print(f"Date range: {ts_dataset['DATE'].min().strftime('%Y-%m')} to {ts_dataset['DATE'].max().strftime('%Y-%m')}")
    
    # Show shock statistics
    shock_cols = [col for col in ts_dataset.columns if '_shock' in col]
    if shock_cols:
        print(f"\nSentiment shocks constructed: {len(shock_cols)}")
        
        # Group by shock type
        ar1_shocks = [col for col in shock_cols if '_shock_ar1' in col]
        var_shocks = [col for col in shock_cols if '_shock_var_orth' in col]
        
        if ar1_shocks:
            print(f"\n  AR(1) residual shocks ({len(ar1_shocks)}):")
            for col in ar1_shocks:
                shock_data = ts_dataset[col].dropna()
                if len(shock_data) > 0:
                    print(f"    {col}: mean={shock_data.mean():.4f}, std={shock_data.std():.4f}, obs={len(shock_data)}")
        
        if var_shocks:
            print(f"\n  VAR orthogonalized shocks ({len(var_shocks)}):")
            for col in var_shocks:
                shock_data = ts_dataset[col].dropna()
                if len(shock_data) > 0:
                    print(f"    {col}: mean={shock_data.mean():.4f}, std={shock_data.std():.4f}, obs={len(shock_data)}")
        
        # Show correlation between shock types if both available
        if ar1_shocks and var_shocks and 'UMCSENT_shock_ar1' in ts_dataset.columns and 'UMCSENT_shock_var_orth' in ts_dataset.columns:
            ar1_data = ts_dataset['UMCSENT_shock_ar1'].dropna()
            var_data = ts_dataset['UMCSENT_shock_var_orth'].dropna()
            if len(ar1_data) > 1 and len(var_data) > 1:
                # Align the series for correlation
                aligned_data = ts_dataset[['UMCSENT_shock_ar1', 'UMCSENT_shock_var_orth']].dropna()
                if len(aligned_data) > 1:
                    corr = aligned_data['UMCSENT_shock_ar1'].corr(aligned_data['UMCSENT_shock_var_orth'])
                    print(f"\n  Correlation (UMCSENT AR1 vs VAR orth): {corr:.4f}")
    
    # Show missingness summary
    missing_summary = ts_dataset.isnull().mean().sort_values(ascending=False)
    high_missing = missing_summary[missing_summary > 0.1]  # >10% missing
    if len(high_missing) > 0:
        print(f"\nVariables with >10% missing data:")
        for var, pct in high_missing.items():
            print(f"  {var}: {pct:.1%} missing")
    
    print(f"\nOutputs saved:")
    print(f"  {out_path}")
    print(f"  {csv_path}")
    print(f"  {shock_path}")
    if var_diagnostics is not None:
        print(f"  {data_dir}/shock_construction_catalog.json")
    print("\nSentiment shock construction complete!")
    print("Available shocks:")
    print("  - AR(1) residual shocks: Traditional univariate approach")
    if var_diagnostics is not None:
        print("  - VAR orthogonalized shock: UMCSENT residual orthogonal to market returns")
    print(f"  - Individual {args.shock} shock: {shock_filename}")

if __name__ == "__main__":
    main()