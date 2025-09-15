import pandas as pd
from src.config import OUT
from src.utils_time import month_end_sample

def main():
    # Use the pre-computed monthly data instead of processing daily data
    print("Loading monthly stock data...")
    me = pd.read_parquet(OUT/"stock_id_master_monthly.parquet")
    print(f"Loaded monthly stock data: {me.shape}")
    
    print("Loading macro data...")
    macro = pd.read_parquet(OUT/"macro_monthly.parquet")
    print(f"Loaded macro data: {macro.shape}")
    
    print("Loading flows data...")
    flows = pd.read_parquet(OUT/"flows_market_monthly.parquet") if (OUT/"flows_market_monthly.parquet").exists() else pd.DataFrame(columns=["DATE"])
    print(f"Loaded flows data: {flows.shape}")
    
    print("Loading microstructure features...")
    microstructure = pd.read_parquet(OUT/"microstructure_features_monthly.parquet") if (OUT/"microstructure_features_monthly.parquet").exists() else pd.DataFrame(columns=["PERMNO", "DATE"])
    print(f"Loaded microstructure data: {microstructure.shape}")
    
    print("Merging datasets...")
    panel = me.merge(macro, on="DATE", how="left")
    print(f"After macro merge: {panel.shape}, columns: {panel.columns.tolist()[:10]}...")
    
    if not flows.empty:
        panel = panel.merge(flows, on="DATE", how="left")
        print(f"After flows merge: {panel.shape}")
    
    if not microstructure.empty:
        panel = panel.merge(microstructure, on=["PERMNO", "DATE"], how="left")
        print(f"After microstructure merge: {panel.shape}")
        print("Microstructure columns added:", [col for col in microstructure.columns if col not in ["PERMNO", "DATE"]])
    
    print("Sorting final panel...")
    # Check if PERMNO column exists
    if "PERMNO" in panel.columns:
        panel = panel.sort_values(["PERMNO","DATE"])
    else:
        print("Warning: PERMNO column not found, sorting by DATE only")
        print(f"Available columns: {panel.columns.tolist()}")
        panel = panel.sort_values(["DATE"])
    
    print("Saving panel...")
    panel.to_parquet(OUT/"panel_monthly.parquet", index=False)
    print("Saved panel_monthly:", panel.shape)
    
    # Display summary of microstructure features if available
    if not microstructure.empty:
        print("\nMicrostructure features summary:")
        for col in ['AMIHUD_AVAILABLE', 'CS_AVAILABLE', 'TAQ_AVAILABLE', 'SI_AVAILABLE']:
            if col in panel.columns:
                available_pct = panel[col].mean() * 100
                print(f"  {col}: {available_pct:.1f}% of panel observations")
        
        if 'AMIHUD' in panel.columns:
            amihud_stats = panel['AMIHUD'].describe()
            print(f"\nAmihud illiquidity in panel (non-null obs: {panel['AMIHUD'].count()}):")
            print(f"  Mean: {amihud_stats['mean']:.2e}")
            print(f"  Std: {amihud_stats['std']:.2e}")
            print(f"  Median: {amihud_stats['50%']:.2e}")

if __name__ == "__main__":
    main()
