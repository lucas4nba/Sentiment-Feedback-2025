import pandas as pd
from src.config import OUT

def main():
    p = pd.read_parquet(OUT/"panel_monthly.parquet")
    print("Rows:", len(p), "Unique PERMNO:", p.PERMNO.nunique(), "Date span:", p.DATE.min(), "→", p.DATE.max())
    # Macro missingness
    macro_cols = [c for c in p.columns if c not in {"PERMNO","DATE","TICKER","NCUSIP","PRC","RET","SHROUT","SHRCD","EXCHCD"}]
    miss = p[macro_cols].isna().mean().sort_values()
    print("\nMacro columns non-missing share (top 10):")
    print((1-miss).sort_values(ascending=False).head(10))
    # Basic bounds
    if "RET" in p.columns:
        print("\nRET summary:", p["RET"].describe())
    if "SHROUT" in p.columns:
        bad = (p["SHROUT"]<=0).sum()
        print("Non-positive SHROUT rows:", bad)

if __name__ == "__main__":
    main()
