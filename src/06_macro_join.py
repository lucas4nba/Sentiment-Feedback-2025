import pandas as pd
from src.config import OUT

def main():
    sent = pd.read_parquet(OUT/"sentiment_monthly.parquet")
    iv = pd.read_parquet(OUT/"option_iv_monthly.parquet")
    ctrl = pd.read_parquet(OUT/"controls_monthly.parquet")
    df = sent.merge(iv, on="DATE", how="outer").merge(ctrl, on="DATE", how="outer")
    df = df.sort_values("DATE")
    df.to_parquet(OUT/"macro_monthly.parquet", index=False)
    print("Saved macro_monthly with", df.shape[1]-1, "columns")

if __name__ == "__main__":
    main()
