import pandas as pd

def to_cusip8(x: pd.Series) -> pd.Series:
    return (x.astype(str)
             .str.replace(r"[^0-9A-Z]", "", regex=True)
             .str.upper()
             .str.slice(0,8))
