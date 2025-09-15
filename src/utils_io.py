import pandas as pd
import re
from pathlib import Path

def read_any_excel(path: Path) -> pd.DataFrame:
    """Read the first sheet of an Excel file with flexible column types."""
    return pd.read_excel(path, sheet_name=0, engine="openpyxl")

def best_date_col(df: pd.DataFrame):
    """Return the most plausible date column name."""
    candidates = [c for c in df.columns if re.search(r"(date|dt|time|calend|month)", str(c), re.I)]
    if candidates:
        return candidates[0]
    # fallback: first column
    return df.columns[0]
