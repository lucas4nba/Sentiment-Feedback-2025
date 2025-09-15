from pathlib import Path
import pandas as pd
from src.config import DIR_CONTROLS, OUT
from src.utils_io import read_any_excel, best_date_col
from src.utils_time import monthly_avg

def one(p: Path):
    if p.suffix.lower() in {".xlsx", ".xls"}:
        df = read_any_excel(p)
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")
    date_col = best_date_col(df)
    num = df.drop(columns=[date_col]).select_dtypes(include="number")
    if num.empty:
        # force numeric on first non-date col
        c = [c for c in df.columns if c != date_col][0]
        df[c] = pd.to_numeric(df[c], errors="coerce")
        num = df[[c]]
    nm = p.stem.upper()
    out = monthly_avg(pd.concat([df[[date_col]], num.iloc[:,[0]]], axis=1), date_col, [num.columns[0]])
    out.rename(columns={num.columns[0]: nm}, inplace=True)
    return out

def main():
    parts = []
    for p in sorted(DIR_CONTROLS.glob("*")):
        if p.suffix.lower() in {".xlsx", ".xls", ".csv"}:
            try:
                parts.append(one(p))
            except Exception as e:
                print("skip", p.name, e)
    if not parts:
        raise SystemExit("No controls files.")
    df = parts[0]
    for x in parts[1:]:
        df = df.merge(x, on="DATE", how="outer")
    df = df.sort_values("DATE")
    df.to_parquet(OUT / "controls_monthly.parquet", index=False)
    print("Saved controls_monthly columns:", list(df.columns))

if __name__ == "__main__":
    main()
