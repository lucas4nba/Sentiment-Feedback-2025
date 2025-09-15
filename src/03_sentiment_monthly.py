from pathlib import Path
import pandas as pd
from src.config import DIR_SENTIMENT, OUT
from src.utils_io import read_any_excel, best_date_col
from src.utils_time import monthly_avg

def load_generic(p: Path) -> pd.DataFrame:
    if p.suffix.lower() in {".xlsx", ".xls"}:
        df = read_any_excel(p)
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")
    date_col = best_date_col(df)
    # if Baker-Wurgler file, try SENT column
    key = p.stem
    if "Investor_Sentiment_Data" in key:
        cols = [c for c in df.columns if c.upper() in {"SENT","SENTZ"}]
        if cols:
            k = "BW_SENT" if "SENT" in cols[0].upper() else "BW_SENTZ"
            out = df[[date_col, cols[0]]].rename(columns={date_col:"DATE", cols[0]:k})
            out["DATE"] = pd.to_datetime(out["DATE"], errors="coerce")
            # Assume monthly already
            out = out.dropna(subset=["DATE"]).drop_duplicates("DATE")
            return out
    # otherwise pick first numeric col
    val = df.drop(columns=[date_col]).select_dtypes(include="number")
    if val.empty:
        # force numeric on first non-date col
        c = [c for c in df.columns if c != date_col][0]
        df[c] = pd.to_numeric(df[c], errors="coerce")
        val = df[[c]]
    series_name = p.stem.upper()
    out = monthly_avg(pd.concat([df[[date_col]], val.iloc[:,[0]]], axis=1), date_col, [val.columns[0]])
    out.rename(columns={val.columns[0]: series_name}, inplace=True)
    return out

def main():
    parts = []
    for p in sorted(DIR_SENTIMENT.glob("*")):
        if p.suffix.lower() in {".xlsx", ".xls", ".csv"}:
            try:
                parts.append(load_generic(p))
            except Exception as e:
                print("skip", p.name, e)
    if not parts:
        raise SystemExit("No sentiment files.")
    df = parts[0]
    for x in parts[1:]:
        df = df.merge(x, on="DATE", how="outer")
    df = df.sort_values("DATE")
    df.to_parquet(OUT / "sentiment_monthly.parquet", index=False)
    print("Saved sentiment_monthly columns:", list(df.columns))

if __name__ == "__main__":
    main()
