from pathlib import Path
import pandas as pd
from src.config import DIR_RPV, OUT
from src.utils_time import to_month_end, month_end_sample

USE_COLS = ["PERMNO","DATE","TICKER","NCUSIP","CUSIP","SHRCD","EXCHCD","PRC","RET","SHROUT"]

def coerce_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Convert numeric columns
    for c in ["PERMNO","SHRCD","EXCHCD","SHROUT"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # Convert date column
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    
    # Convert price and return columns to float
    for c in ["PRC", "RET", "BIDLO", "ASKHI", "VOL", "VWRETD", "VWRETX", "EWRETD", "EWRETX"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # Convert string columns explicitly
    for c in ["TICKER", "NCUSIP", "CUSIP"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
            
    return df

def load_one(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx",".xls"}:
        df = pd.read_excel(path, engine="openpyxl")
    elif path.suffix.lower() in {".csv"}:
        # Use chunking for large CSV files to avoid memory issues
        chunk_list = []
        chunk_size = 100000  # Process 100k rows at a time
        for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
            # Standardize columns to upper
            chunk.columns = [c.upper() for c in chunk.columns]
            keep = [c for c in USE_COLS if c in chunk.columns]
            if keep:  # Only process if we have relevant columns
                chunk = chunk[keep]
                chunk = coerce_cols(chunk)
                chunk_list.append(chunk)
        if chunk_list:
            df = pd.concat(chunk_list, ignore_index=True)
        else:
            return pd.DataFrame()
    else:
        return pd.DataFrame()
    
    # For Excel files, still need to process columns
    if path.suffix.lower() in {".xlsx",".xls"}:
        df.columns = [c.upper() for c in df.columns]
        keep = [c for c in USE_COLS if c in df.columns]
        df = df[keep]
        df = coerce_cols(df)
    
    return df

def main():
    files = list(DIR_RPV.glob("**/*"))
    files = [f for f in files if f.is_file()]  # Only process actual files
    
    if not files:
        raise SystemExit("No CRSP files found in ReturnsPricesVol.")
    
    print(f"Processing {len(files)} files...")
    
    # Process files one by one and write to temporary parquet files
    temp_files = []
    for i, file in enumerate(files):
        print(f"Processing file {i+1}/{len(files)}: {file.name}")
        df = load_one(file)
        if df.empty:
            continue
            
        # Apply filters immediately to reduce memory usage (only if columns exist)
        if "SHRCD" in df.columns and "EXCHCD" in df.columns:
            df = df.query("(SHRCD==10 or SHRCD==11) and (EXCHCD==1 or EXCHCD==2 or EXCHCD==3)")
        
        # Only drop NA on columns that exist
        dropna_cols = [c for c in ["PERMNO","DATE"] if c in df.columns]
        if dropna_cols:
            df = df.dropna(subset=dropna_cols)
        
        if not df.empty:
            temp_file = OUT / f"temp_{i}.parquet"
            df.to_parquet(temp_file, index=False)
            temp_files.append(temp_file)
    
    if not temp_files:
        raise SystemExit("No valid CRSP data found after filtering.")
    
    print(f"Combining {len(temp_files)} temporary files...")
    
    # Read and combine parquet files sequentially to avoid memory issues
    print("Combining files sequentially...")
    df_list = []
    for i, temp_file in enumerate(temp_files):
        print(f"Loading temp file {i+1}/{len(temp_files)}")
        chunk = pd.read_parquet(temp_file)
        df_list.append(chunk)
    
    print("Concatenating all data...")
    df = pd.concat(df_list, ignore_index=True)
    
    # Sort only if we have DATE column and data is not too large
    print("Processing combined data...")
    if "DATE" in df.columns and len(df) < 50_000_000:  # Only sort if less than 50M rows
        print("Sorting by PERMNO and DATE...")
        df = df.sort_values(["PERMNO","DATE"])
    
    print(f"Combined dataset has {len(df)} rows")
    
    # Apply final filters if columns exist
    if "SHRCD" in df.columns and "EXCHCD" in df.columns:
        print("Applying share code and exchange filters...")
        df = df.query("(SHRCD==10 or SHRCD==11) and (EXCHCD==1 or EXCHCD==2 or EXCHCD==3)")
    
    # Save final result
    df.to_parquet(OUT / "stock_id_master.parquet", index=False)
    
    # Clean up temporary files
    for temp_file in temp_files:
        if temp_file.exists() and "temp_" in temp_file.name:
            temp_file.unlink()
    
    print(f"Saved stock_id_master.parquet with {len(df)} rows")

    # Month-end sample for identifiers - process in chunks by PERMNO
    print("Creating monthly sample...")
    keep_cols = [c for c in df.columns if c not in {"DATE"}]
    
    # Process monthly sample in chunks to avoid memory issues
    unique_permnos = df['PERMNO'].unique()
    monthly_chunks = []
    
    chunk_size = 1000  # Process 1000 PERMNOs at a time
    for i in range(0, len(unique_permnos), chunk_size):
        permno_chunk = unique_permnos[i:i+chunk_size]
        df_chunk = df[df['PERMNO'].isin(permno_chunk)]
        
        me_chunk = (df_chunk
                   .groupby("PERMNO", group_keys=True)
                   .apply(lambda g: month_end_sample(g, "DATE", [c for c in keep_cols if c!="PERMNO"]), include_groups=False))
        # Reset index to get PERMNO back as a column
        me_chunk = me_chunk.reset_index(level=0)
        # Ensure we have the right column structure
        if 'PERMNO' not in me_chunk.columns:
            me_chunk = me_chunk.reset_index()
        # Reorder columns to put PERMNO first if it exists
        if 'PERMNO' in me_chunk.columns:
            cols = ["PERMNO"] + [c for c in me_chunk.columns if c != "PERMNO"]
            me_chunk = me_chunk[cols]
        monthly_chunks.append(me_chunk)
        
        print(f"Processed monthly sample for PERMNOs {i+1}-{min(i+chunk_size, len(unique_permnos))}")
    
    me = pd.concat(monthly_chunks, ignore_index=True)
    me.to_parquet(OUT / "stock_id_master_monthly.parquet", index=False)
    print(f"Saved stock_id_master_monthly.parquet with {len(me)} rows")
    print("Completed stock ID master processing.")

if __name__ == "__main__":
    main()
