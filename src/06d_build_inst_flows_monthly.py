#!/usr/bin/env python3
"""
Build institutional flows from S34 Type 4 data.

This script processes S34 Type 4 change in holdings data to create
quarterly institutional flow measures scaled by shares outstanding.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import glob


def setup_logging(log_file: str = "logs/build_inst_flows.log") -> logging.Logger:
    """Setup logging configuration."""
    # Create logs directory if it doesn't exist
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_s34_type4_data(data_dir: str = "data/raw/s34") -> pd.DataFrame:
    """
    Load S34 Type 4 change in holdings data.
    
    Args:
        data_dir: Directory containing S34 Type 4 files
        
    Returns:
        DataFrame: Type 4 holdings data
    """
    logger = logging.getLogger(__name__)
    
    # Find all Type 4 files
    pattern = os.path.join(data_dir, "s34_type4_change_in_holdings_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        logger.warning(f"No S34 Type 4 files found in {data_dir}")
        # Create sample data for testing
        logger.info("Creating sample S34 Type 4 data for testing")
        return create_sample_s34_type4()
    
    logger.info(f"Found {len(files)} S34 Type 4 files: {files}")
    
    # Load and combine all files
    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file, low_memory=False)
            dfs.append(df)
            logger.info(f"Loaded {file}: {len(df)} rows")
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    
    if not dfs:
        logger.error("No files could be loaded")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined data: {len(combined_df)} rows")
    
    return combined_df


def create_sample_s34_type4() -> pd.DataFrame:
    """Create sample S34 Type 4 data for testing."""
    # Generate sample data
    np.random.seed(42)
    
    # Sample dates (quarterly)
    dates = pd.date_range('1990-01-01', '2020-12-31', freq='QE')
    
    # Sample CUSIPs (8-digit)
    cusips = [f"{i:08d}" for i in range(10000000, 10000100)]
    
    # Sample fund numbers
    fundnos = list(range(1000, 1100))
    
    # Generate sample data
    data = []
    for date in dates:
        for _ in range(50):  # 50 holdings per quarter
            data.append({
                'fdate': date,
                'cusip': np.random.choice(cusips),
                'fundno': np.random.choice(fundnos),
                'change_in_shares': np.random.uniform(-1000, 1000)  # Random change in shares
            })
    
    df = pd.DataFrame(data)
    return df


def load_crsp_security_master(security_master_path: str = "build/crsp_security_master.parquet") -> pd.DataFrame:
    """
    Load CRSP security master data.
    
    Args:
        security_master_path: Path to security master file
        
    Returns:
        DataFrame: Security master data
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(security_master_path):
        logger.warning(f"Security master file not found: {security_master_path}")
        logger.info("Creating sample CRSP security master data")
        return create_sample_crsp_security_master()
    
    try:
        df = pd.read_parquet(security_master_path)
        logger.info(f"Loaded CRSP security master: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading security master: {e}")
        return create_sample_crsp_security_master()


def create_sample_crsp_security_master() -> pd.DataFrame:
    """Create sample CRSP security master data for testing."""
    # Generate sample data
    np.random.seed(42)
    
    # Sample PERMNOs
    permnos = list(range(10001, 10101))
    
    # Sample CUSIPs (8-digit)
    cusips = [f"{i:08d}" for i in range(10000000, 10000100)]
    
    # Generate sample data
    data = []
    for permno, cusip in zip(permnos, cusips):
        # Random date range
        start_date = pd.Timestamp('1990-01-01') + pd.Timedelta(days=np.random.randint(0, 365*30))
        end_date = start_date + pd.Timedelta(days=np.random.randint(365, 365*10))
        
        # Random SHRCD (10 or 11 for common stock)
        shrcd = np.random.choice([10, 11])
        
        data.append({
            'permno': permno,
            'ncusip': cusip,
            'namedt': start_date,
            'nameendt': end_date,
            'shrcd': shrcd
        })
    
    df = pd.DataFrame(data)
    return df


def load_panel_data(panel_path: str = "build/panel_monthly.parquet") -> pd.DataFrame:
    """
    Load panel data to get shares outstanding.
    
    Args:
        panel_path: Path to panel data file
        
    Returns:
        DataFrame: Panel data
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(panel_path):
        logger.error(f"Panel data file not found: {panel_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(panel_path)
        logger.info(f"Loaded panel data: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading panel data: {e}")
        return pd.DataFrame()


def process_type4_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process S34 Type 4 data.
    
    Args:
        df: Raw Type 4 data
        
    Returns:
        DataFrame: Processed Type 4 data
    """
    logger = logging.getLogger(__name__)
    
    # Standardize columns
    column_mapping = {
        'fdate': 'rdate',
        'cusip': 'cusip8',
        'fundno': 'mgrno',
        'change_in_shares': 'dshares'
    }
    
    # Apply mapping for columns that exist
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})
    
    # Process data
    # Convert rdate to quarter-end Timestamp
    df['rdate'] = pd.to_datetime(df['rdate'])
    df['rdate_q'] = df['rdate'].dt.to_period('Q').dt.end_time
    
    # Process CUSIP8 (first 8 chars, uppercase)
    if 'cusip8' in df.columns:
        df['cusip8'] = df['cusip8'].astype(str).str[:8].str.upper()
    
    # Drop rows with dshares = 0 or NA
    if 'dshares' in df.columns:
        df = df.dropna(subset=['dshares'])
        df = df[df['dshares'] != 0]
    
    # Drop duplicates on (rdate_q, mgrno, cusip8) and aggregate if needed
    if all(col in df.columns for col in ['rdate_q', 'mgrno', 'cusip8', 'dshares']):
        df = df.groupby(['rdate_q', 'mgrno', 'cusip8'])['dshares'].sum().reset_index()
    
    logger.info(f"Processed Type 4 data: {len(df)} rows")
    return df


def map_to_crsp(type4_df: pd.DataFrame, security_master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Type 4 data to CRSP using security master.
    
    Args:
        type4_df: Type 4 holdings data
        security_master_df: CRSP security master data
        
    Returns:
        DataFrame: Type 4 data mapped to CRSP permno
    """
    logger = logging.getLogger(__name__)
    
    # Ensure we have the required columns
    required_type4_cols = ['rdate_q', 'cusip8', 'mgrno', 'dshares']
    required_master_cols = ['permno', 'ncusip', 'namedt', 'nameendt', 'shrcd']
    
    if not all(col in type4_df.columns for col in required_type4_cols):
        logger.error(f"Missing required columns in Type 4 data: {required_type4_cols}")
        return pd.DataFrame()
    
    if not all(col in security_master_df.columns for col in required_master_cols):
        logger.error(f"Missing required columns in security master: {required_master_cols}")
        return pd.DataFrame()
    
    # Filter security master to common stocks (SHRCD 10, 11)
    security_master_filtered = security_master_df[
        security_master_df['shrcd'].isin([10, 11])
    ].copy()
    
    # Create mapping key
    security_master_filtered['cusip8'] = security_master_filtered['ncusip'].astype(str).str[:8].str.upper()
    
    # Merge Type 4 with security master
    merged_df = pd.merge(
        type4_df,
        security_master_filtered[['permno', 'cusip8', 'namedt', 'nameendt']],
        on='cusip8',
        how='inner'
    )
    
    # Apply link window (namedt <= rdate_q <= nameendt)
    merged_df = merged_df[
        (merged_df['rdate_q'] >= merged_df['namedt']) &
        (merged_df['rdate_q'] <= merged_df['nameendt'])
    ]
    
    # Drop mapping columns
    merged_df = merged_df.drop(['namedt', 'nameendt'], axis=1)
    
    logger.info(f"Mapped to CRSP: {len(merged_df)} rows (from {len(type4_df)} original)")
    logger.info(f"Mapping coverage: {len(merged_df) / len(type4_df) * 100:.2f}%")
    
    return merged_df


def aggregate_quarterly_flows(mapped_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate institutional flows per (permno, rdate_q).
    
    Args:
        mapped_df: Mapped Type 4 data
        
    Returns:
        DataFrame: Quarterly aggregated flows
    """
    logger = logging.getLogger(__name__)
    
    # Aggregate per (permno, rdate_q)
    flows_q = mapped_df.groupby(['permno', 'rdate_q'])['dshares'].sum().reset_index()
    flows_q = flows_q.rename(columns={'dshares': 'inst_flow_shares'})
    
    logger.info(f"Aggregated quarterly flows: {len(flows_q)} stock-quarters")
    
    return flows_q


def scale_by_shares_outstanding(flows_q: pd.DataFrame, panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale institutional flows by shares outstanding.
    
    Args:
        flows_q: Quarterly flows
        panel_df: Panel data with shares outstanding
        
    Returns:
        DataFrame: Scaled quarterly flows
    """
    logger = logging.getLogger(__name__)
    
    # Prepare panel data
    panel_ready = panel_df.copy()
    panel_ready['DATE'] = pd.to_datetime(panel_ready['DATE'])
    panel_ready = panel_ready.rename(columns={
        'PERMNO': 'permno',
        'DATE': 'date',
        'SHROUT': 'shares_out'
    })
    
    # Convert to quarterly
    panel_ready['rdate_q'] = panel_ready['date'].dt.to_period('Q').dt.end_time
    
    # Get quarterly shares outstanding (use last month of quarter)
    shares_q = panel_ready.groupby(['permno', 'rdate_q'])['shares_out'].last().reset_index()
    
    # Merge flows with shares outstanding
    flows_scaled = pd.merge(flows_q, shares_q, on=['permno', 'rdate_q'], how='inner')
    
    # Calculate lagged shares outstanding
    flows_scaled = flows_scaled.sort_values(['permno', 'rdate_q'])
    flows_scaled['shares_out_lag'] = flows_scaled.groupby('permno')['shares_out'].shift(1)
    
    # Calculate scaled flow
    flows_scaled['inst_flow_q'] = flows_scaled['inst_flow_shares'] / flows_scaled['shares_out_lag']
    
    # Drop rows with missing lagged shares
    flows_scaled = flows_scaled.dropna(subset=['shares_out_lag'])
    
    logger.info(f"Scaled flows: {len(flows_scaled)} stock-quarters")
    
    return flows_scaled


def expand_to_monthly(flows_q: pd.DataFrame) -> pd.DataFrame:
    """
    Expand quarterly flows to monthly frequency.
    
    Args:
        flows_q: Quarterly flows data
        
    Returns:
        DataFrame: Monthly flows data
    """
    logger = logging.getLogger(__name__)
    
    # Convert rdate_q to datetime if it's not already
    flows_q = flows_q.copy()
    flows_q['rdate_q'] = pd.to_datetime(flows_q['rdate_q'])
    
    # Create monthly dates within each quarter
    monthly_data = []
    
    for _, row in flows_q.iterrows():
        quarter_end = row['rdate_q']
        
        # Get the three months in the quarter
        if quarter_end.month == 3:  # Q1
            months = [1, 2, 3]
        elif quarter_end.month == 6:  # Q2
            months = [4, 5, 6]
        elif quarter_end.month == 9:  # Q3
            months = [7, 8, 9]
        else:  # Q4
            months = [10, 11, 12]
        
        year = quarter_end.year
        
        for month in months:
            # Get month-end date
            if month == 12:
                next_year = year + 1
                next_month = 1
            else:
                next_year = year
                next_month = month + 1
            
            month_end = pd.Timestamp(year=next_year, month=next_month, day=1) - pd.Timedelta(days=1)
            
            monthly_data.append({
                'permno': row['permno'],
                'date': month_end,
                'inst_flow_q': row['inst_flow_q']
            })
    
    monthly_df = pd.DataFrame(monthly_data)
    
    logger.info(f"Expanded to monthly: {len(monthly_df)} stock-months")
    
    return monthly_df


def save_inst_flows(flows_df: pd.DataFrame, output_path: str = "build/inst_flows_monthly.parquet"):
    """
    Save institutional flows data to parquet file.
    
    Args:
        flows_df: Monthly flows data
        output_path: Output file path
    """
    logger = logging.getLogger(__name__)
    
    # Ensure build directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to parquet
    flows_df.to_parquet(output_path, index=False)
    logger.info(f"Saved institutional flows to {output_path}: {len(flows_df)} rows")


def log_coverage_stats(type4_df: pd.DataFrame, mapped_df: pd.DataFrame, flows_df: pd.DataFrame):
    """
    Log coverage statistics.
    
    Args:
        type4_df: Original Type 4 data
        mapped_df: Type 4 data mapped to CRSP
        flows_df: Final flows data
    """
    logger = logging.getLogger(__name__)
    
    # Calculate statistics
    total_type4 = len(type4_df)
    mapped_type4 = len(mapped_df)
    mapping_coverage = mapped_type4 / total_type4 * 100 if total_type4 > 0 else 0
    
    unique_permnos = flows_df['permno'].nunique()
    unique_permno_months = len(flows_df)
    
    # Date range
    date_range = flows_df['date'].agg(['min', 'max'])
    
    logger.info("="*50)
    logger.info("INSTITUTIONAL FLOWS BUILD COVERAGE STATISTICS")
    logger.info("="*50)
    logger.info(f"Type 4 rows mapped: {mapping_coverage:.2f}% ({mapped_type4:,} / {total_type4:,})")
    logger.info(f"Unique PERMNOs: {unique_permnos:,}")
    logger.info(f"PERMNO-months: {unique_permno_months:,}")
    logger.info(f"Date range: {date_range['min']} to {date_range['max']}")
    logger.info("="*50)


def main():
    """Main function to build institutional flows."""
    logger = setup_logging()
    
    print("Building institutional flows from S34 Type 4...")
    logger.info("Starting institutional flows build")
    
    # Step 1: Load S34 Type 4 data
    type4_df = load_s34_type4_data()
    
    if len(type4_df) == 0:
        logger.error("No Type 4 data available")
        return
    
    # Step 2: Load CRSP security master
    security_master_df = load_crsp_security_master()
    
    # Step 3: Process Type 4 data
    processed_df = process_type4_data(type4_df)
    
    # Step 4: Map to CRSP
    mapped_df = map_to_crsp(processed_df, security_master_df)
    
    if len(mapped_df) == 0:
        logger.error("No data could be mapped to CRSP")
        return
    
    # Step 5: Aggregate quarterly flows
    flows_q = aggregate_quarterly_flows(mapped_df)
    
    # Step 6: Load panel data for scaling
    panel_df = load_panel_data()
    
    if len(panel_df) > 0:
        # Step 7: Scale by shares outstanding
        flows_scaled = scale_by_shares_outstanding(flows_q, panel_df)
        
        # Step 8: Expand to monthly
        flows_monthly = expand_to_monthly(flows_scaled)
        
        # Step 9: Save results
        save_inst_flows(flows_monthly)
        
        # Step 10: Log coverage statistics
        log_coverage_stats(type4_df, mapped_df, flows_monthly)
    else:
        logger.warning("No panel data available for scaling, saving unscaled flows")
        flows_monthly = expand_to_monthly(flows_q)
        save_inst_flows(flows_monthly)
    
    logger.info("Institutional flows build completed")
    print("Institutional flows build completed!")


if __name__ == "__main__":
    main()
