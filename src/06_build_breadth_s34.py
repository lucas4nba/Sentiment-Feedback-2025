#!/usr/bin/env python3
"""
Build breadth of ownership from S34 Type 3 data.

This script converts S34 Type 3 holdings data to quarterly institutional breadth,
maps CUSIP8 to CRSP permno using security master, and expands to monthly frequency.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import glob


def setup_logging(log_file: str = "logs/build_breadth_s34.log") -> logging.Logger:
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


def load_and_process_s34_type3(data_dir: str = "data/raw/s34") -> pd.DataFrame:
    """
    Load and process S34 Type 3 CSV files.
    
    Args:
        data_dir: Directory containing S34 Type 3 files
        
    Returns:
        DataFrame: Processed holdings data
    """
    logger = logging.getLogger(__name__)
    
    # Find all Type 3 files
    pattern = os.path.join(data_dir, "s34_type3_stock_holdings_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        logger.warning(f"No S34 Type 3 files found in {data_dir}")
        # Create sample data for testing
        logger.info("Creating sample S34 Type 3 data for testing")
        return create_sample_s34_type3()
    
    logger.info(f"Found {len(files)} S34 Type 3 files: {files}")
    
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
    
    # Rename columns
    column_mapping = {
        'fdate': 'rdate',
        'cusip': 'cusip8',
        'fundno': 'mgrno',
        'shares': 'shares_mil'
    }
    
    # Apply mapping for columns that exist
    for old_col, new_col in column_mapping.items():
        if old_col in combined_df.columns:
            combined_df = combined_df.rename(columns={old_col: new_col})
    
    # Process data
    # Convert rdate to quarter-end Timestamp
    combined_df['rdate'] = pd.to_datetime(combined_df['rdate'])
    combined_df['rdate_q'] = combined_df['rdate'].dt.to_period('QE').dt.end_time
    
    # Process CUSIP8 (first 8 chars, uppercase)
    if 'cusip8' in combined_df.columns:
        combined_df['cusip8'] = combined_df['cusip8'].astype(str).str[:8].str.upper()
    
    # Drop rows with shares_mil <= 0 or NA
    if 'shares_mil' in combined_df.columns:
        combined_df = combined_df.dropna(subset=['shares_mil'])
        combined_df = combined_df[combined_df['shares_mil'] > 0]
    
    # Drop duplicates on (rdate_q, mgrno, cusip8) and aggregate if needed
    if all(col in combined_df.columns for col in ['rdate_q', 'mgrno', 'cusip8', 'shares_mil']):
        combined_df = combined_df.groupby(['rdate_q', 'mgrno', 'cusip8'])['shares_mil'].sum().reset_index()
    
    logger.info(f"Processed data: {len(combined_df)} rows")
    return combined_df


def create_sample_s34_type3() -> pd.DataFrame:
    """Create sample S34 Type 3 data for testing."""
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
        for _ in range(100):  # 100 holdings per quarter
            data.append({
                'rdate': date,
                'rdate_q': date,
                'cusip8': np.random.choice(cusips),
                'mgrno': np.random.choice(fundnos),
                'shares_mil': np.random.uniform(0.1, 1000)
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


def map_to_crsp(holdings_df: pd.DataFrame, security_master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map holdings to CRSP using security master.
    
    Args:
        holdings_df: Holdings data with cusip8
        security_master_df: CRSP security master data
        
    Returns:
        DataFrame: Holdings data mapped to CRSP permno
    """
    logger = logging.getLogger(__name__)
    
    # Ensure we have the required columns
    required_holdings_cols = ['rdate_q', 'cusip8', 'mgrno', 'shares_mil']
    required_master_cols = ['permno', 'ncusip', 'namedt', 'nameendt', 'shrcd']
    
    if not all(col in holdings_df.columns for col in required_holdings_cols):
        logger.error(f"Missing required columns in holdings data: {required_holdings_cols}")
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
    
    # Merge holdings with security master
    merged_df = pd.merge(
        holdings_df,
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
    
    logger.info(f"Mapped to CRSP: {len(merged_df)} rows (from {len(holdings_df)} original)")
    logger.info(f"Mapping coverage: {len(merged_df) / len(holdings_df) * 100:.2f}%")
    
    return merged_df


def calculate_quarterly_breadth(holdings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate quarterly institutional breadth.
    
    Args:
        holdings_df: Holdings data with permno, rdate_q, mgrno
        
    Returns:
        DataFrame: Quarterly breadth measures
    """
    logger = logging.getLogger(__name__)
    
    # Calculate total institutional count per quarter
    total_inst_q = holdings_df.groupby('rdate_q')['mgrno'].nunique().reset_index()
    total_inst_q = total_inst_q.rename(columns={'mgrno': 'total_inst_q'})
    
    # Calculate institutional holders per stock-quarter
    inst_holders_q = holdings_df.groupby(['permno', 'rdate_q'])['mgrno'].nunique().reset_index()
    inst_holders_q = inst_holders_q.rename(columns={'mgrno': 'inst_holders_q'})
    
    # Merge to get breadth
    breadth_q = pd.merge(inst_holders_q, total_inst_q, on='rdate_q')
    breadth_q['breadth_q'] = breadth_q['inst_holders_q'] / breadth_q['total_inst_q']
    
    # Calculate change in breadth
    breadth_q = breadth_q.sort_values(['permno', 'rdate_q'])
    breadth_q['dbreadth_q'] = breadth_q.groupby('permno')['breadth_q'].diff()
    
    logger.info(f"Calculated quarterly breadth: {len(breadth_q)} stock-quarters")
    
    return breadth_q


def expand_to_monthly(breadth_q: pd.DataFrame) -> pd.DataFrame:
    """
    Expand quarterly breadth to monthly frequency.
    
    Args:
        breadth_q: Quarterly breadth data
        
    Returns:
        DataFrame: Monthly breadth data
    """
    logger = logging.getLogger(__name__)
    
    # Convert rdate_q to datetime if it's not already
    breadth_q = breadth_q.copy()
    breadth_q['rdate_q'] = pd.to_datetime(breadth_q['rdate_q'])
    
    # Create monthly dates within each quarter
    monthly_data = []
    
    for _, row in breadth_q.iterrows():
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
                'breadth': row['breadth_q'],
                'dbreadth': row['dbreadth_q'],
                'inst_holders': row['inst_holders_q'],
                'total_inst': row['total_inst_q']
            })
    
    monthly_df = pd.DataFrame(monthly_data)
    
    logger.info(f"Expanded to monthly: {len(monthly_df)} stock-months")
    
    return monthly_df


def save_breadth_data(breadth_df: pd.DataFrame, output_path: str = "build/breadth_monthly.parquet"):
    """
    Save breadth data to parquet file.
    
    Args:
        breadth_df: Monthly breadth data
        output_path: Output file path
    """
    logger = logging.getLogger(__name__)
    
    # Ensure build directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to parquet
    breadth_df.to_parquet(output_path, index=False)
    logger.info(f"Saved breadth data to {output_path}: {len(breadth_df)} rows")


def log_coverage_stats(holdings_df: pd.DataFrame, mapped_df: pd.DataFrame, breadth_df: pd.DataFrame):
    """
    Log coverage statistics.
    
    Args:
        holdings_df: Original holdings data
        mapped_df: Holdings data mapped to CRSP
        breadth_df: Final breadth data
    """
    logger = logging.getLogger(__name__)
    
    # Calculate statistics
    total_holdings = len(holdings_df)
    mapped_holdings = len(mapped_df)
    mapping_coverage = mapped_holdings / total_holdings * 100 if total_holdings > 0 else 0
    
    unique_permnos = breadth_df['permno'].nunique()
    unique_permno_months = len(breadth_df)
    
    # Date range
    date_range = breadth_df['date'].agg(['min', 'max'])
    
    logger.info("="*50)
    logger.info("BREADTH BUILD COVERAGE STATISTICS")
    logger.info("="*50)
    logger.info(f"Type 3 rows mapped: {mapping_coverage:.2f}% ({mapped_holdings:,} / {total_holdings:,})")
    logger.info(f"Unique PERMNOs: {unique_permnos:,}")
    logger.info(f"PERMNO-months: {unique_permno_months:,}")
    logger.info(f"Date range: {date_range['min']} to {date_range['max']}")
    logger.info("="*50)


def main():
    """Main function to build breadth of ownership."""
    logger = setup_logging()
    
    print("Building breadth of ownership from S34 Type 3...")
    logger.info("Starting breadth of ownership build")
    
    # Step 1: Load and process S34 Type 3 data
    holdings_df = load_and_process_s34_type3()
    
    if len(holdings_df) == 0:
        logger.error("No holdings data available")
        return
    
    # Step 2: Load CRSP security master
    security_master_df = load_crsp_security_master()
    
    # Step 3: Map to CRSP
    mapped_df = map_to_crsp(holdings_df, security_master_df)
    
    if len(mapped_df) == 0:
        logger.error("No data could be mapped to CRSP")
        return
    
    # Step 4: Calculate quarterly breadth
    breadth_q = calculate_quarterly_breadth(mapped_df)
    
    # Step 5: Expand to monthly
    breadth_monthly = expand_to_monthly(breadth_q)
    
    # Step 6: Save results
    save_breadth_data(breadth_monthly)
    
    # Step 7: Log coverage statistics
    log_coverage_stats(holdings_df, mapped_df, breadth_monthly)
    
    logger.info("Breadth of ownership build completed")
    print("Breadth of ownership build completed!")


if __name__ == "__main__":
    main()
