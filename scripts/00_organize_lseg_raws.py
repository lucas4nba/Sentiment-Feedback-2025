#!/usr/bin/env python3
"""
Organize LSEG raw data files by detecting file type and moving to appropriate directories.

This script scans data/ and data/raw/ for loose CSVs exported from WRDS (S34 Type 2/3/4 and LSEG common files)
and organizes them based on their column structure.
"""

import os
import sys
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import re


def setup_logging(log_file: str = "logs/organize_lseg_raws.log") -> logging.Logger:
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


def normalize_column_names(columns: List[str]) -> List[str]:
    """Normalize column names to lowercase."""
    return [col.lower().strip() for col in columns]


def detect_file_type(columns: List[str]) -> str:
    """
    Detect file type based on column patterns.
    
    Returns:
        str: File type ('s34_type3', 's34_type2', 's34_type4', 'security_master', 'security_mapping', 'unknown')
    """
    normalized_cols = normalize_column_names(columns)
    
    # Type 3: contains (case-insensitive) {'fdate','cusip','fundno'}; 'shares' or 'shares held' in columns
    if all(col in normalized_cols for col in ['fdate', 'cusip', 'fundno']):
        if any('shares' in col for col in normalized_cols):
            return 's34_type3'
    
    # Type 2: contains {'cusip'} and one of {'title','class','security type'}
    if 'cusip' in normalized_cols:
        if any(col in normalized_cols for col in ['title', 'class', 'security type']):
            return 's34_type2'
    
    # Type 4: contains {'fdate','cusip','fundno'} and a column with 'change' AND 'shares' or 'value'
    if all(col in normalized_cols for col in ['fdate', 'cusip', 'fundno']):
        has_change = any('change' in col for col in normalized_cols)
        has_shares_or_value = any(any(term in col for term in ['shares', 'value']) for col in normalized_cols)
        if has_change and has_shares_or_value:
            return 's34_type4'
    
    # Security Master/Mapping: contains {'startdate','enddate'} and at least one of {'isin','sedol','ric','exchange','typ','vencode','rank'}
    if all(col in normalized_cols for col in ['startdate', 'enddate']):
        lseg_indicators = ['isin', 'sedol', 'ric', 'exchange', 'typ', 'vencode', 'rank']
        if any(col in normalized_cols for col in lseg_indicators):
            # Distinguish between master and mapping based on additional columns
            if any(col in normalized_cols for col in ['name', 'country', 'seccode']):
                return 'security_master'
            else:
                return 'security_mapping'
    
    # Additional patterns based on observed data:
    
    # Simple S34-like files with ticker, fdate, cusip (could be Type 2 or 3)
    if all(col in normalized_cols for col in ['ticker', 'fdate', 'cusip']):
        return 's34_type2'  # Default to Type 2 for simple files
    
    # Security mapping files with ISIN, SEDOL, CUSIP identifiers
    if 'isin' in normalized_cols and ('sedol' in normalized_cols or 'cusip' in normalized_cols):
        if any(col in normalized_cols for col in ['name', 'country', 'seccode']):
            return 'security_master'
        else:
            return 'security_mapping'
    
    return 'unknown'


def extract_date_range(df: pd.DataFrame, date_columns: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract min and max dates from the dataframe.
    
    Args:
        df: DataFrame to analyze
        date_columns: List of potential date column names
    
    Returns:
        Tuple of (min_date, max_date) in YYYY-MM format, or (None, None) if no valid dates found
    """
    min_date = None
    max_date = None
    
    for date_col in date_columns:
        if date_col in df.columns:
            try:
                # Convert to datetime, handling various formats
                date_series = pd.to_datetime(df[date_col], errors='coerce')
                date_series = date_series.dropna()
                
                if len(date_series) > 0:
                    min_dt = date_series.min()
                    max_dt = date_series.max()
                    
                    min_date = min_dt.strftime('%Y-%m')
                    max_date = max_dt.strftime('%Y-%m')
                    break
                    
            except Exception as e:
                logger.warning(f"Could not parse dates from column {date_col}: {e}")
                continue
    
    return min_date, max_date


def get_destination_path(file_type: str, min_date: Optional[str], max_date: Optional[str], 
                        base_dir: str = "data/raw") -> str:
    """
    Generate destination path based on file type and date range.
    
    Args:
        file_type: Detected file type
        min_date: Minimum date in YYYY-MM format
        max_date: Maximum date in YYYY-MM format
        base_dir: Base directory for organization
    
    Returns:
        str: Destination path
    """
    # Create base directories
    if file_type.startswith('s34'):
        subdir = "s34"
    else:
        subdir = "lseg_common"
    
    # Generate filename based on type
    if file_type == 's34_type3':
        filename = "s34_type3_stock_holdings"
    elif file_type == 's34_type2':
        filename = "s34_type2_stock_chars"
    elif file_type == 's34_type4':
        filename = "s34_type4_change_in_holdings"
    elif file_type == 'security_master':
        filename = "security_master"
    elif file_type == 'security_mapping':
        filename = "security_mapping"
    else:
        filename = "unknown_type"
    
    # Add date suffix if available
    if min_date and max_date:
        filename += f"_{min_date}_{max_date}"
    
    filename += ".csv"
    
    return os.path.join(base_dir, subdir, filename)


def ensure_unique_filename(filepath: str) -> str:
    """
    Ensure filename is unique by appending numeric suffix if needed.
    
    Args:
        filepath: Original filepath
    
    Returns:
        str: Unique filepath
    """
    if not os.path.exists(filepath):
        return filepath
    
    base_path = filepath[:-4]  # Remove .csv
    extension = ".csv"
    counter = 1
    
    while os.path.exists(filepath):
        filepath = f"{base_path}_{counter}{extension}"
        counter += 1
    
    return filepath


def process_csv_file(file_path: str, logger: logging.Logger) -> Dict:
    """
    Process a single CSV file to detect type and extract metadata.
    
    Args:
        file_path: Path to the CSV file
        logger: Logger instance
    
    Returns:
        Dict: Processing results
    """
    result = {
        'old_path': file_path,
        'new_path': None,
        'detected_type': 'unknown',
        'min_date': None,
        'max_date': None,
        'n_rows': 0,
        'error': None
    }
    
    try:
        # Read sample of the file (first 50k rows max for speed)
        df_sample = pd.read_csv(file_path, low_memory=False, nrows=50000)
        result['n_rows'] = len(df_sample)
        
        # Detect file type
        file_type = detect_file_type(df_sample.columns.tolist())
        result['detected_type'] = file_type
        
        if file_type == 'unknown':
            logger.warning(f"Could not determine file type for {file_path}")
            return result
        
        # Extract date range
        date_columns = ['fdate', 'filedate', 'startdate', 'enddate']
        min_date, max_date = extract_date_range(df_sample, date_columns)
        result['min_date'] = min_date
        result['max_date'] = max_date
        
        # Generate destination path
        new_path = get_destination_path(file_type, min_date, max_date)
        new_path = ensure_unique_filename(new_path)
        result['new_path'] = new_path
        
        logger.info(f"Processed {file_path}: type={file_type}, rows={result['n_rows']}, dates={min_date} to {max_date}")
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Error processing {file_path}: {e}")
    
    return result


def move_file(old_path: str, new_path: str, logger: logging.Logger) -> bool:
    """
    Move file to new location, creating directories as needed.
    
    Args:
        old_path: Source file path
        new_path: Destination file path
        logger: Logger instance
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create destination directory
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        
        # Move the file
        import shutil
        shutil.move(old_path, new_path)
        
        logger.info(f"Moved {old_path} -> {new_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error moving {old_path} to {new_path}: {e}")
        return False


def scan_and_organize_files(data_dirs: List[str] = None, logger: logging.Logger = None) -> List[Dict]:
    """
    Scan directories for CSV files and organize them.
    
    Args:
        data_dirs: List of directories to scan (default: ['data', 'data/raw'])
        logger: Logger instance
    
    Returns:
        List[Dict]: Results of processing all files
    """
    if data_dirs is None:
        data_dirs = ['data', 'data/raw']
    
    if logger is None:
        logger = setup_logging()
    
    results = []
    csv_files = []
    
    # Find all CSV files in the specified directories
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.lower().endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    # Process each file
    for csv_file in csv_files:
        result = process_csv_file(csv_file, logger)
        results.append(result)
        
        # Move file if processing was successful
        if result['new_path'] and not result['error']:
            success = move_file(result['old_path'], result['new_path'], logger)
            if not success:
                result['error'] = 'Failed to move file'
    
    return results


def print_summary(results: List[Dict]):
    """Print a neat summary of the organization results."""
    print("\n" + "="*80)
    print("LSEG RAW DATA ORGANIZATION SUMMARY")
    print("="*80)
    
    # Count by type
    type_counts = {}
    successful_moves = 0
    errors = 0
    
    for result in results:
        file_type = result['detected_type']
        type_counts[file_type] = type_counts.get(file_type, 0) + 1
        
        if result['error']:
            errors += 1
        elif result['new_path']:
            successful_moves += 1
    
    print(f"\nTotal files processed: {len(results)}")
    print(f"Successfully moved: {successful_moves}")
    print(f"Errors: {errors}")
    
    print(f"\nFile type distribution:")
    for file_type, count in sorted(type_counts.items()):
        print(f"  {file_type}: {count}")
    
    print(f"\nDetailed results:")
    print(f"{'Old Path':<50} {'Type':<15} {'New Path':<50}")
    print("-" * 115)
    
    for result in results:
        old_path = os.path.basename(result['old_path'])
        file_type = result['detected_type']
        new_path = os.path.basename(result['new_path']) if result['new_path'] else 'N/A'
        
        print(f"{old_path:<50} {file_type:<15} {new_path:<50}")
    
    print("\n" + "="*80)


def main():
    """Main function to run the organizer."""
    logger = setup_logging()
    
    print("Starting LSEG raw data organization...")
    logger.info("Starting LSEG raw data organization")
    
    # Scan and organize files
    results = scan_and_organize_files(logger=logger)
    
    # Print summary
    print_summary(results)
    
    logger.info("LSEG raw data organization completed")
    print("\nOrganization completed! Check logs/organize_lseg_raws.log for details.")


if __name__ == "__main__":
    main()
