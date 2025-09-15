#!/usr/bin/env python3
"""
Merge breadth of ownership data to the main panel.

This script merges the breadth data to the panel_ready.parquet file
and saves the result as panel_with_breadth.parquet.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def setup_logging(log_file: str = "logs/merge_breadth_to_panel.log") -> logging.Logger:
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


def load_panel_data(panel_path: str = "build/panel_monthly.parquet") -> pd.DataFrame:
    """
    Load the main panel data.
    
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


def load_breadth_data(breadth_path: str = "build/breadth_monthly.parquet") -> pd.DataFrame:
    """
    Load breadth of ownership data.
    
    Args:
        breadth_path: Path to breadth data file
        
    Returns:
        DataFrame: Breadth data
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(breadth_path):
        logger.error(f"Breadth data file not found: {breadth_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(breadth_path)
        logger.info(f"Loaded breadth data: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading breadth data: {e}")
        return pd.DataFrame()


def prepare_data_for_merge(panel_df: pd.DataFrame, breadth_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare panel and breadth data for merging.
    
    Args:
        panel_df: Panel data
        breadth_df: Breadth data
        
    Returns:
        Tuple of (panel_df, breadth_df) ready for merge
    """
    logger = logging.getLogger(__name__)
    
    # Prepare panel data
    panel_ready = panel_df.copy()
    
    # Ensure DATE is datetime
    panel_ready['DATE'] = pd.to_datetime(panel_ready['DATE'])
    
    # Rename columns to match merge keys
    panel_ready = panel_ready.rename(columns={
        'PERMNO': 'permno',
        'DATE': 'date'
    })
    
    # Prepare breadth data
    breadth_ready = breadth_df.copy()
    
    # Ensure date is datetime
    breadth_ready['date'] = pd.to_datetime(breadth_ready['date'])
    
    logger.info(f"Panel data prepared: {len(panel_ready)} rows")
    logger.info(f"Breadth data prepared: {len(breadth_ready)} rows")
    
    return panel_ready, breadth_ready


def merge_breadth_to_panel(panel_df: pd.DataFrame, breadth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge breadth data to panel data.
    
    Args:
        panel_df: Panel data
        breadth_df: Breadth data
        
    Returns:
        DataFrame: Merged panel with breadth
    """
    logger = logging.getLogger(__name__)
    
    # Merge on permno and date
    merged_df = pd.merge(
        panel_df,
        breadth_df,
        on=['permno', 'date'],
        how='left'
    )
    
    # Calculate merge statistics
    total_rows = len(panel_df)
    rows_with_breadth = merged_df['breadth'].notna().sum()
    merge_coverage = rows_with_breadth / total_rows * 100
    
    logger.info(f"Merge completed:")
    logger.info(f"  Total panel rows: {total_rows:,}")
    logger.info(f"  Rows with breadth: {rows_with_breadth:,}")
    logger.info(f"  Merge coverage: {merge_coverage:.2f}%")
    
    return merged_df


def save_merged_panel(merged_df: pd.DataFrame, output_path: str = "build/panel_with_breadth.parquet"):
    """
    Save merged panel data.
    
    Args:
        merged_df: Merged panel data
        output_path: Output file path
    """
    logger = logging.getLogger(__name__)
    
    # Ensure build directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to parquet
    merged_df.to_parquet(output_path, index=False)
    logger.info(f"Saved merged panel to {output_path}: {len(merged_df)} rows")


def print_merge_summary(panel_df: pd.DataFrame, merged_df: pd.DataFrame):
    """
    Print summary of the merge operation.
    
    Args:
        panel_df: Original panel data
        merged_df: Merged panel data
    """
    total_rows = len(panel_df)
    rows_with_breadth = merged_df['breadth'].notna().sum()
    merge_coverage = rows_with_breadth / total_rows * 100
    
    print("\n" + "="*60)
    print("BREADTH MERGE SUMMARY")
    print("="*60)
    print(f"Total panel rows: {total_rows:,}")
    print(f"Rows gaining breadth: {rows_with_breadth:,}")
    print(f"Merge coverage: {merge_coverage:.2f}%")
    
    # Date range statistics
    if 'breadth' in merged_df.columns:
        breadth_dates = merged_df[merged_df['breadth'].notna()]['date']
        if len(breadth_dates) > 0:
            print(f"Breadth date range: {breadth_dates.min()} to {breadth_dates.max()}")
    
    print("="*60)


def main():
    """Main function to merge breadth to panel."""
    logger = setup_logging()
    
    print("Merging breadth data to panel...")
    logger.info("Starting breadth merge to panel")
    
    # Load data
    panel_df = load_panel_data()
    breadth_df = load_breadth_data()
    
    if len(panel_df) == 0 or len(breadth_df) == 0:
        logger.error("Could not load required data files")
        return
    
    # Prepare data for merge
    panel_ready, breadth_ready = prepare_data_for_merge(panel_df, breadth_df)
    
    # Merge breadth to panel
    merged_df = merge_breadth_to_panel(panel_ready, breadth_ready)
    
    # Save merged panel
    save_merged_panel(merged_df)
    
    # Print summary
    print_merge_summary(panel_ready, merged_df)
    
    logger.info("Breadth merge to panel completed")
    print("Breadth merge to panel completed!")


if __name__ == "__main__":
    main()
