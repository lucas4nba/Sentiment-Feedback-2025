"""
Comprehensive Quality Control Script for Financial Data Pipeline
Analyzes schema, coverage, missingness, and data quality across all datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.config import OUT

def analyze_dates(df, date_col='DATE', dataset_name=''):
    """Analyze date patterns and month-end alignment"""
    if date_col not in df.columns:
        return {
            'dataset': dataset_name,
            'date_analysis': 'NO_DATE_COLUMN',
            'date_range': 'N/A',
            'month_end_pct': 0,
            'date_gaps': 'N/A'
        }
    
    dates = pd.to_datetime(df[date_col])
    
    # Check month-end alignment
    month_ends = dates.dt.is_month_end
    month_end_pct = month_ends.mean() * 100
    
    # Date range
    date_range = f"{dates.min().strftime('%Y-%m')} to {dates.max().strftime('%Y-%m')}"
    
    # Check for gaps (monthly data)
    date_series = dates.dt.to_period('M').drop_duplicates().sort_values()
    if len(date_series) > 1:
        expected_periods = pd.period_range(date_series.min(), date_series.max(), freq='M')
        missing_periods = set(expected_periods) - set(date_series)
        gap_info = f"{len(missing_periods)} gaps" if missing_periods else "No gaps"
    else:
        gap_info = "Insufficient data"
    
    return {
        'dataset': dataset_name,
        'date_range': date_range,
        'month_end_pct': round(month_end_pct, 1),
        'date_gaps': gap_info,
        'unique_dates': len(dates.unique()),
        'total_obs': len(df)
    }

def analyze_missingness(df, dataset_name=''):
    """Analyze missing data patterns"""
    missing_stats = []
    
    for col in df.columns:
        if col == 'DATE':
            continue
            
        total_obs = len(df)
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / total_obs) * 100
        
        # Data type
        dtype = str(df[col].dtype)
        
        # For numeric columns, get basic stats
        if df[col].dtype in ['int64', 'float64']:
            if missing_count < total_obs:
                non_missing = df[col].dropna()
                stats_info = f"mean={non_missing.mean():.3f}, std={non_missing.std():.3f}"
            else:
                stats_info = "all_missing"
        else:
            unique_vals = df[col].nunique()
            stats_info = f"unique_values={unique_vals}"
        
        missing_stats.append({
            'dataset': dataset_name,
            'column': col,
            'missing_count': missing_count,
            'missing_pct': round(missing_pct, 2),
            'dtype': dtype,
            'stats': stats_info
        })
    
    return missing_stats

def analyze_coverage_by_date(df, date_col='DATE', dataset_name=''):
    """Analyze data coverage over time"""
    if date_col not in df.columns:
        return pd.DataFrame()
    
    # Group by date and count non-missing values for each column
    coverage_by_date = []
    
    for date in df[date_col].unique():
        date_subset = df[df[date_col] == date]
        
        for col in df.columns:
            if col == date_col:
                continue
                
            non_missing = (~date_subset[col].isna()).sum()
            total = len(date_subset)
            coverage_pct = (non_missing / total) * 100 if total > 0 else 0
            
            coverage_by_date.append({
                'dataset': dataset_name,
                'date': date,
                'column': col,
                'coverage_pct': coverage_pct,
                'non_missing_count': non_missing,
                'total_obs': total
            })
    
    return pd.DataFrame(coverage_by_date)

def create_coverage_plot(coverage_data, output_path):
    """Create coverage visualization"""
    try:
        # Filter to market-level datasets (not stock-level)
        market_datasets = ['macro_monthly', 'sentiment_monthly', 'option_iv_monthly', 
                          'controls_monthly', 'flows_market_monthly']
        
        market_coverage = coverage_data[coverage_data['dataset'].isin(market_datasets)]
        
        if market_coverage.empty:
            print("No market-level data found for coverage plot")
            return
        
        # Create pivot table for heatmap
        pivot_data = market_coverage.pivot_table(
            values='coverage_pct', 
            index=['dataset', 'column'], 
            columns='date', 
            fill_value=0
        )
        
        # Create plot
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # Convert dates to strings for better display
        pivot_data.columns = [pd.to_datetime(col).strftime('%Y-%m') if pd.notnull(col) else 'NaT' 
                             for col in pivot_data.columns]
        
        # Sample dates if too many (show every 12th month)
        if len(pivot_data.columns) > 50:
            step = max(1, len(pivot_data.columns) // 50)
            pivot_data = pivot_data.iloc[:, ::step]
        
        sns.heatmap(pivot_data, 
                   cmap='RdYlGn', 
                   cbar_kws={'label': 'Coverage %'},
                   ax=ax,
                   vmin=0, vmax=100)
        
        ax.set_title('Data Coverage Over Time (Market-Level Variables)', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Dataset.Column', fontsize=12)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Coverage plot saved to: {output_path}")
        
    except Exception as e:
        print(f"Error creating coverage plot: {e}")

def main():
    """Main QC analysis function"""
    print("Starting comprehensive QC analysis...")
    print("=" * 60)
    
    # Find all parquet files in build directory
    parquet_files = list(OUT.glob("*.parquet"))
    
    if not parquet_files:
        print("No parquet files found in build directory!")
        return
    
    print(f"Found {len(parquet_files)} datasets to analyze:")
    for f in parquet_files:
        print(f"  - {f.name}")
    print()
    
    # Initialize results storage
    all_date_analysis = []
    all_missing_analysis = []
    all_coverage_data = []
    dataset_summaries = []
    
    # Analyze each dataset
    for file_path in parquet_files:
        dataset_name = file_path.stem
        print(f"Analyzing {dataset_name}...")
        
        try:
            # Load dataset
            df = pd.read_parquet(file_path)
            
            # Basic dataset info
            dataset_summaries.append({
                'dataset': dataset_name,
                'rows': len(df),
                'columns': len(df.columns),
                'file_size_mb': round(file_path.stat().st_size / (1024*1024), 2),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024*1024), 2)
            })
            
            # Date analysis
            date_analysis = analyze_dates(df, dataset_name=dataset_name)
            all_date_analysis.append(date_analysis)
            
            # Missingness analysis
            missing_analysis = analyze_missingness(df, dataset_name=dataset_name)
            all_missing_analysis.extend(missing_analysis)
            
            # Coverage analysis (for time series datasets)
            if 'DATE' in df.columns and dataset_name != 'stock_id_master':
                coverage_data = analyze_coverage_by_date(df, dataset_name=dataset_name)
                if not coverage_data.empty:
                    all_coverage_data.append(coverage_data)
            
            print(f"  ✓ {dataset_name}: {len(df)} rows, {len(df.columns)} columns")
            
        except Exception as e:
            print(f"  ✗ Error analyzing {dataset_name}: {e}")
            continue
    
    print("\nGenerating summary reports...")
    
    # Create summary DataFrames
    date_summary_df = pd.DataFrame(all_date_analysis)
    missing_summary_df = pd.DataFrame(all_missing_analysis)
    dataset_summary_df = pd.DataFrame(dataset_summaries)
    
    # Combine coverage data
    if all_coverage_data:
        coverage_df = pd.concat(all_coverage_data, ignore_index=True)
    else:
        coverage_df = pd.DataFrame()
    
    # Generate QC summary report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save individual components
    date_summary_df.to_csv(OUT / f"qc_date_analysis_{timestamp}.csv", index=False)
    missing_summary_df.to_csv(OUT / f"qc_missing_analysis_{timestamp}.csv", index=False)
    dataset_summary_df.to_csv(OUT / f"qc_dataset_summary_{timestamp}.csv", index=False)
    
    if not coverage_df.empty:
        coverage_df.to_csv(OUT / f"qc_coverage_data_{timestamp}.csv", index=False)
    
    # Create comprehensive summary
    qc_summary = []
    
    # Dataset overview
    qc_summary.append("=== DATASET OVERVIEW ===")
    qc_summary.append(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    qc_summary.append(f"Total datasets analyzed: {len(dataset_summaries)}")
    qc_summary.append(f"Total disk space: {sum(s['file_size_mb'] for s in dataset_summaries):.1f} MB")
    qc_summary.append("")
    
    for _, row in dataset_summary_df.iterrows():
        qc_summary.append(f"{row['dataset']}: {row['rows']:,} rows × {row['columns']} cols ({row['file_size_mb']} MB)")
    qc_summary.append("")
    
    # Date analysis summary
    qc_summary.append("=== DATE ANALYSIS ===")
    for _, row in date_summary_df.iterrows():
        qc_summary.append(f"{row['dataset']}:")
        qc_summary.append(f"  Date range: {row['date_range']}")
        qc_summary.append(f"  Month-end alignment: {row['month_end_pct']}%")
        qc_summary.append(f"  Date gaps: {row['date_gaps']}")
        qc_summary.append("")
    
    # Top missingness issues
    qc_summary.append("=== TOP MISSINGNESS ISSUES ===")
    top_missing = missing_summary_df.nlargest(20, 'missing_pct')
    for _, row in top_missing.iterrows():
        if row['missing_pct'] > 0:
            qc_summary.append(f"{row['dataset']}.{row['column']}: {row['missing_pct']}% missing ({row['missing_count']:,} obs)")
    qc_summary.append("")
    
    # Schema summary
    qc_summary.append("=== SCHEMA SUMMARY ===")
    schema_summary = missing_summary_df.groupby(['dataset', 'dtype']).size().reset_index(name='count')
    for dataset in schema_summary['dataset'].unique():
        dataset_schema = schema_summary[schema_summary['dataset'] == dataset]
        qc_summary.append(f"{dataset}:")
        for _, row in dataset_schema.iterrows():
            qc_summary.append(f"  {row['dtype']}: {row['count']} columns")
        qc_summary.append("")
    
    # Write summary file
    summary_text = "\n".join(qc_summary)
    summary_file = OUT / "qc_summary.csv"
    
    # Also create a proper CSV version
    qc_summary_data = []
    
    # Add key metrics to CSV
    for _, row in dataset_summary_df.iterrows():
        date_info = date_summary_df[date_summary_df['dataset'] == row['dataset']].iloc[0] if len(date_summary_df) > 0 else {}
        
        qc_summary_data.append({
            'dataset': row['dataset'],
            'rows': row['rows'],
            'columns': row['columns'],
            'file_size_mb': row['file_size_mb'],
            'date_range': date_info.get('date_range', 'N/A'),
            'month_end_pct': date_info.get('month_end_pct', 0),
            'date_gaps': date_info.get('date_gaps', 'N/A'),
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    qc_summary_csv = pd.DataFrame(qc_summary_data)
    qc_summary_csv.to_csv(summary_file, index=False)
    
    # Also save text summary
    with open(OUT / "qc_summary.txt", 'w') as f:
        f.write(summary_text)
    
    print(f"\n✓ QC Summary saved to: {summary_file}")
    print(f"✓ Detailed text summary: {OUT / 'qc_summary.txt'}")
    
    # Create coverage plot
    if not coverage_df.empty:
        print("\nGenerating coverage plot...")
        plot_path = OUT / "coverage_plot.png"
        create_coverage_plot(coverage_df, plot_path)
    
    print("\n" + "="*60)
    print("QC ANALYSIS COMPLETE")
    print("="*60)
    
    # Print key findings
    print(f"\nKEY FINDINGS:")
    print(f"• Total observations across all datasets: {sum(s['rows'] for s in dataset_summaries):,}")
    print(f"• Largest dataset: {max(dataset_summaries, key=lambda x: x['rows'])['dataset']} ({max(s['rows'] for s in dataset_summaries):,} rows)")
    print(f"• Date coverage: {date_summary_df['date_range'].iloc[0] if len(date_summary_df) > 0 else 'N/A'}")
    
    # Highlight any major issues
    high_missing = missing_summary_df[missing_summary_df['missing_pct'] > 50]
    if not high_missing.empty:
        print(f"• WARNING: {len(high_missing)} variables with >50% missing data")
    
    non_month_end = date_summary_df[date_summary_df['month_end_pct'] < 90]
    if not non_month_end.empty:
        print(f"• WARNING: {len(non_month_end)} datasets with poor month-end alignment")

if __name__ == "__main__":
    main()



