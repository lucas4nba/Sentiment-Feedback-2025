# Data Access Guide

This document describes data requirements for reproducing the Sentiment Feedback analysis, including what data is publicly available and what requires restricted access.

## Data Classification

### 🟢 **Public Data** (No credentials required)

#### Consumer Sentiment Data
- **Source**: University of Michigan Consumer Sentiment Survey (UMCSENT)
- **Access**: FRED (Federal Reserve Economic Data)
- **URL**: https://fred.stlouisfed.org/series/UMCSENT
- **Usage**: Main sentiment shock variable
- **File**: `sentiment_monthly.parquet`

#### Market Data
- **VIX**: CBOE Volatility Index
- **Source**: FRED
- **URL**: https://fred.stlouisfed.org/series/VIXCLS
- **Usage**: Volatility state indicators
- **File**: `option_iv_monthly.parquet`

#### Stock Returns and Characteristics
- **Source**: CRSP (Center for Research in Security Prices)
- **Access**: Public academic access (free with institutional affiliation)
- **Usage**: Stock returns, market capitalization, book-to-market ratios
- **File**: `panel_monthly.parquet`

### 🔴 **Restricted Data** (Requires credentials)

#### High-Frequency Trading Data
- **Source**: TAQ (Trade and Quote) via WRDS
- **Access**: Requires WRDS subscription (~$5,000/year)
- **Usage**: Intraday trading patterns (not used in public reproduction)
- **Alternative**: Use CRSP daily data for basic analysis

#### Institutional Holdings
- **Source**: 13F filings via WRDS
- **Access**: Requires WRDS subscription
- **Usage**: Institutional ownership patterns
- **Alternative**: Use CRSP institutional holdings data

#### Retail Trading Data
- **Source**: Robinhood Holdings (proprietary)
- **Access**: Not publicly available
- **Usage**: Retail investor breadth measures
- **Alternative**: Use public retail sentiment proxies

## Minimal Public Reproduction Dataset

For public reproduction, you need these **pre-computed files**:

```
DATA_ROOT/
├── panel_monthly.parquet          # Main analysis panel
├── sentiment_monthly.parquet       # Consumer sentiment data
├── option_iv_monthly.parquet      # VIX and option implied volatility
└── proxies/                       # Retail sentiment proxies
    ├── retailera_breadth.parquet
    └── miller_breadth_flows.parquet
```

## Building from Raw Data

### Step 1: Set Up Data Directory

```bash
# Set DATA_ROOT environment variable
export DATA_ROOT=/path/to/your/data
mkdir -p $DATA_ROOT
```

### Step 2: Download Public Data

#### Consumer Sentiment (FRED)
```python
import pandas_datareader as pdr
import pandas as pd

# Download UMCSENT
sentiment = pdr.get_data_fred('UMCSENT', start='1990-01-01')
sentiment.to_parquet('sentiment_monthly.parquet')
```

#### VIX Data (FRED)
```python
# Download VIX
vix = pdr.get_data_fred('VIXCLS', start='1990-01-01')
vix.to_parquet('option_iv_monthly.parquet')
```

#### CRSP Data (Academic Access)
```python
# Requires CRSP access - contact your institution
# Download monthly stock returns, characteristics
# Process into panel_monthly.parquet format
```

### Step 3: Generate Minimal Script

Create `scripts/generate_minimal_data.py`:

```python
#!/usr/bin/env python3
"""
Generate minimal data files for public reproduction.
Skips TAQ/WRDS fetches, uses only public data.
"""

import pandas as pd
import pandas_datareader as pdr
from pathlib import Path
import os

def download_public_data(data_root):
    """Download publicly available data."""
    data_path = Path(data_root)
    data_path.mkdir(exist_ok=True)
    
    print("Downloading consumer sentiment data...")
    sentiment = pdr.get_data_fred('UMCSENT', start='1990-01-01')
    sentiment.to_parquet(data_path / 'sentiment_monthly.parquet')
    
    print("Downloading VIX data...")
    vix = pdr.get_data_fred('VIXCLS', start='1990-01-01')
    vix.to_parquet(data_path / 'option_iv_monthly.parquet')
    
    print("Creating minimal panel data...")
    # Create minimal panel with public data only
    # This would need to be customized based on available CRSP access
    
    print("✓ Minimal data files created")

if __name__ == "__main__":
    data_root = os.environ.get('DATA_ROOT', '../Data')
    download_public_data(data_root)
```

### Step 4: Run Data Generation

```bash
# Set DATA_ROOT
export DATA_ROOT=/path/to/your/data

# Run minimal data generation
python scripts/generate_minimal_data.py

# Verify files exist
make data DATA_ROOT=$DATA_ROOT
```

## Full Reproduction with Credentials

### WRDS Access Setup

1. **Obtain WRDS subscription** through your institution
2. **Set up WRDS credentials**:
   ```bash
   export WRDS_USERNAME=your_username
   export WRDS_PASSWORD=your_password
   ```

3. **Install WRDS Python package**:
   ```bash
   pip install wrds
   ```

### Full Data Pipeline

```python
# scripts/build_full_data.py
import wrds
import pandas as pd

def build_full_dataset():
    """Build complete dataset with WRDS access."""
    
    # Connect to WRDS
    db = wrds.Connection()
    
    # Download CRSP data
    crsp_data = db.raw_sql("""
        SELECT permno, date, ret, me, be
        FROM crsp.msf
        WHERE date >= '1990-01-01'
    """)
    
    # Download TAQ data (if needed)
    # taq_data = db.raw_sql("SELECT * FROM taq.ctm WHERE...")
    
    # Process and save
    crsp_data.to_parquet('panel_monthly.parquet')
    
    db.close()
```

## Data Validation

### Check Data Completeness

```bash
# Validate data files
python -c "
import pandas as pd
from pathlib import Path

data_root = Path('$DATA_ROOT')
files = [
    'panel_monthly.parquet',
    'sentiment_monthly.parquet', 
    'option_iv_monthly.parquet'
]

for file in files:
    path = data_root / file
    if path.exists():
        df = pd.read_parquet(path)
        print(f'✓ {file}: {df.shape[0]} rows, {df.shape[1]} cols')
    else:
        print(f'✗ {file}: MISSING')
"
```

### Expected Data Ranges

- **Panel data**: ~1.8M observations (1990-2023)
- **Sentiment data**: ~400 monthly observations
- **VIX data**: ~400 monthly observations
- **Proxy data**: Variable coverage depending on source

## Troubleshooting

### Common Issues

1. **"DATA_ROOT not found"**
   ```bash
   export DATA_ROOT=/path/to/your/data
   make data DATA_ROOT=$DATA_ROOT
   ```

2. **"Missing parquet files"**
   - Run `python scripts/generate_minimal_data.py`
   - Or download pre-computed files from repository

3. **"WRDS connection failed"**
   - Check credentials: `export WRDS_USERNAME=...`
   - Verify institutional access
   - Use public data alternatives

4. **"Permission denied"**
   - Ensure write access to DATA_ROOT directory
   - Check file permissions

### Data Quality Checks

```bash
# Check data quality
python -c "
import pandas as pd
df = pd.read_parquet('$DATA_ROOT/panel_monthly.parquet')
print('Data quality check:')
print(f'  Rows: {df.shape[0]:,}')
print(f'  Columns: {df.shape[1]}')
print(f'  Date range: {df.date.min()} to {df.date.max()}')
print(f'  Missing values: {df.isnull().sum().sum()}')
"
```

## Contact and Support

For data access issues:
- **Public data**: Check FRED documentation
- **CRSP access**: Contact your institution's library
- **WRDS access**: Contact WRDS support
- **Code issues**: Open GitHub issue

## License and Citation

When using this data:
- **FRED data**: Public domain
- **CRSP data**: Academic use license
- **WRDS data**: Subscription terms apply
- **Code**: MIT License (see LICENSE file)

Please cite the original data sources in your work.
