# Sentiment Feedback in Equity Markets - Reproducibility Package

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

**Complete reproducibility package for "Sentiment Feedback in Equity Markets: Asymmetries, Retail Heterogeneity, and Structural Calibration"**

## 🚀 Quick Start

### One-Button Reproduction
```bash
# Run everything from scratch
./reproducibility/run.sh

# Or use Make
make -f reproducibility/Makefile all
```

### Docker Reproduction
```bash
# Build and run in container
cd reproducibility/docker
docker build -t sentiment-feedback .
docker run -v $(pwd)/../project_files:/workspace sentiment-feedback make all
```

## 📋 What This Package Contains

This reproducibility package includes everything needed to reproduce the complete analysis:

- **Complete analysis pipeline** from raw data to final paper
- **Multiple reproduction methods** (Docker, Make, PowerShell)
- **Comprehensive documentation** with step-by-step instructions
- **Data access guide** for public vs. restricted data
- **Environment management** with pinned dependencies
- **Quality validation** and robustness testing

## 🏗️ Project Structure

```
├── reproducibility/           # Complete reproduction infrastructure
│   ├── README.md             # Detailed reproduction guide
│   ├── Dockerfile            # Containerized reproduction
│   ├── run.sh               # One-button reproduction script
│   ├── Makefile             # Build automation
│   ├── requirements.txt     # Pinned Python dependencies
│   ├── env.yml              # Conda environment
│   ├── manifest.yaml        # Dependency mapping
│   ├── check_env.py         # Environment validation
│   └── docs/                # Additional documentation
├── project_files/           # Main analysis code
│   ├── src/                 # Data processing scripts
│   ├── scripts/             # Analysis scripts
│   └── configs/             # Configuration files
├── CITATION.cff            # Citation information
├── LICENSE                  # MIT license
└── README.md               # This file
```

## 🔧 Hardware Requirements

### Minimum Requirements
- **CPU**: 4 cores, 2.0 GHz
- **RAM**: 8 GB
- **Storage**: 10 GB free space
- **OS**: Linux/macOS/Windows with WSL2

### Recommended Requirements
- **CPU**: 8 cores, 3.0 GHz
- **RAM**: 16 GB
- **Storage**: 20 GB free space (SSD recommended)

## ⏱️ Expected Wall Times

### Full Reproduction Pipeline
- **Data preparation**: 15-30 minutes
- **IRF estimation**: 45-90 minutes
- **GMM calibration**: 30-60 minutes
- **Panel analysis**: 20-40 minutes
- **Portfolio analysis**: 10-20 minutes
- **Figure generation**: 5-10 minutes

**Total**: 2-4 hours (depending on hardware)

## 📊 Key Research Findings

### Structural Calibration
- **Amplification parameter (κ)**: 1.06 bps per 1σ sentiment shock
- **Persistence parameter (ρ)**: 0.940
- **Half-life**: 11.2 months
- **Peak response**: 1.20 bps at 12-month horizon

### Portfolio Performance
- **D10-D1 long-short strategy**: 4-13 bps per month
- **Sharpe ratios**: 0.18-0.85 across horizons
- **Transaction cost sensitivity**: Profitable after 10 bps costs

### Cross-sectional Heterogeneity
- **Low-breadth amplification**: 1.72-8.69 bps across horizons
- **VIX regime interactions**: 12-31 bps for triple interactions
- **Retail era effects**: Enhanced short-horizon amplification

## 📚 Data Requirements

### Public Data (No credentials required)
- **Consumer Sentiment**: University of Michigan Consumer Sentiment Survey (FRED)
- **Market Data**: VIX, stock returns (CRSP academic access)
- **Option Data**: Implied volatility measures

### Restricted Data (Requires credentials)
- **High-frequency trading data**: TAQ via WRDS (~$5,000/year)
- **Institutional holdings**: 13F filings via WRDS
- **Retail trading data**: Robinhood Holdings (proprietary)

**Note**: The package includes scripts to generate minimal reproduction datasets using only public data.

## 🛠️ Installation & Setup

### Method 1: Python Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r reproducibility/requirements.txt

# Verify installation
python reproducibility/check_env.py
```

### Method 2: Conda Environment
```bash
# Create conda environment
conda env create -f reproducibility/env.yml
conda activate sentiment-feedback-env

# Verify installation
python reproducibility/check_env.py
```

### Method 3: Docker
```bash
# Build Docker image
cd reproducibility/docker
docker build -t sentiment-feedback .

# Run analysis
docker run -v $(pwd)/../project_files:/workspace sentiment-feedback make all
```

## 🎯 Running the Analysis

### Complete Pipeline
```bash
# One-button reproduction
./reproducibility/run.sh

# Or step-by-step
make -f reproducibility/Makefile env    # Check environment
make -f reproducibility/Makefile data   # Build data files
make -f reproducibility/Makefile irf    # Estimate IRFs
make -f reproducibility/Makefile gmm    # Structural calibration
make -f reproducibility/Makefile panel  # Panel analysis
make -f reproducibility/Makefile ports  # Portfolio analysis
make -f reproducibility/Makefile figures # Generate figures
make -f reproducibility/Makefile tables # Generate tables
```

### Individual Components
```bash
# Data preparation only
make -f reproducibility/Makefile data

# IRF estimation only
make -f reproducibility/Makefile irf

# Portfolio analysis only
make -f reproducibility/Makefile ports
```

## 📈 Outputs Generated

### Data Files
- `project_files/build/panel_monthly.parquet` - Main analysis panel
- `project_files/build/sentiment_monthly.parquet` - Sentiment measures
- `project_files/build/option_iv_monthly.parquet` - Option implied volatility
- `project_files/build/proxies/*.parquet` - Retail sentiment proxies

### Analysis Results
- `project_files/build/irf_estimates.csv` - IRF coefficients
- `project_files/build/kappa_rho_estimates.csv` - Structural parameters
- `project_files/build/panel_results.csv` - Panel regression results
- `project_files/build/portfolio_results.csv` - Portfolio performance

### Figures & Tables
- `project_files/tables_figures/final_figures/*.pdf` - All research figures
- `project_files/tables_figures/latex/*.tex` - LaTeX tables

## 🔍 Validation & Quality Control

### Data Validation
```bash
# Validate data files
python project_files/scripts/validate_data.py

# Check data quality
python project_files/src/qc_comprehensive.py
```

### Output Validation
```bash
# Validate all outputs
make -f reproducibility/Makefile validate

# Check figure quality
python project_files/scripts/lint_figures.py
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory errors during IRF estimation**
   - Reduce `project_files/configs/baseline.yml` → `gmm.draws` from 2000 to 1000
   - Use `--chunk-size 100` flag for large datasets

2. **Missing data files**
   - Verify `DATA_ROOT` environment variable is set
   - Run `make -f reproducibility/Makefile data` to rebuild from scratch

3. **Permission errors (Windows)**
   - Run PowerShell as Administrator
   - Use WSL2 for Linux compatibility

### Debug Mode
```bash
# Verbose output
make -f reproducibility/Makefile all VERBOSE=1

# Dry run (show commands without executing)
make -f reproducibility/Makefile all DRY_RUN=1

# Check specific component
make -f reproducibility/Makefile irf DEBUG=1
```

## 📖 Documentation

- **Detailed reproduction guide**: `reproducibility/README.md`
- **Data access instructions**: `reproducibility/docs/DATA_ACCESS.md`
- **Implementation status**: `project_files/IMPLEMENTATION_STATUS.md`
- **Data codebook**: `project_files/DATA_CODEBOOK.md`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the full reproduction pipeline
5. Submit a pull request

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{sentiment_feedback_2024,
  title={Sentiment Feedback in Equity Markets: Asymmetries, Retail Heterogeneity, and Structural Calibration},
  author={Sneller, Lucas},
  journal={arXiv preprint},
  year={2024},
  doi={10.48550/arXiv.XXXX.XXXXX}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review `reproducibility/README.md` for detailed instructions
3. Open an issue on the repository
4. Contact the author

