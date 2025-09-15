# Reproducibility Guide

This document provides step-by-step instructions for reproducing the analysis in "Sentiment Feedback in Equity Markets: Asymmetries, Retail Heterogeneity, and Structural Calibration".

## Quick Start

### One-Button Reproduction
```bash
# Run everything from scratch
./run.sh

# Or use Make
make all
```

### Docker Reproduction
```bash
# Build and run in container
cd docker
docker build -t sentiment-feedback .
docker run -v $(pwd)/../project_files:/workspace sentiment-feedback make all
```

## Hardware Requirements

### Minimum Requirements
- **CPU**: 4 cores, 2.0 GHz
- **RAM**: 8 GB
- **Storage**: 10 GB free space
- **OS**: Linux/macOS/Windows with WSL2

### Recommended Requirements
- **CPU**: 8 cores, 3.0 GHz
- **RAM**: 16 GB
- **Storage**: 20 GB free space (SSD recommended)

## Software Dependencies

### Required Software
- **Python**: 3.8+ (tested with 3.9, 3.10, 3.11)
- **LaTeX**: TeXLive 2020+ or MiKTeX 21+
- **Git**: For version control
- **Docker**: Optional, for containerized reproduction

### Python Packages
See `requirements.txt` for pinned versions. Key packages:
- pandas >= 1.5.0
- numpy >= 1.20.0
- scipy >= 1.7.0
- statsmodels >= 0.13.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- pyarrow >= 10.0.0

## Expected Wall Times

### Full Reproduction Pipeline
- **Data preparation**: 15-30 minutes
- **IRF estimation**: 45-90 minutes
- **GMM calibration**: 30-60 minutes
- **Panel analysis**: 20-40 minutes
- **Portfolio analysis**: 10-20 minutes
- **Figure generation**: 5-10 minutes
- **LaTeX compilation**: 2-5 minutes

**Total**: 2-4 hours (depending on hardware)

### Individual Components
- `make env`: 2-5 minutes (environment check)
- `make data`: 15-30 minutes (data processing)
- `make irf`: 45-90 minutes (impulse response functions)
- `make gmm`: 30-60 minutes (structural calibration)
- `make panel`: 20-40 minutes (panel regressions)
- `make ports`: 10-20 minutes (portfolio analysis)
- `make figures`: 5-10 minutes (figure generation)
- `make tables`: 2-5 minutes (table generation)

## Step-by-Step Instructions

### 1. Environment Setup

```bash
# Clone repository
git clone [REPO_URL]
cd sentiment-feedback

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r reproducibility/requirements.txt

# Verify installation
make env
```

### 2. Data Preparation

```bash
# Process raw data files
make data

# Verify data quality
python scripts/validate_data.py
```

**What gets built:**
- `project_files/build/panel_monthly.parquet` - Main analysis panel
- `project_files/build/sentiment_monthly.parquet` - Sentiment measures
- `project_files/build/option_iv_monthly.parquet` - Option implied volatility
- `project_files/build/proxies/*.parquet` - Retail sentiment proxies

### 3. Core Analysis

#### Impulse Response Functions (IRFs)
```bash
make irf
```

**Scripts executed:**
- `scripts/fit_irf_gmm.py` - Main IRF estimation
- `scripts/generate_irf_grid.py` - IRF visualization
- `scripts/generate_irf_forest.py` - Forest plots

**Outputs:**
- `project_files/tables_figures/final_figures/irf_grid.pdf`
- `project_files/tables_figures/final_figures/irf_forest.pdf`
- `project_files/tables_figures/latex/T_irf_peaks_half_life.tex`

#### Structural Calibration (κ-ρ GMM)
```bash
make gmm
```

**Scripts executed:**
- `scripts/fit_irf_gmm.py --mode gmm` - GMM estimation
- `scripts/generate_kappa_rho_body_table.py` - Results table

**Outputs:**
- `project_files/tables_figures/latex/T_kappa_rho_body.tex`
- `project_files/build/kappa_rho_estimates.csv`

#### Panel Interactions
```bash
make panel
```

**Scripts executed:**
- `scripts/panel_jackknife.py` - Panel regressions with jackknife SE
- `scripts/generate_proxy_interactions.py` - Interaction effects

**Outputs:**
- `project_files/tables_figures/latex/T_baseline_interactions.tex`
- `project_files/tables_figures/latex/T_proxy_interactions.tex`

#### Portfolio Analysis
```bash
make ports
```

**Scripts executed:**
- `scripts/port_alpha.py` - Portfolio performance analysis
- `scripts/generate_portfolio_metrics_table.py` - Performance metrics

**Outputs:**
- `project_files/tables_figures/latex/T_portfolio_sorts.tex`
- `project_files/tables_figures/final_figures/portfolio_performance.pdf`

### 4. Figure and Table Generation

```bash
# Generate all figures
make figures

# Generate all tables
make tables

# Copy figures to LaTeX directory
make copy-figures
```

### 5. Paper Compilation

```bash
# Compile LaTeX paper
cd Paperfinal91225
latexmk -pdf main.tex
```

## What Gets Rebuilt

### Data Files (make data)
- Raw data processing from `project_files/Data/raw/`
- Monthly aggregation and alignment
- Proxy construction and validation
- Quality control reports

### Analysis Results (make irf, gmm, panel, ports)
- IRF coefficients and confidence intervals
- GMM parameter estimates (κ, ρ)
- Panel regression results with robust standard errors
- Portfolio performance metrics

### Outputs (make figures, tables)
- All figures in `project_files/tables_figures/final_figures/`
- All tables in `project_files/tables_figures/latex/`
- LaTeX compilation in `Paperfinal91225/`

## Troubleshooting

### Common Issues

1. **Memory errors during IRF estimation**
   - Reduce `configs/baseline.yml` → `gmm.draws` from 2000 to 1000
   - Use `--chunk-size 100` flag for large datasets

2. **LaTeX compilation errors**
   - Ensure all required packages installed: `tcolorbox`, `booktabs`, `siunitx`
   - Check for missing figures: `make copy-figures`

3. **Data file not found**
   - Verify `project_files/Data/raw/` contains required CSV files
   - Run `make data` to rebuild from scratch

4. **Permission errors (Windows)**
   - Run PowerShell as Administrator
   - Use WSL2 for Linux compatibility

### Debug Mode

```bash
# Verbose output
make all VERBOSE=1

# Dry run (show commands without executing)
make all DRY_RUN=1

# Check specific component
make irf DEBUG=1
```

### Validation

```bash
# Verify all outputs exist
python scripts/validate_outputs.py

# Check figure quality
python scripts/lint_figures.py

# Verify LaTeX references
python tools/check_tex_refs.py
```

## Reproducibility Notes

### Deterministic Results
- All random seeds are fixed in `configs/baseline.yml`
- Bootstrap procedures use reproducible random number generation
- Results should be identical across runs on same hardware

### Version Control
- All dependencies pinned in `requirements.txt`
- Configuration files versioned in `configs/`
- Output checksums stored in `project_files/build/manifests/`

### Platform Compatibility
- Tested on Ubuntu 20.04+, macOS 11+, Windows 10+ with WSL2
- Docker image provides consistent Linux environment
- PowerShell scripts provided for Windows users

## Citation

If you use this code in your research, please cite:

```bibtex
@article{sentiment_feedback_2024,
  title={Sentiment Feedback in Equity Markets: Asymmetries, Retail Heterogeneity, and Structural Calibration},
  author={[Authors]},
  journal={[Journal]},
  year={2024},
  doi={[DOI]}
}
```

## License

This project is licensed under the MIT License - see `LICENSE` file for details.

## Support

For questions or issues:
1. Check this README and troubleshooting section
2. Review `project_files/IMPLEMENTATION_STATUS.md`
3. Open an issue on the repository
4. Contact the authors at [email]
