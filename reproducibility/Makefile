# Makefile for Sentiment Feedback Analysis Reproduction
# Usage: make [target]

# Configuration
PROJECT_DIR := $(shell pwd)
PROJECT_FILES_DIR := $(PROJECT_DIR)/project_files
SCRIPTS_DIR := $(PROJECT_DIR)/scripts
CONFIGS_DIR := $(PROJECT_FILES_DIR)/configs
BUILD_DIR := $(PROJECT_FILES_DIR)/build
TABLES_FIGURES_DIR := $(PROJECT_FILES_DIR)/tables_figures
PAPER_DIR := $(PROJECT_DIR)/Paperfinal91225

# Data configuration
DATA_ROOT ?= ../Data
ARTIFACTS_DIR := reproducibility/artifacts
DIST_DIR := dist

# Python executable
PYTHON := python3
PIP := pip3

# Parallel jobs
JOBS := 4

# Verbose output
ifeq ($(VERBOSE),1)
    Q :=
else
    Q := @
endif

# Dry run mode
ifeq ($(DRY_RUN),1)
    EXEC := echo "[DRY RUN]"
else
    EXEC :=
endif

# Default target
.DEFAULT_GOAL := all

# Phony targets
.PHONY: all env data irf gmm panel ports figures tables copy-figures clean help validate package

# Help target
help:
	@echo "Sentiment Feedback Analysis - Available targets:"
	@echo ""
	@echo "Main targets:"
	@echo "  all          - Complete reproduction pipeline (default)"
	@echo "  env          - Check environment and dependencies"
	@echo "  data         - Build all data files"
	@echo "  irf          - Estimate impulse response functions"
	@echo "  gmm          - Structural calibration (κ-ρ GMM)"
	@echo "  panel        - Panel regression analysis"
	@echo "  ports        - Portfolio analysis"
	@echo "  figures      - Generate all figures"
	@echo "  tables       - Generate all tables"
	@echo "  copy-figures - Copy figures to LaTeX directory"
	@echo "  paper        - Compile LaTeX paper"
	@echo ""
	@echo "Utility targets:"
	@echo "  clean        - Remove build artifacts"
	@echo "  validate     - Validate outputs"
	@echo "  package      - Create reproducibility package"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Options:"
	@echo "  VERBOSE=1    - Enable verbose output"
	@echo "  DRY_RUN=1    - Show commands without executing"
	@echo "  JOBS=N       - Number of parallel jobs (default: 4)"
	@echo ""
	@echo "Examples:"
	@echo "  make all"
	@echo "  make data VERBOSE=1"
	@echo "  make irf DRY_RUN=1"
	@echo "  make all JOBS=8"

# Environment check
env:
	@echo "Checking environment..."
	$(Q)$(EXEC) $(PYTHON) reproducibility/check_env.py
	@echo "✓ Environment check completed"

# Data preparation targets
data: check-data-files $(BUILD_DIR)/panel_monthly.parquet $(BUILD_DIR)/sentiment_monthly.parquet $(BUILD_DIR)/option_iv_monthly.parquet $(BUILD_DIR)/proxies

# Check for minimal required data files
check-data-files:
	@echo "Checking data files in DATA_ROOT=$(DATA_ROOT)..."
	@if [ ! -d "$(DATA_ROOT)" ]; then \
		echo "❌ ERROR: DATA_ROOT directory '$(DATA_ROOT)' not found"; \
		echo ""; \
		echo "Please set DATA_ROOT environment variable or ensure data directory exists:"; \
		echo "  export DATA_ROOT=/path/to/your/data"; \
		echo "  make data DATA_ROOT=/path/to/your/data"; \
		echo ""; \
		echo "For public reproduction, you need these minimal files:"; \
		echo "  $(DATA_ROOT)/panel_monthly.parquet"; \
		echo "  $(DATA_ROOT)/sentiment_monthly.parquet"; \
		echo "  $(DATA_ROOT)/option_iv_monthly.parquet"; \
		echo "  $(DATA_ROOT)/proxies/"; \
		echo ""; \
		echo "See reproducibility/docs/DATA_ACCESS.md for detailed instructions."; \
		exit 1; \
	fi
	@echo "✓ DATA_ROOT directory found"
	@if [ ! -f "$(DATA_ROOT)/panel_monthly.parquet" ]; then \
		echo "⚠️  WARNING: panel_monthly.parquet not found in $(DATA_ROOT)"; \
		echo "  This file will be built from raw data if available."; \
	fi
	@if [ ! -f "$(DATA_ROOT)/sentiment_monthly.parquet" ]; then \
		echo "⚠️  WARNING: sentiment_monthly.parquet not found in $(DATA_ROOT)"; \
		echo "  This file will be built from raw data if available."; \
	fi
	@if [ ! -f "$(DATA_ROOT)/option_iv_monthly.parquet" ]; then \
		echo "⚠️  WARNING: option_iv_monthly.parquet not found in $(DATA_ROOT)"; \
		echo "  This file will be built from raw data if available."; \
	fi
	@if [ ! -d "$(DATA_ROOT)/proxies" ]; then \
		echo "⚠️  WARNING: proxies directory not found in $(DATA_ROOT)"; \
		echo "  Proxy files will be built from raw data if available."; \
	fi

$(BUILD_DIR)/panel_monthly.parquet:
	@echo "Building main panel..."
	$(Q)$(EXEC) cd $(PROJECT_FILES_DIR) && $(PYTHON) -m src.build_panel
	@echo "✓ Panel data built"

$(BUILD_DIR)/sentiment_monthly.parquet:
	@echo "Building sentiment data..."
	$(Q)$(EXEC) cd $(PROJECT_FILES_DIR) && $(PYTHON) -m src.build_sentiment
	@echo "✓ Sentiment data built"

$(BUILD_DIR)/option_iv_monthly.parquet:
	@echo "Building option IV data..."
	$(Q)$(EXEC) cd $(PROJECT_FILES_DIR) && $(PYTHON) -m src.build_option_iv
	@echo "✓ Option IV data built"

$(BUILD_DIR)/proxies:
	@echo "Building retail proxies..."
	$(Q)$(EXEC) cd $(PROJECT_FILES_DIR) && $(PYTHON) scripts/generate_retailera_breadth.py
	$(Q)$(EXEC) cd $(PROJECT_FILES_DIR) && $(PYTHON) scripts/generate_miller_breadth_flows.py
	@echo "✓ Retail proxies built"

# IRF estimation
irf: $(BUILD_DIR)/irf_estimates.csv

$(BUILD_DIR)/irf_estimates.csv: data
	@echo "Estimating impulse response functions..."
	$(Q)$(EXEC) cd $(PROJECT_FILES_DIR) && $(PYTHON) scripts/fit_irf_gmm.py --config $(CONFIGS_DIR)/baseline.yml
	@echo "✓ IRF estimation completed"

# GMM calibration
gmm: $(BUILD_DIR)/kappa_rho_estimates.csv

$(BUILD_DIR)/kappa_rho_estimates.csv: irf
	@echo "Running structural calibration (κ-ρ GMM)..."
	$(Q)$(EXEC) cd $(PROJECT_FILES_DIR) && $(PYTHON) scripts/fit_irf_gmm.py --mode gmm --config $(CONFIGS_DIR)/baseline.yml
	@echo "✓ GMM calibration completed"

# Panel analysis
panel: $(BUILD_DIR)/panel_results.csv

$(BUILD_DIR)/panel_results.csv: data
	@echo "Running panel regression analysis..."
	$(Q)$(EXEC) cd $(PROJECT_FILES_DIR) && $(PYTHON) scripts/panel_jackknife.py --config $(CONFIGS_DIR)/baseline.yml
	@echo "✓ Panel analysis completed"

# Portfolio analysis
ports: $(BUILD_DIR)/portfolio_results.csv

$(BUILD_DIR)/portfolio_results.csv: panel
	@echo "Running portfolio analysis..."
	$(Q)$(EXEC) cd $(PROJECT_FILES_DIR) && $(PYTHON) scripts/port_alpha.py --config $(CONFIGS_DIR)/baseline.yml
	@echo "✓ Portfolio analysis completed"

# Figure generation (parallel)
figures: $(TABLES_FIGURES_DIR)/final_figures/irf_grid.pdf $(TABLES_FIGURES_DIR)/final_figures/irf_forest.pdf $(TABLES_FIGURES_DIR)/final_figures/portfolio_performance.pdf $(TABLES_FIGURES_DIR)/final_figures/retail_coverage.pdf

$(TABLES_FIGURES_DIR)/final_figures/irf_grid.pdf: irf
	@echo "Generating IRF grid figure..."
	$(Q)$(EXEC) cd $(PROJECT_FILES_DIR) && $(PYTHON) scripts/generate_irf_grid.py
	@echo "✓ IRF grid figure generated"

$(TABLES_FIGURES_DIR)/final_figures/irf_forest.pdf: irf
	@echo "Generating IRF forest plot..."
	$(Q)$(EXEC) cd $(PROJECT_FILES_DIR) && $(PYTHON) scripts/generate_irf_forest.py
	@echo "✓ IRF forest plot generated"

$(TABLES_FIGURES_DIR)/final_figures/portfolio_performance.pdf: ports
	@echo "Generating portfolio performance figure..."
	$(Q)$(EXEC) cd $(PROJECT_FILES_DIR) && $(PYTHON) scripts/generate_portfolio_core.py
	@echo "✓ Portfolio performance figure generated"

$(TABLES_FIGURES_DIR)/final_figures/retail_coverage.pdf: data
	@echo "Generating retail coverage figure..."
	$(Q)$(EXEC) cd $(PROJECT_FILES_DIR) && $(PYTHON) scripts/generate_retail_coverage.py
	@echo "✓ Retail coverage figure generated"

# Table generation
tables: $(TABLES_FIGURES_DIR)/latex/T_baseline_interactions.tex $(TABLES_FIGURES_DIR)/latex/T_kappa_rho_body.tex $(TABLES_FIGURES_DIR)/latex/T_portfolio_sorts.tex $(TABLES_FIGURES_DIR)/latex/T_irf_peaks_half_life.tex

$(TABLES_FIGURES_DIR)/latex/T_baseline_interactions.tex: panel
	@echo "Generating baseline interactions table..."
	$(Q)$(EXEC) cd $(PROJECT_FILES_DIR) && $(PYTHON) scripts/generate_proxy_interactions.py
	@echo "✓ Baseline interactions table generated"

$(TABLES_FIGURES_DIR)/latex/T_kappa_rho_body.tex: gmm
	@echo "Generating κ-ρ body table..."
	$(Q)$(EXEC) cd $(PROJECT_FILES_DIR) && $(PYTHON) scripts/generate_kappa_rho_body_table.py
	@echo "✓ κ-ρ body table generated"

$(TABLES_FIGURES_DIR)/latex/T_portfolio_sorts.tex: ports
	@echo "Generating portfolio sorts table..."
	$(Q)$(EXEC) cd $(PROJECT_FILES_DIR) && $(PYTHON) scripts/generate_portfolio_metrics_table.py
	@echo "✓ Portfolio sorts table generated"

$(TABLES_FIGURES_DIR)/latex/T_irf_peaks_half_life.tex: irf
	@echo "Generating IRF peaks and half-life table..."
	$(Q)$(EXEC) cd $(PROJECT_FILES_DIR) && $(PYTHON) scripts/generate_irf_peaks_half_life_table.py
	@echo "✓ IRF peaks table generated"

# Copy figures to LaTeX directory
copy-figures: figures
	@echo "Copying figures to LaTeX directory..."
	$(Q)$(EXEC) cd $(PROJECT_FILES_DIR) && $(PYTHON) scripts/copy_figures.py
	@echo "✓ Figures copied to LaTeX directory"

# LaTeX paper compilation
paper: copy-figures tables
	@echo "Compiling LaTeX paper..."
	$(Q)$(EXEC) cd $(PAPER_DIR) && latexmk -pdf main.tex
	@echo "✓ Paper compiled"

# Validation
validate:
	@echo "Validating outputs..."
	$(Q)$(EXEC) cd $(PROJECT_FILES_DIR) && $(PYTHON) scripts/validate_outputs.py
	@echo "✓ Validation completed"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	$(Q)$(EXEC) rm -rf $(BUILD_DIR)/*
	$(Q)$(EXEC) rm -rf $(TABLES_FIGURES_DIR)/final_figures/*
	$(Q)$(EXEC) rm -rf $(TABLES_FIGURES_DIR)/latex/*
	$(Q)$(EXEC) rm -rf $(PROJECT_FILES_DIR)/logs/*
	@echo "✓ Build artifacts cleaned"

# Complete reproduction pipeline
all: env data irf gmm panel ports figures tables copy-figures
	@echo ""
	@echo "✓ Complete reproduction pipeline finished!"
	@echo ""
	@echo "Outputs:"
	@echo "  - Data files: $(BUILD_DIR)/"
	@echo "  - Figures: $(TABLES_FIGURES_DIR)/final_figures/"
	@echo "  - Tables: $(TABLES_FIGURES_DIR)/latex/"
	@echo ""
	@echo "To compile the paper, run: make paper"

# Parallel execution helpers
parallel-irf:
	@echo "Running IRF estimation in parallel..."
	$(Q)$(EXEC) cd $(PROJECT_FILES_DIR) && $(PYTHON) scripts/fit_irf_gmm.py --config $(CONFIGS_DIR)/baseline.yml --parallel $(JOBS)

parallel-figures:
	@echo "Generating figures in parallel..."
	$(Q)$(EXEC) cd $(PROJECT_FILES_DIR) && $(PYTHON) scripts/generate_irf_grid.py &
	$(Q)$(EXEC) cd $(PROJECT_FILES_DIR) && $(PYTHON) scripts/generate_irf_forest.py &
	$(Q)$(EXEC) cd $(PROJECT_FILES_DIR) && $(PYTHON) scripts/generate_portfolio_core.py &
	$(Q)$(EXEC) cd $(PROJECT_FILES_DIR) && $(PYTHON) scripts/generate_retail_coverage.py &
	$(Q)$(EXEC) wait
	@echo "✓ All figures generated"

# Debug targets
debug-data:
	@echo "Debug: Data files status"
	@echo "Panel data: $(shell test -f $(BUILD_DIR)/panel_monthly.parquet && echo "✓ EXISTS" || echo "✗ MISSING")"
	@echo "Sentiment data: $(shell test -f $(BUILD_DIR)/sentiment_monthly.parquet && echo "✓ EXISTS" || echo "✗ MISSING")"
	@echo "Option IV data: $(shell test -f $(BUILD_DIR)/option_iv_monthly.parquet && echo "✓ EXISTS" || echo "✗ MISSING")"
	@echo "Proxies directory: $(shell test -d $(BUILD_DIR)/proxies && echo "✓ EXISTS" || echo "✗ MISSING")"

debug-outputs:
	@echo "Debug: Output files status"
	@echo "IRF estimates: $(shell test -f $(BUILD_DIR)/irf_estimates.csv && echo "✓ EXISTS" || echo "✗ MISSING")"
	@echo "GMM estimates: $(shell test -f $(BUILD_DIR)/kappa_rho_estimates.csv && echo "✓ EXISTS" || echo "✗ MISSING")"
	@echo "Panel results: $(shell test -f $(BUILD_DIR)/panel_results.csv && echo "✓ EXISTS" || echo "✗ MISSING")"
	@echo "Portfolio results: $(shell test -f $(BUILD_DIR)/portfolio_results.csv && echo "✓ EXISTS" || echo "✗ MISSING")"

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	$(Q)$(EXEC) $(PIP) install -r reproducibility/requirements.txt
	@echo "✓ Dependencies installed"

# Package target for reproducibility distribution
package: all
	@echo "Creating reproducibility package..."
	@mkdir -p $(ARTIFACTS_DIR) $(DIST_DIR)
	@COMMIT_HASH=$$(git rev-parse --short HEAD 2>/dev/null || echo "unknown"); \
	ARTIFACT_DIR="$(ARTIFACTS_DIR)/$$COMMIT_HASH"; \
	mkdir -p "$$ARTIFACT_DIR"; \
	echo "Copying outputs to $$ARTIFACT_DIR..."; \
	cp -r $(TABLES_FIGURES_DIR)/final_figures "$$ARTIFACT_DIR/"; \
	cp -r $(TABLES_FIGURES_DIR)/latex "$$ARTIFACT_DIR/"; \
	cp reproducibility/README.md "$$ARTIFACT_DIR/"; \
	cp reproducibility/manifest.yaml "$$ARTIFACT_DIR/"; \
	if [ -f reproducibility/RUN_LOG.json ]; then cp reproducibility/RUN_LOG.json "$$ARTIFACT_DIR/"; fi; \
	echo "Validating outputs with make_zip.py..."; \
	$(PYTHON) reproducibility/make_zip.py "$$ARTIFACT_DIR"; \
	echo "Creating distribution package..."; \
	cd $(ARTIFACTS_DIR) && zip -r "../../$(DIST_DIR)/sneller-sentiment-feedback-reproducibility-$$COMMIT_HASH.zip" "$$COMMIT_HASH"; \
	echo "✓ Package created: $(DIST_DIR)/sneller-sentiment-feedback-reproducibility-$$COMMIT_HASH.zip"
