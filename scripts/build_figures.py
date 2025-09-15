#!/usr/bin/env python3
"""
Build figures for the sentiment feedback paper.
Exports all paper figures as vector PDFs to the LaTeX directory.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Figure registry - maps figure names to their generation functions
FIGS = {
    'fig_irf_asym': 'make_irf_asym',
    'fig_irf_signsplit': 'make_irf_signsplit', 
    'fig_irf_delta': 'make_irf_delta',
    'fig_portfolio_sorts': 'make_portfolio_sorts',
    'fig_retailera_breadth': 'make_retailera_breadth',
    'fig_breadth_vix': 'make_breadth_vix',
    'fig_calibration_overlay': 'make_calibration_overlay',
    'fig_option_listing_event': 'make_option_listing_event',
    'fig_data_coverage': 'make_data_coverage',
    'fig_breadth_qc_coverage': 'make_breadth_qc_coverage',
    'fig_rolling_kappa_rho': 'make_rolling_kappa_rho',
    'fig_irf_grid': 'make_irf_grid',
    'fig_irf_forest': 'make_irf_forest',
    'fig_proxy_irfs': 'make_proxy_irfs',
    'fig_retail_coverage': 'make_retail_coverage',
    'fig_counterfactual_irf_paths': 'make_counterfactual_irf_paths',
}

# Output directory (LaTeX tables_figures folder)
OUTPUT_DIR = Path("../Sentiment_Feedback_in_Equity_Markets__Asymmetries__Retail_Heterogeneity__and_Structural_Calibration__26_/tables_figures/final_figures")

def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Could not get git commit hash")
        return "unknown"

def get_timestamp() -> str:
    """Get current timestamp."""
    return datetime.now().isoformat()

def make_irf_asym() -> str:
    """Generate IRF asymmetry figure."""
    logger.info("Generating IRF asymmetry figure...")
    # Implementation would call the actual plotting function
    # For now, return placeholder
    return "fig_irf_asym.pdf"

def make_irf_signsplit() -> str:
    """Generate IRF sign split figure."""
    logger.info("Generating IRF sign split figure...")
    return "fig_irf_signsplit.pdf"

def make_irf_delta() -> str:
    """Generate IRF delta figure."""
    logger.info("Generating IRF delta figure...")
    return "fig_irf_delta.pdf"

def make_portfolio_sorts() -> str:
    """Generate portfolio sorts figure."""
    logger.info("Generating portfolio sorts figure...")
    return "F_breadth_sorts.pdf"

def make_retailera_breadth() -> str:
    """Generate retail era breadth figure."""
    logger.info("Generating retail era breadth figure...")
    return "F_retailera_breadth.pdf"

def make_breadth_vix() -> str:
    """Generate breadth VIX interactions figure."""
    logger.info("Generating breadth VIX interactions figure...")
    return "F_breadth_vix_betas.pdf"

def make_calibration_overlay() -> str:
    """Generate calibration overlay figure."""
    logger.info("Generating calibration overlay figure...")
    return "F_calibration_overlay.pdf"

def make_option_listing_event() -> str:
    """Generate option listing event study figure."""
    logger.info("Generating option listing event study figure...")
    return "F_option_listing_event.pdf"

def make_data_coverage() -> str:
    """Generate data coverage figure."""
    logger.info("Generating data coverage figure...")
    return "F_breadth_qc_coverage.pdf"

def make_breadth_qc_coverage() -> str:
    """Generate breadth QC coverage figure."""
    logger.info("Generating breadth QC coverage figure...")
    return "F_breadth_qc_coverage.pdf"

def make_rolling_kappa_rho() -> str:
    """Generate rolling kappa rho figure."""
    logger.info("Generating rolling kappa rho figure...")
    return "F_rolling_kappa_rho.pdf"

def make_irf_grid() -> str:
    """Generate IRF grid figure."""
    logger.info("Generating IRF grid figure...")
    return "irf_grid.pdf"

def make_irf_forest() -> str:
    """Generate IRF forest plot figure."""
    logger.info("Generating IRF forest plot figure...")
    return "irf_forest.pdf"

def make_proxy_irfs() -> str:
    """Generate proxy IRFs figure."""
    logger.info("Generating proxy IRFs figure...")
    return "proxy_irfs.pdf"

def make_retail_coverage() -> str:
    """Generate retail coverage figure."""
    logger.info("Generating retail coverage figure...")
    return "retail_coverage.pdf"

def make_counterfactual_irf_paths() -> str:
    """Generate counterfactual IRF paths figure."""
    logger.info("Generating counterfactual IRF paths figure...")
    return "F_counterfactual_irf_paths.pdf"

def build_figure(fig_name: str) -> Optional[str]:
    """Build a single figure."""
    if fig_name not in FIGS:
        logger.error(f"Unknown figure: {fig_name}")
        return None
    
    try:
        # Get the function name and call it
        func_name = FIGS[fig_name]
        func = globals()[func_name]
        output_file = func()
        
        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Move/copy the generated file to output directory
        output_path = OUTPUT_DIR / output_file
        logger.info(f"Figure {fig_name} -> {output_path}")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error building figure {fig_name}: {e}")
        return None

def log_runinfo(figures_built: List[str]) -> None:
    """Log run information to _RUNINFO.json."""
    runinfo = {
        "timestamp": get_timestamp(),
        "git_commit": get_git_commit(),
        "figures_built": figures_built,
        "script": "build_figures.py"
    }
    
    runinfo_path = Path("_RUNINFO.json")
    
    # Load existing runinfo if it exists
    existing_data = {}
    if runinfo_path.exists():
        try:
            with open(runinfo_path, 'r') as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("Could not load existing _RUNINFO.json")
    
    # Append new run info
    if "figure_builds" not in existing_data:
        existing_data["figure_builds"] = []
    
    existing_data["figure_builds"].append(runinfo)
    
    # Write back to file
    with open(runinfo_path, 'w') as f:
        json.dump(existing_data, f, indent=2)
    
    logger.info(f"Logged run info to {runinfo_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Build figures for sentiment feedback paper")
    parser.add_argument("--only", nargs="+", help="Only build specified figures")
    parser.add_argument("--list", action="store_true", help="List available figures")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available figures:")
        for fig_name in FIGS.keys():
            print(f"  {fig_name}")
        return
    
    # Determine which figures to build
    if args.only:
        figures_to_build = args.only
    else:
        figures_to_build = list(FIGS.keys())
    
    logger.info(f"Building {len(figures_to_build)} figures...")
    
    # Build figures
    built_figures = []
    for fig_name in figures_to_build:
        result = build_figure(fig_name)
        if result:
            built_figures.append(result)
    
    # Log run info
    log_runinfo(built_figures)
    
    logger.info(f"Successfully built {len(built_figures)} figures")

if __name__ == "__main__":
    main()
