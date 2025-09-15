#!/bin/bash

# Build all plots and tables
echo "Running IRF GMM estimation..."
python scripts/fit_irf_gmm.py

echo "Running panel jackknife analysis..."
python scripts/panel_jackknife.py

echo "Running portfolio alpha analysis..."
python scripts/port_alpha.py

echo "Running option listing event study..."
python scripts/option_listing_event.py

echo "Generating IRF grid plot..."
python scripts/plot_irf_grid.py

echo "Generating IRF forest plot..."
python scripts/plot_irf_forest.py

echo "Generating LaTeX tables..."
python - <<'PY'
from src.table_censor import *  # writes table_kappa_rho.tex
PY

echo "All analysis complete!"
