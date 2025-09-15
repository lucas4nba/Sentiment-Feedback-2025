# Build all plots and tables
Write-Host "Running IRF GMM estimation..."
python scripts/fit_irf_gmm.py

Write-Host "Running panel jackknife analysis..."
python scripts/panel_jackknife.py

Write-Host "Running portfolio alpha analysis..."
python scripts/port_alpha.py

Write-Host "Running option listing event study..."
python scripts/option_listing_event.py

Write-Host "Generating IRF grid plot..."
python scripts/plot_irf_grid.py

Write-Host "Generating IRF forest plot..."
python scripts/plot_irf_forest.py

Write-Host "Generating LaTeX tables..."
python -c "from src.table_censor import *"

Write-Host "All analysis complete!"
