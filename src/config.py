from pathlib import Path
from typing import Dict, Any

# Root = the folder I'm running from (adjust if you keep a different layout)
ROOT = Path(".")
# Check if we're running from src directory
if ROOT.resolve().name == "src":
    DATA = ROOT.parent  # Go up one level to Data directory
else:
    DATA = ROOT  # We're already in the Data directory

# Subfolders based on my screenshots
DIR_SENTIMENT = DATA / "Sentiment"
DIR_CONTROLS = DATA / "controls"
DIR_RPV = DATA / "ReturnsPricesVol"
DIR_HHF = DATA / "Household flows"
DIR_OPTION_IV = DATA / "Option Implied Vol"

# Output directory
OUT = DATA / "build"
OUT.mkdir(parents=True, exist_ok=True)

# Data filtering configuration
DATA_FILTERS = {
    "price_floor": 5.0,
    "min_months": 12,
    "exclude_sectors": ["REIT", "UTIL", "FIN"],  # keep empty if not excluding
    "winsor": {"returns": 0.01, "chars": 0.01}   # 1% tails
}

def get_filters() -> Dict[str, Any]:
    """Return the data filters configuration."""
    return DATA_FILTERS.copy()
