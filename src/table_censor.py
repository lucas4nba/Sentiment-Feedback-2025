import json, math
from pathlib import Path
import sys
sys.path.append('.')

def format_half_life(hl_ci, rho_ci):
    lo, hi = hl_ci
    if rho_ci[1] >= 1.0:  # upper crosses 1
        return f'{lo:.1f} [ {lo:.1f}, $\\infty$ ]'
    return f'{(lo+0):.1f} [ {lo:.1f}, {hi:.1f} ]'

est = json.loads(Path('outputs/estimates/kappa_rho_gmm.json').read_text())
s = f"""
\\begin{{tabular}}{{lccc}}
\\toprule
 & $\\hat\\kappa$ [CI] & $\\hat\\rho$ [CI] & Half-life [CI] \\\\
\\midrule
Aggregate & {est['kappa_hat']:.3f} [{est['kappa_ci'][0]:.3f}, {est['kappa_ci'][1]:.3f}] &
{est['rho_hat']:.3f} [{est['rho_ci'][0]:.3f}, {est['rho_ci'][1]:.3f}] &
{format_half_life(est['half_life_ci'], est['rho_ci'])} \\\\
\\bottomrule
\\end{{tabular}}
"""
Path('tables_figures/latex/table_kappa_rho.tex').write_text(s)
