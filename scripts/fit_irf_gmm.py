import json, yaml, sys
from pathlib import Path
import numpy as np
sys.path.append('.')
from src.irf_io import load_irf
from src.gmm_fit import fit_gmm
from src.gmm_bootstrap import parametric_boot, half_life

cfg = yaml.safe_load(Path('configs/baseline.yml').read_text())
h, beta, Sigma = load_irf('outputs/irf/aggregate_irf.json')
res = fit_gmm(beta, Sigma, h, cfg['irf']['type'], tuple(cfg['gmm']['rho_bounds']))
boot = parametric_boot(beta, Sigma, h, cfg['irf']['type'],
                       draws=cfg['gmm']['draws'], rho_bounds=tuple(cfg['gmm']['rho_bounds']))
out = {
    'kappa_hat': res['kappa'],
    'rho_hat': res['rho'],
    'J': res['J'],
    'df': res['df'],
    'half_life_hat': float(half_life(res['rho'])),
    'kappa_ci': boot['kappa_ci'],
    'rho_ci': boot['rho_ci'],
    'half_life_ci': boot['half_life_ci'],
}
Path('outputs/estimates').mkdir(parents=True, exist_ok=True)
Path('outputs/estimates/kappa_rho_gmm.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
