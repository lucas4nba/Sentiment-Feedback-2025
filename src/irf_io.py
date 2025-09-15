from pathlib import Path
import json
import numpy as np

def save_irf(path, horizons, beta_hat, Sigma_hat):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    obj = {'horizons': horizons, 'beta': list(map(float, beta_hat)),
           'Sigma': np.asarray(Sigma_hat).tolist()}
    path.write_text(json.dumps(obj))

def load_irf(path):
    obj = json.loads(Path(path).read_text())
    return obj['horizons'], np.array(obj['beta']), np.array(obj['Sigma'])
