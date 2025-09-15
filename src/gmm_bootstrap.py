import numpy as np
from numpy.random import multivariate_normal
import sys
sys.path.append('.')
from .gmm_fit import fit_gmm

def half_life(rho):
    if rho <= 0 or rho >= 1:
        return np.inf
    return np.log(0.5) / np.log(rho)

def parametric_boot(beta_hat, Sigma_hat, horizons, irf_type, draws=2000, rho_bounds=(-0.999,0.9999), seed=42):
    rng = np.random.default_rng(seed)
    fits = []
    for _ in range(draws):
        beta_b = rng.multivariate_normal(beta_hat, Sigma_hat)
        res = fit_gmm(beta_b, Sigma_hat, horizons, irf_type, rho_bounds)
        fits.append((res['kappa'], res['rho']))
    K = np.array([f[0] for f in fits])
    R = np.array([f[1] for f in fits])
    HL = np.array([half_life(r) for r in R])

    def ci(arr, a=0.025, b=0.975):
        lo, hi = np.quantile(arr, [a, b])
        return float(lo), float(hi)

    return {'kappa_ci': ci(K), 'rho_ci': ci(R), 'half_life_ci': ci(HL)}
