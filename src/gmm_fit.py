import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize

def irf_model(theta, horizons, irf_type='level'):
    kappa, rho = theta
    h = np.array(horizons, dtype=float)
    if irf_type == 'level':
        return kappa * np.power(rho, h - 1)
    elif irf_type == 'cum':
        eps = 1e-10
        denom = (1 - rho) if abs(1 - rho) > eps else np.sign(1 - rho) * eps
        return kappa * (1 - np.power(rho, h)) / denom
    else:
        raise ValueError("irf_type must be 'level' or 'cum'")

def jstat(theta, beta_hat, Sigma_hat, horizons, irf_type='level'):
    g = irf_model(theta, horizons, irf_type)
    m = beta_hat - g
    W = inv(Sigma_hat)
    return float(m.T @ W @ m)

def fit_gmm(beta_hat, Sigma_hat, horizons, irf_type='level',
            rho_bounds=(-0.999, 0.9999)):
    # crude starting values
    k0 = beta_hat[0]
    # estimate rho via log ratios if possible
    if len(beta_hat) >= 2 and abs(beta_hat[1]) > 1e-8 and abs(beta_hat[0]) > 1e-8:
        r0 = np.sign(beta_hat[1]/beta_hat[0]) * min(abs(beta_hat[1]/beta_hat[0])**(1/(horizons[1]-horizons[0])), 0.95)
    else:
        r0 = 0.5
    x0 = np.array([k0, r0])

    bounds = [(None, None), rho_bounds]
    obj = lambda th: jstat(th, beta_hat, Sigma_hat, horizons, irf_type)
    res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds)
    kappa, rho = res.x
    J = obj(res.x)
    df = len(horizons) - 2  # moments - params
    return {'kappa': kappa, 'rho': rho, 'J': J, 'df': df, 'success': res.success, 'message': res.message}
