import json, matplotlib.pyplot as plt, numpy as np
from pathlib import Path

rows=[]
for px in ['BW','IBES','MarketPsych','PCA_CF']:
    est = json.loads(Path(f'outputs/estimates/{px}_kappa_rho_gmm.json').read_text())
    rows.append((px, est['kappa_hat'], *est['kappa_ci']))
labels = [r[0] for r in rows]
centers = np.array([r[1] for r in rows])
los = np.array([r[2] for r in rows])
his = np.array([r[3] for r in rows])

y = np.arange(len(labels))
plt.figure(figsize=(5,3.2))
plt.errorbar(centers, y, xerr=[centers-los, his-centers], fmt='o', capsize=3)
plt.yticks(y, labels); plt.axvline(0, lw=0.6)
plt.xlabel('Peak IRF (bps per 1 s.d.)')
plt.tight_layout(); plt.savefig('tables_figures/final_figures/irf_forest.pdf')
