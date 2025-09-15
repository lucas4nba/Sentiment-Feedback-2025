import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the rolling parameters data
df = pd.read_csv('outputs/irf/rolling_kappa_rho.csv')

# Convert dates to datetime
df['start'] = pd.to_datetime(df['start'])
df['end'] = pd.to_datetime(df['end'])

# Create the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Plot kappa over time
ax1.plot(df['start'], df['kappa'], 'o-', linewidth=2, markersize=8, color='blue')
ax1.set_ylabel('$\\hat{\\kappa}$ (bps)', fontsize=12)
ax1.set_title('Rolling Window GMM Estimates', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=1.044, color='red', linestyle='--', alpha=0.7, label='Full Sample $\\hat{\\kappa}$')
ax1.legend()

# Plot rho over time
ax2.plot(df['start'], df['rho'], 'o-', linewidth=2, markersize=8, color='green')
ax2.set_ylabel('$\\hat{\\rho}$', fontsize=12)
ax2.set_xlabel('Window Start Date', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0.967, color='red', linestyle='--', alpha=0.7, label='Full Sample $\\hat{\\rho}$')
ax2.legend()

# Format x-axis
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('tables_figures/final_figures/F_rolling_kappa_rho.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Rolling parameters plot saved to tables_figures/final_figures/F_rolling_kappa_rho.pdf")
