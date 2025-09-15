import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define the counterfactual scenarios based on empirical findings
scenarios = {
    'Baseline': {'kappa': 1.044, 'rho': 0.967, 'half_life': 20.6},
    'Low Breadth': {'kappa': 1.5, 'rho': 0.98, 'half_life': 34.3},  # Higher amplification, more persistent
    'High Breadth': {'kappa': 0.7, 'rho': 0.95, 'half_life': 13.5}   # Lower amplification, less persistent
}

# Create the data for plotting
scenario_names = list(scenarios.keys())
half_lives = [scenarios[name]['half_life'] for name in scenario_names]
peak_irfs = [scenarios[name]['kappa'] for name in scenario_names]  # Peak IRF ≈ κ

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Colors for the scenarios
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

# Plot 1: Half-life comparison
bars1 = ax1.bar(scenario_names, half_lives, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
ax1.set_ylabel('Half-life (months)', fontsize=12, fontweight='bold')
ax1.set_title('Counterfactual: Half-life by Breadth Regime', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars1, half_lives)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{value:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Plot 2: Peak IRF comparison
bars2 = ax2.bar(scenario_names, peak_irfs, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
ax2.set_ylabel('Peak IRF (bps per 1 s.d.)', fontsize=12, fontweight='bold')
ax2.set_title('Counterfactual: Peak IRF by Breadth Regime', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars2, peak_irfs)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Add κ and ρ annotations
for i, name in enumerate(scenario_names):
    kappa = scenarios[name]['kappa']
    rho = scenarios[name]['rho']
    ax1.text(i, -3, f'κ={kappa:.2f}, ρ={rho:.3f}', ha='center', va='top', 
             fontsize=10, style='italic', color=colors[i])

# Adjust layout and save
plt.tight_layout()
plt.savefig('tables_figures/final_figures/F_counterfactual_breadth.pdf', 
            dpi=300, bbox_inches='tight')
plt.close()

# Also create a line chart version showing the IRF paths
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

horizons = np.array([1, 3, 6, 12])
for i, (name, params) in enumerate(scenarios.items()):
    kappa = params['kappa']
    rho = params['rho']
    irf_path = kappa * (rho ** (horizons - 1))  # Level IRF model
    ax.plot(horizons, irf_path, 'o-', linewidth=2, markersize=6, 
            color=colors[i], label=f'{name} (κ={kappa:.2f}, ρ={rho:.3f})')

ax.set_xlabel('Horizon (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('IRF (bps per 1 s.d.)', fontsize=12, fontweight='bold')
ax.set_title('Counterfactual: IRF Paths by Breadth Regime', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('tables_figures/final_figures/F_counterfactual_irf_paths.pdf', 
            dpi=300, bbox_inches='tight')
plt.close()

print("Counterfactual charts saved:")
print("- F_counterfactual_breadth.pdf (bar charts)")
print("- F_counterfactual_irf_paths.pdf (line chart)")
