"""
Spearman Correlation Heatmaps for Phosphorus Fertilizer Study

Purpose: Compute and visualize year-by-year Spearman correlations between yield 
components, soil properties, root and leaf nutrient levels.

Pre-requisite: In 'merging_phosfert_021925' find lien: 'merged_df = pd.merge(df_yield, df_leaf, on=common_cols, how='inner')',
and change df_leaf to df_root or df_soil as needed to plot correlations for root or soil nutrients.

Input: merged_dataset_022725.csv - Available upon request.

Methods: Spearman correlation across numeric variables (yield, soil, leaf nutrients) 
for each year; upper-triangle masked heatmaps displayed side by side with shared colorbar.

Outputs: Combined heatmap figure (leaf_combined_spearman_corr_heatmaps.png), 
per-year correlation matrices (leaf_spearman_corr_{year}.csv), and console output 
of high correlations (|r| > 0.5).

Dependencies: pandas, numpy, matplotlib, seaborn.

Results published in https://doi.org/10.1002/saj2.70145
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set(style="white")

# Load dataset
df = pd.read_csv('merged_dataset_022725.csv')

# Column renaming map
name_mapping = {
    "Variety": "Variety", "P_appln": "Phosphorus Fertilizer", "Replicate": "Replicate", "Plot": "Plot", "Year": "Year",
    "No.1_wt": "US No.1", "Can_wt": "Canner", "Jumbo_wt": "Jumbo", "Mkt_wt": "Total marketable",
    "pH": "pH", "C%": "Carbon", "OM%": "Organic matter", "CEC_meq": "Cation exchange", "Partial_charge_H": "Hydrogen ions",
    "N%": "Nitrogen", "P%": "Phosphorus", "K%": "Potassium", "Ca%": "Calcium", "Mg%": "Magnesium", "S%": "Sulfur",
    "P_ppa": "Phosphorus", "K_ppa": "Potassium", "Ca_ppa": "Calcium", "Mg_ppa": "Magnesium",
    "Na%": "Sodium", "Na_ppa": "Sodium", "B_ppm": "Boron", "Zn_ppm": "Zinc", "Zn_ppa": "Zinc",
    "Mn_ppm": "Manganese", "Fe_ppm": "Iron", "Cu_ppm": "Copper", "Co_ppm": "Cobalt", "Al_ppm": "Aluminium"
}

# Convert identifiers to string
id_cols = ["Variety", "P_appln", "Replicate", "Plot", "Year"]
df[id_cols] = df[id_cols].astype(str)

# Filter and rename columns
valid_cols = [col for col in name_mapping if col in df.columns]
df = df[valid_cols].rename(columns={col: name_mapping[col] for col in valid_cols})

# Unique years for plotting
years = sorted(df['Year'].unique())
n_years = len(years)

# Setup subplots
fig, axes = plt.subplots(1, n_years, figsize=(n_years * 6, 6), squeeze=False)
fig.patch.set_facecolor('white')
vmin, vmax = -1, 1

# Plot heatmaps
for idx, year in enumerate(years):
    ax = axes[0][idx]
    df_year = df[df['Year'] == year]
    corr = df_year.select_dtypes(include='number').corr(method='spearman')
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr, mask=mask, cmap="coolwarm", vmin=vmin, vmax=vmax,
        ax=ax, cbar=False, square=True, linewidths=0.5, linecolor='white'
    )

    ax.set_title(f'{year}', fontsize=24)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    if idx > 0:
        ax.set_yticklabels([])

    # Save correlation matrix
    corr.to_csv(f'leaf_spearman_corr_{year}.csv')

    # Print high correlations
    high_corr = corr[(corr.abs() > 0.5) & (corr.abs() < 1.0)].stack().reset_index()
    high_corr.columns = ['Variable 1', 'Variable 2', 'Correlation']
    print(f"\nHigh correlations for year {year}:\n", high_corr)

# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.3, 0.015, 0.4])
norm = plt.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, ticks=np.linspace(vmin, vmax, 3))
cbar.ax.set_yticklabels([f"{v:.2f}" for v in np.linspace(vmin, vmax, 3)], fontsize=20)
cbar.set_label('Spearman Correlation', fontsize=20)
cbar.outline.set_visible(False)

# Final layout
plt.subplots_adjust(left=0.05, right=1.01, top=0.9, bottom=0.2, wspace=-0.4)
plt.savefig('leaf_combined_spearman_corr_heatmaps.png', bbox_inches='tight', facecolor='white')
plt.show()

#print("Saved heatmap and correlation matrix.")
