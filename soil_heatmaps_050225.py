"""
Soil Nutrient Heatmaps — Multi-Year Comparison

Generates side-by-side heatmaps showing soil nutrient levels across phosphorus
fertilizer rates for multiple years. Applies logit transformation to stabilize
and spread nutrient values for better visualization.

Input:
- soil_phosfert_merged.csv (requires Year, P_appln, and soil nutrient columns)

Output:
- combined_leaf_heatmaps.png: multi-year soil nutrient heatmaps with shared colorbar

Method:
- Aggregates nutrients by Year and Phosphorus Fertilizer (kg/ha)
- Scales to (0.01, 0.99) and applies logit transform
- Plots one heatmap per year with consistent color scale
- Colorbar shows expit-transformed values (0–1 scale) for interpretability

Dependencies: pandas, numpy, matplotlib, seaborn, scipy.special, sklearn
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import logit, expit
from sklearn.preprocessing import MinMaxScaler

# Load dataset

df = pd.read_csv('soil_phosfert_merged.csv')
df.info()
df.head()

# Column name mapping
name_mapping = {
    "Variety": "Variety", "P_appln": "Phosphorus Fertilizer",  "Replicate": "Replicate", "Plot": "Plot", "Year": "Year",
    "No.1_wt": "US No.1", "Can_wt": "Canner", "Jumbo_wt": "Jumbo", "Mkt_wt": "Total marketable",
    "pH": "pH", "C%": "Carbon", "OM%": "Organic matter", "CEC_meq": "Cation exchange", "Partial_charge_H": "Hydrogen ions",
    "N%": "Nitrogen", "P%": "Phosphorus", "K%": "Potassium", "Ca%": "Calcium", "Mg%": "Magnesium", "S%": "Sulfur",
    "P_ppa": "Phosphorus", "K_ppa": "Potassium", "Ca_ppa": "Calcium", "Mg_ppa": "Magnesium",
    "Na%": "Sodium", "Na_ppa": "Sodium", "B_ppm": "Boron", "Zn_ppm": "Zinc", "Zn_ppa": "Zinc",
    "Mn_ppm": "Manganese", "Fe_ppm": "Iron", "Cu_ppm": "Copper", "Co_ppm": "Cobalt", "Al_ppm": "Aluminium"
}

# Filter and rename columns
df = df[[col for col in name_mapping if col in df]].rename(columns=name_mapping)

# Convert P_appln from lb/A to kg/ha
df['Phosphorus Fertilizer'] = (df['Phosphorus Fertilizer'] * 1.12085).astype(int)

# Convert identifiers
categoric_cols = ["Variety", "Phosphorus Fertilizer", "Replicate", "Plot", "Year"]
df[categoric_cols] = df[categoric_cols].astype('category')

df.to_csv('leaf_df_hm_050725.csv', index=False)

df.tail()

# Group and average
all_nutrients = df.groupby(['Year', 'Phosphorus Fertilizer'], observed=True).mean(numeric_only=True).reset_index()

# Scale and logit-transform
scaler = MinMaxScaler(feature_range=(0.01, 0.99))
scaled = scaler.fit_transform(all_nutrients.iloc[:, 2:])
all_nutrients.iloc[:, 2:] = logit(scaled)

# Melt for plotting
nutrient_cols = [col for col in all_nutrients.columns if col not in ['Year', 'Phosphorus Fertilizer']]
melted = all_nutrients.melt(id_vars=['Year', 'Phosphorus Fertilizer'],
                            value_vars=nutrient_cols,
                            var_name='Nutrient', value_name='Logit(Nutrient%)')

# Plotting
vmin, vmax = melted['Logit(Nutrient%)'].min(), melted['Logit(Nutrient%)'].max()
years = all_nutrients['Year'].cat.categories
fig, axes = plt.subplots(1, len(years), figsize=(len(years)*5.3, 5.3), squeeze=False)

for j, year in enumerate(years):
    ax = axes[0][j]
    pivot = melted[melted['Year'] == year].pivot(index='Nutrient', columns='Phosphorus Fertilizer', values='Logit(Nutrient%)')
    pivot = pivot.reindex(nutrient_cols)

    sns.heatmap(pivot, cmap='coolwarm', linewidths=0.5, vmin=vmin, vmax=vmax, ax=ax, cbar=False)
    ax.set_title(f'{year}', fontsize=28)
    ax.set_xlabel("Phosphorus Fertilizer (kg/ha)" if j == 1 else "", fontsize=28)
    ax.set_ylabel("")
    ax.tick_params(axis='x', rotation=45, labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    if j > 0:
        ax.set_yticklabels([])

# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.3, 0.015, 0.4])
norm = plt.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
sm.set_array([])
tick_values = np.linspace(vmin, vmax, 3)
tick_labels = [f"{expit(v):.2f}" for v in tick_values]
cbar = fig.colorbar(sm, cax=cbar_ax, ticks=tick_values)
cbar.ax.set_yticklabels(tick_labels, fontsize=24)
cbar.ax.yaxis.set_ticks_position('right')
cbar.set_label('Relative Nutrient Levels', fontsize=24)

plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05, wspace=0.1)
plt.savefig('combined_leaf_heatmaps.png', bbox_inches='tight')
plt.show()




#%%
##SINGLE HMS

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.special import logit
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv('soil_phosfert_merged.csv')
df[['Variety', 'P_appln', 'Replicate', 'Year']] = df[
    ['Variety', 'P_appln', 'Replicate', 'Year']].astype('category')

# Group by Variety, Year, and P_appln and calculate mean values for nutrients
all_nutrients = df.groupby(['Variety', 'Year', 'P_appln'], observed=True).agg({
    'P_ppa': 'mean',
    'K_ppa': 'mean',
    'N%': 'mean',
    'C%': 'mean',
    'OM%': 'mean', 
    'Ca_ppa': 'mean',
    'Mg_ppa': 'mean',
    'Zn_ppa': 'mean',
    'Na_ppa': 'mean',
    'CEC_meq': 'mean',
    'Partial_charge_H': 'mean',
    'pH': 'mean',
}).reset_index()

# Scale nutrient percentages to (0.01, 0.99) to avoid logit issues
scaler = MinMaxScaler(feature_range=(0.01, 0.99))
scaled_values = scaler.fit_transform(all_nutrients.iloc[:, 3:])
logit_transformed = logit(scaled_values)
all_nutrients.iloc[:, 3:] = logit_transformed

# Nutrient column renaming
name_mapping = {
    'pH': 'pH', 'N%': 'Nitrogen', 'P_ppa': 'Phosphorus', 'K_ppa': 'Potassium', 'Ca_ppa': 'Calcium',
    'Mg_ppa': 'Magnesium', 'Zn_ppa': 'Zinc', 'Na_ppa': 'Sodium', 'C%': 'Carbon',
    'OM%': 'Organic Matter', 'CEC_meq': 'Cation exchange capacity', 'Partial_charge_H': 'Hydrogen ions'
}

# Melt DataFrame for plotting
melted_df = all_nutrients.melt(id_vars=['Variety', 'Year', 'P_appln'], 
                                value_vars=list(name_mapping.keys()),
                                var_name='Nutrient', value_name='Logit(Nutrient%)')
melted_df['Nutrient'] = melted_df['Nutrient'].replace(name_mapping)

# Define nutrient order
nutrient_order = [name_mapping[col] for col in all_nutrients.columns[3:]]

# Determine global min and max for consistent color scale
vmin = melted_df['Logit(Nutrient%)'].min()
vmax = melted_df['Logit(Nutrient%)'].max()

# Create heatmaps
sns.set(style="whitegrid")

for variety in all_nutrients['Variety'].cat.categories:
    for year in all_nutrients['Year'].cat.categories:
        subset = melted_df[(melted_df['Variety'] == variety) & (melted_df['Year'] == year)]
        pivot_table = subset.pivot(index='Nutrient', columns='P_appln', values='Logit(Nutrient%)')
        pivot_table = pivot_table.reindex(nutrient_order)

        plt.figure(figsize=(6, 6))
        ax = sns.heatmap(pivot_table, annot=False, cmap='coolwarm', linewidths=0.5, vmin=vmin, vmax=vmax)

        plt.title(f'{variety} | {year}', fontsize=20)
        plt.xlabel('Phosphorus Applied (lb/A)', fontsize=16)
        plt.ylabel('', fontsize=16)
        plt.xticks(rotation=45, fontsize=14)
        plt.yticks(rotation=0, fontsize=14)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label('', fontsize=18)

        plt.savefig(f'soil_hm_{variety}_{year}.png', bbox_inches='tight')
        plt.show()

print("Plots saved")

#%%


