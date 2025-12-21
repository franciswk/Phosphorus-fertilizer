# Phosphorus Fertilizer Study - Analysis Scripts

Analysis scripts for a multi-year sweet potato phosphorus fertilizer study examining effects on yield, root quality, and nutrient dynamics.

## Publication

**DOI:** [10.1002/saj2.70145](https://doi.org/10.1002/saj2.70145)

Please cite this publication when using these scripts.

## Overview

The workflow consists of:
1. **Data merging** - Combining fertilizer application with yield and nutrient measurements
2. **Statistical analysis** - ANOVA and correlation analyses  
3. **Visualization** - Yield plots, regression models, and nutrient heatmaps

## Scripts

### Data Merging

**`merging_phosfert_021925.py`** - Merges fertilizer application data with yield, leaf, soil, and root nutrient datasets. See docstring for detailed input/output files.

### Statistical Analysis

**`corln_phsofert_021225.py`** - Two-way ANOVA and Tukey's HSD for yield responses to Phosphorus application.

**`corelation_hm_050625.py`** - Year-by-year Spearman correlation heatmaps across yield, soil, and tissue nutrients.

*Note:* To analyze root or soil nutrients, modify the merge line to use `df_root` or `df_soil` instead of `df_leaf`.

### Visualization

**`combined_yield_plot_102725.ipynb`** - Multi-year line plots of yield by root quality class across phosphorus rates.

**`combined_quadratic_regression_102725.ipynb`** - Quadratic regression to identify optimal phosphorus fertilizer rate.

**`leaf_heatmaps_050625.py`** - Multi-year heatmaps of leaf nutrient levels with logit transformation.

**`root_heatmaps_050625.py`** - Multi-year heatmaps of root nutrient levels with logit transformation.

**`soil_heatmaps_050225.py`** - Multi-year heatmaps of soil nutrient levels with logit transformation.

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn