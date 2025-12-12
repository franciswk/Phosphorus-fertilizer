"""
Two-Way ANOVA for Phosphorus Fertilizer Yield Study

Purpose: Analyze yield responses to phosphorus application and variety; clean data, visualize distributions, run ANOVA and Tukey HSD.

Input: merged_phosfert_021125.csv - Available upon request.

Data prep: Median-impute Soil_N%, Soil_C%, Soil_OM%; normalize headers (spaces/dots removed, % → _perc); set Variety, P_appln, Year, Plot as categorical.

Methods: Two-way ANOVA (C(Variety) + C(P_appln)) across No1_wt, Can_wt, Jumbo_wt, Mkt_wt; Tukey’s HSD within Variety and P_appln; boxplots by Variety, P_appln, Year.

Outputs: sum_stats_merged_phosfert_021125.csv, nonas_merged_phosfert_021125.csv, {measure}_Variety.png, {measure}_P_appln.png, {measure}_Year.png, anova_results.csv, tukey_hsd_results.csv.

Dependencies: pandas, seaborn, matplotlib, scipy, statsmodels, numpy.

Results published in https://doi.org/10.1002/saj2.70145
"""


#%%
#!pip install pandas seaborn matplotlib scipy statsmodels numpy
#%%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np

df = pd.read_csv('merged_phosfert_021125.csv')

print(df.info())
summary_stats = df.describe()

summary_stats.to_csv('sum_stats_merged_phosfert_021125.csv')
#%%
# Count missing values in each column
missing_counts = df.isnull().sum()

# Display columns with missing values only
print(missing_counts[missing_counts > 0])

#%%
# Impute missing values with median from affected numeric cols
numeric_cols = ['Soil_N%', 'Soil_C%', 'Soil_OM%']

# Impute missing values using median
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

#%%
df.columns = df.columns.str.replace(' ', '_').str.replace('.', '').str.replace('%', '_perc')
print(df.columns)

#%%
# Convert categorical variables
df[['Variety', 'P_appln', 'Year', 'Plot']] = df[['Variety', 'P_appln', 'Year', 'Plot']].astype('category')

print(df.info())

df.to_csv('nonas_merged_phosfert_021125.csv',  index=False)

#%%
# Boxplots to visualize yield distribution across Variety, Year, and P_appln
yield_measures = ["No1_wt", "Can_wt", "Jumbo_wt", "Mkt_wt"]
for measure in yield_measures:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Variety', y=measure, data=df)
    plt.xticks(rotation=90)
    plt.title(f"{measure} across Varieties")
    plt.savefig(f"{measure}_Variety.png", dpi=300, bbox_inches='tight')  # Save the plot
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='P_appln', y=measure, data=df)
    plt.title(f"{measure} across P_appln Levels")
    plt.savefig(f"{measure}_P_appln.png", dpi=300, bbox_inches='tight')  # Save the plot
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Year', y=measure, data=df)
    plt.title(f"{measure} across Years")
    plt.savefig(f"{measure}_Year.png", dpi=300, bbox_inches='tight')  # Save the plot
    plt.show()

#%%
# Perform ANOVA for each yield measure
anova_results = {}
for measure in yield_measures:
    model = smf.ols(f"{measure} ~ C(Variety) + C(P_appln)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)  # Type II ANOVA table
    anova_results[measure] = anova_table

# Display ANOVA results
for measure, result in anova_results.items():
    print(f"ANOVA Results for {measure}:\n")
    print(result)
    print("-" * 50)
#%%

# List of dependent variables (yield measures)
yield_measures = ['No1_wt', 'Can_wt', 'Jumbo_wt', 'Mkt_wt']

# List of categorical factors
factors = ['Variety', 'P_appln']

# Loop over each yield measure and each factor
for measure in yield_measures:
    for factor in factors:
        # Perform Tukey HSD test for each combination of yield measure and factor
        tukey_result = pairwise_tukeyhsd(endog=df[measure],  # Dependent variable
                                         groups=df[factor],  # Independent variable
                                         alpha=0.05)
        
        # Print the results for each combination
        print(f"\nTukey HSD for {measure} by {factor}:")
        print(tukey_result)

#%%
# To save the results
# Prepare data for ANOVA results
anova_data = []

# Perform ANOVA for each yield measure
yield_measures = ['No1_wt', 'Can_wt', 'Jumbo_wt', 'Mkt_wt']
for measure in yield_measures:
    model = smf.ols(f"{measure} ~ C(Variety) + C(P_appln)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)  # Type II ANOVA table
    anova_data.append({
        'Measure': measure,
        'Sum of Squares (Variety)': anova_table.loc['C(Variety)', 'sum_sq'],
        'df (Variety)': anova_table.loc['C(Variety)', 'df'],
        'F-Statistic (Variety)': anova_table.loc['C(Variety)', 'F'],
        'p-value (Variety)': anova_table.loc['C(Variety)', 'PR(>F)'],
        'Sum of Squares (P_appln)': anova_table.loc['C(P_appln)', 'sum_sq'],
        'df (P_appln)': anova_table.loc['C(P_appln)', 'df'],
        'F-Statistic (P_appln)': anova_table.loc['C(P_appln)', 'F'],
        'p-value (P_appln)': anova_table.loc['C(P_appln)', 'PR(>F)']
    })

# Create DataFrame for ANOVA results
anova_df = pd.DataFrame(anova_data)

# Save ANOVA results to CSV
anova_df.to_csv('anova_results.csv', index=False)

# Prepare data for Tukey HSD results
tukey_data = []

# List of categorical factors
factors = ['Variety', 'P_appln']

# Perform Tukey HSD test for each measure and factor
for measure in yield_measures:
    for factor in factors:
        tukey_result = pairwise_tukeyhsd(endog=df[measure],  # Dependent variable
                                         groups=df[factor],  # Independent variable
                                         alpha=0.05)
        
        # Collect Tukey HSD results
        for row in tukey_result.summary().data[1:]:  # Skipping the header
            tukey_data.append({
                'Measure': measure,
                'Factor': factor,
                'Group1': row[0],
                'Group2': row[1],
                'Mean Difference': row[2],
                'p-adj': row[3],
                'Reject Null': row[6]
            })

# Create DataFrame for Tukey HSD results
tukey_df = pd.DataFrame(tukey_data)

# Save Tukey HSD results to CSV
tukey_df.to_csv('tukey_hsd_results.csv', index=False)

print("Results saved to 'anova_results.csv' and 'tukey_hsd_results.csv'.")

