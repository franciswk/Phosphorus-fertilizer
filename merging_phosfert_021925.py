"""
Phosphorus Fertilizer Study - Data Merging Script

This script processes and merges multiple datasets from a phosphorus fertilizer 
study, combining data from different measurement categories (yield, leaf, soil, 
and root) into unified merged datasets for analysis.

The script performs the following operations:
1. Converts fertilizer application data from wide to long format
2. Merges yield data with fertilizer application data
3. Merges leaf nutrient data with fertilizer application data
4. Merges soil nutrient data with fertilizer application data
5. Merges root nutrient data with fertilizer application data
6. Combines yield and leaf datasets based on common identifiers

Input Files:
    - fert_appln.csv: Fertilizer application data
    - yield_phosfert_011524.csv: Yield measurements
    - leaf_phosfert_021925.csv: Leaf nutrient analysis
    - soil_phosfert_021925.csv: Soil nutrient analysis
    - roots_phosfert_031125.csv: Root nutrient analysis

Output Files:
    - fert_appln_long.csv: Fertilizer data in long format
    - yield_phosfert_merged.csv: Yield + fertilizer application data
    - leaf_phosfert_merged.csv: Leaf nutrients + fertilizer application data
    - soil_phosfert_merged.csv: Soil nutrients + fertilizer application data
    - root_phosfert_merged.csv: Root nutrients + fertilizer application data
    - merged_dataset_022725.csv: Combined yield and leaf nutrient data

Data Structure:
    Common identifiers: Variety, P_appln (phosphorus application rate), 
    Replicate, Plot, Year
"""

import pandas as pd
# Convert from wide to long format

# Read the CSV file
#df = pd.read_csv('fert_appln.csv')
df = pd.read_csv(r'C:\Users\frank\OneDrive - Mississippi State University\Research_0924\Phosphorus fertilizer\da_phosfert_0125\fert_appln.csv')

df.info()

# Drop unwanted column
df = df.drop(columns=['Trt_No'])

# Convert from wide to long format
df1 = pd.melt(df, 
                  id_vars=['Variety', 'P_appln'], 
                  var_name='Replicate', 
                  value_name='Plot')
df1.info()
df1.head()

# Convert categorical variables
df1[['Variety', 'P_appln', 'Replicate']] = df1[['Variety', 'P_appln', 'Replicate']].astype('category')

# Save the long format DataFrame to a new CSV file
df1.to_csv('fert_appln_long.csv', index=False)

print("Long format data saved to 'fert_appln_long.csv'")
#%%

# Merge two DataFrames based on the 'Plot' column
# YIELD

# Read the second CSV file
#df2 = pd.read_csv('yield_phosfert_011524.csv')
df2 = pd.read_csv(r'C:\Users\frank\OneDrive - Mississippi State University\Research_0924\Phosphorus fertilizer\da_phosfert_0125\yield_phosfert_011524.csv')

df2.info()
df2.head(),
# Convert categorical variables
df2[['Year']] = df2[['Year']].astype('category')

# Merge the two DataFrames based on the 'Plot' column
df3 = pd.merge(df1, df2, on='Plot')
df3.info()
df3.head()

# Save the merged DataFrame to a new CSV file
df3.to_csv('yield_phosfert_merged.csv', index=False)

print("Merged data saved to 'yield_phosfert_merged.csv'")

#%%
# LEAF
# Read the CSV file
df1 = pd.read_csv('fert_appln_long.csv')
df1.info()
df1[['Variety', 'P_appln', 'Replicate']] = df1[['Variety', 'P_appln', 'Replicate']].astype('category')

# Read the second CSV file
#df2 = pd.read_csv('leaf_phosfert_021925.csv')
df2 = pd.read_csv(r'C:\Users\frank\OneDrive - Mississippi State University\Research_0924\Phosphorus fertilizer\da_phosfert_0125\leaf_phosfert_021925.csv')

df2.info()
df2[['Year']] = df2[['Year']].astype('category')
df2.head()

# Merge the two DataFrames based on the 'Plot' column
df3 = pd.merge(df1, df2, on='Plot')
df3.info()
df3.head()

# Save the merged DataFrame to a new CSV file
df3.to_csv('leaf_phosfert_merged.csv', index=False)

print("Merged data saved to 'leaf_phosfert_merged.csv'")

#%%

# SOIL
# Read the CSV file
df1 = pd.read_csv('fert_appln_long.csv')
df1.info()
df1[['Variety', 'P_appln', 'Replicate']] = df1[['Variety', 'P_appln', 'Replicate']].astype('category')

# Read the second CSV file
#df2 = pd.read_csv('soil_phosfert_021925.csv')
df2 = pd.read_csv(r'C:\Users\frank\OneDrive - Mississippi State University\Research_0924\Phosphorus fertilizer\da_phosfert_0125\soil_phosfert_021925.csv')

df2.info()
df2[['Year']] = df2[['Year']].astype('category')
df2.head()

# Merge the two DataFrames based on the 'Plot' column
df3 = pd.merge(df1, df2, on='Plot')
df3.info()
df3.head()

# Save the merged DataFrame to a new CSV file
df3.to_csv('soil_phosfert_merged.csv', index=False)

print("Merged data saved to 'soil_phosfert_merged.csv'")
#%%
# ROOT
# Read the CSV file
df1 = pd.read_csv('fert_appln_long.csv')
df1.info()
df1[['Variety', 'P_appln', 'Replicate']] = df1[['Variety', 'P_appln', 'Replicate']].astype('category')

# Read the second CSV file
#df2 = pd.read_csv('root_phosfert_021925.csv')
#df2 = pd.read_csv('roots_phosfert_031125.csv')
df2 = pd.read_csv(r'C:\Users\frank\OneDrive - Mississippi State University\Research_0924\Phosphorus fertilizer\da_phosfert_0125\roots_phosfert_031125.csv')


df2.info()
df2[['Year']] = df2[['Year']].astype('category')
df2.head()

# Merge the two DataFrames based on the 'Plot' column
df3 = pd.merge(df1, df2, on='Plot')
df3.info()
df3.head()

# Save the merged DataFrame to a new CSV file
df3.to_csv('root_phosfert_merged.csv', index=False)

print("Merged data saved to 'root_phosfert_merged.csv'")
#%%

df_leaf = pd.read_csv('leaf_phosfert_merged.csv')
df_root = pd.read_csv('root_phosfert_merged.csv')
df_soil = pd.read_csv('soil_phosfert_merged.csv')
df_yield = pd.read_csv('yield_phosfert_merged.csv')

df_leaf.info()
df_soil.info()
df_yield.info()
df_root.info()
#%%
# Load datasets (assuming they are named df1, df2, df3)
# Ensure identifiers are of the same type
common_cols = ["Variety", "P_appln", "Replicate", "Plot", "Year"]

for df in [df_leaf, df_soil, df_root, df_yield]:
    df[common_cols] = df[common_cols].astype(str)  # Convert to string for consistency

# Merge the first two datasets
merged_df = pd.merge(df_yield, df_leaf, on=common_cols, how='inner')
merged_df.info()

# Merge the result with the third dataset
#merged_df = pd.merge(merged_df, df_leaf, on=common_cols, how='inner')

# Merge the result with the third dataset
#merged_df = pd.merge(merged_df, df_root, on=common_cols, how='inner')

# Display merged dataset info
print(merged_df.info())

# Save to CSV (optional)
merged_df.to_csv("merged_dataset_022725.csv", index=False)

print("Nutrients compared to Yield. Merged data saved to 'merged_dataset_022725.csv'")












#%%

'''

# Yield+Leaf Merge
df4 = pd.read_csv('leaf_phosfert_merged.csv')
df4.info()

df4[['Year']] = df4[['Year']].astype('category')

df4_selected = df4[['Plot', 'Leaf_P']]
df4_selected.info()

# Merge the two DataFrames based on the 'Plot' column
df3_4 = pd.merge(df3, df4_selected, on='Plot')
df3_4.info()
#%%
# Yield+Leaf+Soil Merge
df5 = pd.read_csv('soil_phosfert_merged.csv')
df5.info()

df5_selected = df5[['Plot', 'Soil_P']]
df5_selected.info()

# Merge the two DataFrames based on the 'Plot' column
df3_4_5 = pd.merge(df3_4, df5_selected, on='Plot')
df3_4_5.info()
df3_4_5.head()
#%%
# Yield+Leaf+Soil+Root Merge
df6 = pd.read_csv('root_phosfert_011424.csv')
df6.info()

df6_selected = df6[['Plot', 'Root_P']]
df6_selected.info()

# Merge the two DataFrames based on the 'Plot' column
df3_4_5_6 = pd.merge(df3_4_5, df6_selected, on='Plot')
df3_4_5_6.info()
df3_4_5_6.head()

# Save the merged DataFrame to a new CSV file
df3_4_5_6.to_csv('merged_phosfert.csv', index=False)

print("Merged data saved to 'merged_phosfert.csv'")
#%%
'''