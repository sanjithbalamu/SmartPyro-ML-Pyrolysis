import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('pyrolysis.csv')

print("=" * 50)
print("ORIGINAL DATA SHAPE")
print("=" * 50)
print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
print()

# Clean the column names (remove extra spaces)
data.columns = data.columns.str.strip()

# Convert output columns to numeric (they're currently objects/strings)
data['Solid phase'] = pd.to_numeric(data['Solid phase'], errors='coerce')
data['Liquid phase'] = pd.to_numeric(data['Liquid phase'], errors='coerce')
data['Gas phase'] = pd.to_numeric(data['Gas phase'], errors='coerce')

# Convert PS (particle size) to numeric
data['PS'] = pd.to_numeric(data['PS'], errors='coerce')

# Remove obviously wrong data
data = data[data['M'] > -50]  # Remove the -100 moisture value

print("=" * 50)
print("CLEANED DATA SHAPE")
print("=" * 50)
print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
print()

print("=" * 50)
print("MISSING VALUES AFTER CLEANING")
print("=" * 50)
missing = data.isnull().sum()
print(missing[missing > 0])
print()

print("=" * 50)
print("OUTPUT STATISTICS (What we want to predict)")
print("=" * 50)
print(data[['Solid phase', 'Liquid phase', 'Gas phase']].describe())
print()

print("=" * 50)
print("BIOMASS SPECIES COUNT")
print("=" * 50)
print(data['Biomass species'].value_counts())
print()

# Save cleaned data
data.to_csv('pyrolysis_cleaned.csv', index=False)
print("✓ Saved cleaned data to 'pyrolysis_cleaned.csv'")