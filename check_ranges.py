import pandas as pd

data = pd.read_csv('pyrolysis_cleaned.csv')

print("=" * 60)
print("TRAINING DATA RANGES")
print("=" * 60)
print()

features = ['M', 'Ash', 'VM', 'FC', 'C', 'H', 'O', 'N', 'PS', 'FT', 'HR', 'FR']

for feat in features:
    print(f"{feat:15} Min: {data[feat].min():8.2f}  Max: {data[feat].max():8.2f}  Mean: {data[feat].mean():8.2f}")

print()
print("=" * 60)
print("OUTPUT RANGES")
print("=" * 60)
print()

outputs = ['Solid phase', 'Liquid phase', 'Gas phase']
for out in outputs:
    print(f"{out:15} Min: {data[out].min():8.2f}  Max: {data[out].max():8.2f}  Mean: {data[out].mean():8.2f}")