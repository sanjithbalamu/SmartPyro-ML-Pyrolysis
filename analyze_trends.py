import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv('pyrolysis_cleaned.csv')

print("=" * 60)
print("ANALYZING TEMPERATURE TRENDS IN TRAINING DATA")
print("=" * 60)
print()

# Create temperature bins
data['Temp_bin'] = pd.cut(data['FT'], bins=[0, 450, 550, 650, 1000], 
                           labels=['<450°C', '450-550°C', '550-650°C', '>650°C'])

# Group by temperature and calculate means
temp_analysis = data.groupby('Temp_bin')[['Solid phase', 'Liquid phase', 'Gas phase', 'FT']].agg(['mean', 'std', 'count'])

print("Average yields by temperature range:")
print(temp_analysis)
print()

# Detailed analysis
print("=" * 60)
print("AVERAGE YIELDS BY TEMPERATURE RANGE")
print("=" * 60)
print()

for temp_range in ['<450°C', '450-550°C', '550-650°C', '>650°C']:
    subset = data[data['Temp_bin'] == temp_range]
    if len(subset) > 0:
        print(f"{temp_range}:")
        print(f"  Samples: {len(subset)}")
        print(f"  Biochar:  {subset['Solid phase'].mean():.2f}% ± {subset['Solid phase'].std():.2f}%")
        print(f"  Bio-oil:  {subset['Liquid phase'].mean():.2f}% ± {subset['Liquid phase'].std():.2f}%")
        print(f"  Syngas:   {subset['Gas phase'].mean():.2f}% ± {subset['Gas phase'].std():.2f}%")
        print()

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, phase in enumerate(['Solid phase', 'Liquid phase', 'Gas phase']):
    ax = axes[idx]
    
    # Scatter plot
    ax.scatter(data['FT'], data[phase], alpha=0.3, s=20)
    
    # Trend line
    z = np.polyfit(data['FT'].dropna(), data[phase].dropna(), 2)
    p = np.poly1d(z)
    temps = np.linspace(data['FT'].min(), data['FT'].max(), 100)
    ax.plot(temps, p(temps), "r--", linewidth=2, label='Trend')
    
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel(f'{phase} (%)')
    ax.set_title(phase)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('temperature_trends.png', dpi=150, bbox_inches='tight')
print("✓ Saved: temperature_trends.png")
print()

# Correlation analysis
print("=" * 60)
print("CORRELATION: Temperature vs Yields")
print("=" * 60)
print()

for phase in ['Solid phase', 'Liquid phase', 'Gas phase']:
    corr = data[['FT', phase]].corr().iloc[0, 1]
    print(f"{phase:20} correlation with temp: {corr:+.3f}")

print()
print("Expected:")
print("  Biochar:  negative correlation (decreases with temp)")
print("  Bio-oil:  weak/parabolic (peaks mid-range)")
print("  Syngas:   positive correlation (increases with temp)")