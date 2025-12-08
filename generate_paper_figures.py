import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import r2_score, mean_absolute_error

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data and models
data = pd.read_csv('pyrolysis_cleaned.csv')
models = joblib.load('models_improved.pkl')
scaler = joblib.load('scaler_improved.pkl')
imputer = joblib.load('imputer_improved.pkl')
feature_cols = joblib.load('feature_cols_improved.pkl')

print("=" * 60)
print("GENERATING PUBLICATION-QUALITY FIGURES")
print("=" * 60)
print()

# Prepare test data
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

X = data[feature_cols].copy()
y = data[['Solid phase', 'Liquid phase', 'Gas phase']].copy()

imputer_y = SimpleImputer(strategy='median')
y_imputed = pd.DataFrame(imputer_y.fit_transform(y), columns=y.columns)

X_imputed = pd.DataFrame(imputer.transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y_imputed, test_size=0.2, random_state=42
)

X_test_scaled = scaler.transform(X_test)

# Get predictions
predictions = {}
for target, model in models.items():
    predictions[target] = model.predict(X_test_scaled)

# ==================================================
# FIGURE 1: Model Performance (Actual vs Predicted)
# ==================================================

print("Creating Figure 1: Model Performance...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for idx, target in enumerate(['Solid phase', 'Liquid phase', 'Gas phase']):
    ax = axes[idx]
    
    actual = y_test[target]
    pred = predictions[target]
    
    # Scatter
    ax.scatter(actual, pred, alpha=0.5, s=40, edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(actual.min(), pred.min())
    max_val = max(actual.max(), pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Perfect prediction')
    
    # Stats
    r2 = r2_score(actual, pred)
    mae = mean_absolute_error(actual, pred)
    
    ax.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.2f}%', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Actual Yield (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Yield (%)', fontsize=12, fontweight='bold')
    ax.set_title(target, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('Figure1_ModelPerformance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Figure1_ModelPerformance.png")
print()

# ==================================================
# FIGURE 2: Temperature Effect on Product Distribution
# ==================================================

print("Creating Figure 2: Temperature Effect...")

base_conditions = {
    'M': data['M'].median(),
    'Ash': data['Ash'].median(),
    'VM': data['VM'].median(),
    'FC': data['FC'].median(),
    'C': data['C'].median(),
    'H': data['H'].median(),
    'O': data['O'].median(),
    'N': data['N'].median(),
    'PS': data['PS'].median(),
    'HR': data['HR'].median(),
    'FR': data['FR'].median()
}

temperatures = np.linspace(350, 750, 50)
temp_results = {'temp': [], 'Biochar': [], 'Bio-oil': [], 'Syngas': []}

for temp in temperatures:
    test = base_conditions.copy()
    test['FT'] = temp
    
    df = pd.DataFrame([test])[feature_cols]
    X_imp = imputer.transform(df)
    X_sc = scaler.transform(X_imp)
    
    preds = {}
    for target, model in models.items():
        preds[target] = model.predict(X_sc)[0]
    
    total = sum(preds.values())
    
    temp_results['temp'].append(temp)
    temp_results['Biochar'].append((preds['Solid phase']/total)*100)
    temp_results['Bio-oil'].append((preds['Liquid phase']/total)*100)
    temp_results['Syngas'].append((preds['Gas phase']/total)*100)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(temp_results['temp'], temp_results['Biochar'], 
        'o-', linewidth=2.5, markersize=6, label='Biochar (Solid)', color='#8B4513')
ax.plot(temp_results['temp'], temp_results['Bio-oil'], 
        's-', linewidth=2.5, markersize=6, label='Bio-oil (Liquid)', color='#FF8C00')
ax.plot(temp_results['temp'], temp_results['Syngas'], 
        '^-', linewidth=2.5, markersize=6, label='Syngas (Gas)', color='#4169E1')

ax.set_xlabel('Pyrolysis Temperature (°C)', fontsize=13, fontweight='bold')
ax.set_ylabel('Product Yield (%)', fontsize=13, fontweight='bold')
ax.set_title('Effect of Temperature on Product Distribution', fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(350, 750)

plt.tight_layout()
plt.savefig('Figure2_TemperatureEffect.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Figure2_TemperatureEffect.png")
print()

# ==================================================
# FIGURE 3: Feature Importance Comparison
# ==================================================

print("Creating Figure 3: Feature Importance...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

feature_names_clean = {
    'M': 'Moisture',
    'Ash': 'Ash Content',
    'VM': 'Volatile Matter',
    'FC': 'Fixed Carbon',
    'C': 'Carbon',
    'H': 'Hydrogen',
    'O': 'Oxygen',
    'N': 'Nitrogen',
    'PS': 'Particle Size',
    'FT': 'Temperature',
    'HR': 'Heating Rate',
    'FR': 'Flow Rate'
}

for idx, target in enumerate(['Solid phase', 'Liquid phase', 'Gas phase']):
    ax = axes[idx]
    
    importance = models[target].feature_importances_
    feature_importance = pd.DataFrame({
        'feature': [feature_names_clean.get(f, f) for f in feature_cols],
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
    
    ax.barh(range(len(feature_importance)), feature_importance['importance'], color=colors)
    ax.set_yticks(range(len(feature_importance)))
    ax.set_yticklabels(feature_importance['feature'], fontsize=10)
    ax.set_xlabel('Importance', fontsize=11, fontweight='bold')
    ax.set_title(target, fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('Figure3_FeatureImportance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Figure3_FeatureImportance.png")
print()

# ==================================================
# FIGURE 4: Residual Plots
# ==================================================

print("Creating Figure 4: Residual Analysis...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for idx, target in enumerate(['Solid phase', 'Liquid phase', 'Gas phase']):
    ax = axes[idx]
    
    actual = y_test[target]
    pred = predictions[target]
    residuals = actual - pred
    
    # Scatter residuals
    ax.scatter(pred, residuals, alpha=0.5, s=40, edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    
    # Add ±2 std lines
    std_residual = residuals.std()
    ax.axhline(y=2*std_residual, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axhline(y=-2*std_residual, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Predicted Yield (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residual (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{target}\nResidual Plot', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Figure4_Residuals.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Figure4_Residuals.png")
print()

# ==================================================
# FIGURE 5: Heatmap - Temperature vs Heating Rate
# ==================================================

print("Creating Figure 5: 2D Optimization Heatmap...")

temps = np.linspace(400, 700, 20)
heating_rates = np.linspace(10, 100, 20)

biooil_matrix = np.zeros((len(heating_rates), len(temps)))

for i, hr in enumerate(heating_rates):
    for j, temp in enumerate(temps):
        test = base_conditions.copy()
        test['FT'] = temp
        test['HR'] = hr
        
        df = pd.DataFrame([test])[feature_cols]
        X_imp = imputer.transform(df)
        X_sc = scaler.transform(X_imp)
        
        preds = {}
        for target, model in models.items():
            preds[target] = model.predict(X_sc)[0]
        
        total = sum(preds.values())
        biooil_matrix[i, j] = (preds['Liquid phase']/total)*100

fig, ax = plt.subplots(figsize=(10, 7))

im = ax.contourf(temps, heating_rates, biooil_matrix, levels=15, cmap='YlOrRd')
contours = ax.contour(temps, heating_rates, biooil_matrix, levels=10, colors='black', linewidths=0.5, alpha=0.4)
ax.clabel(contours, inline=True, fontsize=8)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Bio-oil Yield (%)', fontsize=12, fontweight='bold')

ax.set_xlabel('Temperature (°C)', fontsize=13, fontweight='bold')
ax.set_ylabel('Heating Rate (°C/min)', fontsize=13, fontweight='bold')
ax.set_title('Bio-oil Yield Optimization Map', fontsize=14, fontweight='bold')

# Mark optimal point
max_idx = np.unravel_index(biooil_matrix.argmax(), biooil_matrix.shape)
optimal_hr = heating_rates[max_idx[0]]
optimal_temp = temps[max_idx[1]]
ax.plot(optimal_temp, optimal_hr, 'w*', markersize=20, markeredgecolor='black', markeredgewidth=2)
ax.text(optimal_temp, optimal_hr + 5, f'Optimal\n({optimal_temp:.0f}°C, {optimal_hr:.0f}°C/min)', 
        ha='center', fontsize=10, color='white', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

plt.tight_layout()
plt.savefig('Figure5_OptimizationHeatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Figure5_OptimizationHeatmap.png")
print()

# ==================================================
# SUMMARY
# ==================================================

print("=" * 60)
print("✓ ALL FIGURES GENERATED!")
print("=" * 60)
print()
print("Created:")
print("  1. Figure1_ModelPerformance.png - Actual vs Predicted")
print("  2. Figure2_TemperatureEffect.png - Temperature trends")
print("  3. Figure3_FeatureImportance.png - What matters most")
print("  4. Figure4_Residuals.png - Error analysis")
print("  5. Figure5_OptimizationHeatmap.png - 2D optimization")
print()
print("All figures are publication quality (300 DPI)")
print("Ready to insert into your research paper!")