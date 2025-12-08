import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib

# Load cleaned data
data = pd.read_csv('pyrolysis_cleaned.csv')

print("=" * 60)
print("SMARTPYRO: IMPROVED MODEL WITH PHYSICS CONSTRAINTS")
print("=" * 60)
print()

# Remove rows where ALL outputs are missing
data = data.dropna(subset=['Solid phase', 'Liquid phase', 'Gas phase'], how='all')

# Define input features
feature_cols = ['M', 'Ash', 'VM', 'FC', 'C', 'H', 'O', 'N', 'PS', 'FT', 'HR', 'FR']

# Define output targets
target_cols = ['Solid phase', 'Liquid phase', 'Gas phase']

# Extract features and targets
X = data[feature_cols].copy()
y = data[target_cols].copy()

# Impute missing values
imputer_X = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer_X.fit_transform(X), columns=X.columns, index=X.index)

imputer_y = SimpleImputer(strategy='median')
y_imputed = pd.DataFrame(imputer_y.fit_transform(y), columns=y.columns, index=y.index)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y_imputed, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")
print()

# ==================================================
# DEFINE MONOTONIC CONSTRAINTS
# ==================================================
# FT is at index 9 in feature_cols
# We want: Biochar ↓ with temp, Syngas ↑ with temp

monotone_constraints = {
    'Solid phase': tuple([0]*9 + [-1] + [0]*2),   # Biochar decreases with FT
    'Gas phase': tuple([0]*9 + [1] + [0]*2),      # Syngas increases with FT
    'Liquid phase': tuple([0]*12)                  # Bio-oil no constraint (parabolic)
}

print("=" * 60)
print("TRAINING IMPROVED MODELS WITH PHYSICS CONSTRAINTS")
print("=" * 60)
print()

models = {}
predictions_test = {}

for target in target_cols:
    print(f"Training: {target}")
    print("-" * 40)
    
    # XGBoost with improved hyperparameters and monotonic constraints
    params = {
        'n_estimators': 300,
        'max_depth': 4,              # Reduced to prevent overfitting
        'learning_rate': 0.05,        # Slower learning
        'subsample': 0.8,             # Use 80% of data per tree
        'colsample_bytree': 0.8,      # Use 80% of features per tree
        'reg_alpha': 1.0,             # L1 regularization
        'reg_lambda': 1.0,            # L2 regularization
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Add monotonic constraint if applicable
    if target in monotone_constraints:
        params['monotone_constraints'] = monotone_constraints[target]
    
    model = xgb.XGBRegressor(**params)
    
    # Train with early stopping
    model.fit(
        X_train_scaled, 
        y_train[target],
        eval_set=[(X_val_scaled, y_val[target])],
        verbose=False
    )
    
    # Predict on test set
    y_pred_test = model.predict(X_test_scaled)
    predictions_test[target] = y_pred_test
    
    # Metrics
    mae = mean_absolute_error(y_test[target], y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test[target], y_pred_test))
    r2 = r2_score(y_test[target], y_pred_test)
    
    print(f"  MAE:  {mae:.3f}%")
    print(f"  RMSE: {rmse:.3f}%")
    print(f"  R²:   {r2:.3f}")
    print()
    
    models[target] = model

# ==================================================
# TEST TEMPERATURE TRENDS
# ==================================================

print("=" * 60)
print("VALIDATING TEMPERATURE TRENDS")
print("=" * 60)
print()

test_conditions = {
    'M': 6.0,
    'Ash': 5.0,
    'VM': 75.0,
    'FC': 14.0,
    'C': 47.0,
    'H': 6.2,
    'O': 45.0,
    'N': 1.5,
    'PS': 0.5,
    'HR': 20.0,
    'FR': 100.0
}

temperatures = [400, 450, 500, 550, 600, 650, 700]
trend_results = []

print(f"{'Temp (°C)':<12} {'Biochar %':<12} {'Bio-oil %':<12} {'Syngas %':<12} {'Total %':<12}")
print("-" * 60)

for temp in temperatures:
    # Create input
    test_input = test_conditions.copy()
    test_input['FT'] = temp
    
    X_test_input = pd.DataFrame([test_input])[feature_cols]
    X_test_imputed = imputer_X.transform(X_test_input)
    X_test_scaled_input = scaler.transform(X_test_imputed)
    
    # Predict
    preds = {}
    for target, model in models.items():
        preds[target] = model.predict(X_test_scaled_input)[0]
    
    # Normalize to 100%
    total = sum(preds.values())
    preds_norm = {k: (v/total)*100 for k, v in preds.items()}
    
    trend_results.append({
        'temp': temp,
        'biochar': preds_norm['Solid phase'],
        'biooil': preds_norm['Liquid phase'],
        'syngas': preds_norm['Gas phase']
    })
    
    print(f"{temp:<12} {preds_norm['Solid phase']:<12.2f} {preds_norm['Liquid phase']:<12.2f} {preds_norm['Gas phase']:<12.2f} {sum(preds_norm.values()):<12.2f}")

print()

# Validate trends
biochar_vals = [r['biochar'] for r in trend_results]
syngas_vals = [r['syngas'] for r in trend_results]
biooil_vals = [r['biooil'] for r in trend_results]

biochar_decreases = all(biochar_vals[i] >= biochar_vals[i+1] for i in range(len(biochar_vals)-1))
syngas_increases = all(syngas_vals[i] <= syngas_vals[i+1] for i in range(len(syngas_vals)-1))
biooil_peaks_mid = biooil_vals.index(max(biooil_vals)) in [2, 3, 4]  # Should peak around 500-600°C

print("Trend validation:")
print(f"  Biochar decreases with temp: {'✓ PASS' if biochar_decreases else '✗ FAIL'}")
print(f"  Syngas increases with temp:  {'✓ PASS' if syngas_increases else '✗ FAIL'}")
print(f"  Bio-oil peaks mid-range:     {'✓ PASS' if biooil_peaks_mid else '✗ FAIL'}")
print()

# Save improved models
joblib.dump(models, 'models_improved.pkl')
joblib.dump(scaler, 'scaler_improved.pkl')
joblib.dump(imputer_X, 'imputer_improved.pkl')
joblib.dump(feature_cols, 'feature_cols_improved.pkl')

print("=" * 60)
print("✓ IMPROVED MODEL TRAINING COMPLETE!")
print("=" * 60)
print()
print("Saved:")
print("  - models_improved.pkl")
print("  - scaler_improved.pkl")
print("  - imputer_improved.pkl")