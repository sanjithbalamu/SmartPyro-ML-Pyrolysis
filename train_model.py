import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load cleaned data
data = pd.read_csv('pyrolysis_cleaned.csv')

print("=" * 60)
print("SMARTPYRO: PYROLYSIS YIELD PREDICTION MODEL")
print("=" * 60)
print()

# Remove rows where ALL outputs are missing
data = data.dropna(subset=['Solid phase', 'Liquid phase', 'Gas phase'], how='all')

print(f"Training samples: {len(data)}")
print()

# ==================================================
# STEP 1: PREPARE FEATURES (X) AND TARGETS (y)
# ==================================================

# Define input features
feature_cols = ['M', 'Ash', 'VM', 'FC', 'C', 'H', 'O', 'N', 'PS', 'FT', 'HR', 'FR']

# Define output targets
target_cols = ['Solid phase', 'Liquid phase', 'Gas phase']

# Encode biomass species (categorical -> numeric)
le = LabelEncoder()
data['Biomass_encoded'] = le.fit_transform(data['Biomass species'])
feature_cols.append('Biomass_encoded')

# Extract features and targets
X = data[feature_cols].copy()
y = data[target_cols].copy()

# ==================================================
# STEP 2: HANDLE MISSING VALUES
# ==================================================

print("Handling missing values...")

# Impute missing values with median
imputer_X = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer_X.fit_transform(X),
    columns=X.columns,
    index=X.index
)

imputer_y = SimpleImputer(strategy='median')
y_imputed = pd.DataFrame(
    imputer_y.fit_transform(y),
    columns=y.columns,
    index=y.index
)

print(f"✓ Features shape: {X_imputed.shape}")
print(f"✓ Targets shape: {y_imputed.shape}")
print()

# ==================================================
# STEP 3: SPLIT DATA (Train/Validation/Test)
# ==================================================

print("Splitting data...")

# First split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y_imputed, test_size=0.2, random_state=42
)

# Second split: 80% train, 20% validation (from the training set)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Validation set: {len(X_val)} samples")
print(f"✓ Test set: {len(X_test)} samples")
print()

# ==================================================
# STEP 4: SCALE FEATURES
# ==================================================

print("Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled")
print()

# ==================================================
# STEP 5: TRAIN MODELS (One for each output)
# ==================================================

print("=" * 60)
print("TRAINING MODELS...")
print("=" * 60)
print()

models = {}
predictions = {}

for target in target_cols:
    print(f"Training model for: {target}")
    print("-" * 40)
    
    # XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    # Train
    model.fit(
        X_train_scaled, 
        y_train[target],
        eval_set=[(X_val_scaled, y_val[target])],
        verbose=False
    )
    
    # Predict on validation set
    y_pred = model.predict(X_val_scaled)
    
    # Calculate metrics
    mae = mean_absolute_error(y_val[target], y_pred)
    rmse = np.sqrt(mean_squared_error(y_val[target], y_pred))
    r2 = r2_score(y_val[target], y_pred)
    
    print(f"  MAE:  {mae:.3f}%")
    print(f"  RMSE: {rmse:.3f}%")
    print(f"  R²:   {r2:.3f}")
    print()
    
    # Store model and predictions
    models[target] = model
    predictions[target] = y_pred

# ==================================================
# STEP 6: EVALUATE ON TEST SET
# ==================================================

print("=" * 60)
print("TEST SET PERFORMANCE")
print("=" * 60)
print()

test_predictions = {}

for target in target_cols:
    y_pred_test = models[target].predict(X_test_scaled)
    test_predictions[target] = y_pred_test
    
    mae = mean_absolute_error(y_test[target], y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test[target], y_pred_test))
    r2 = r2_score(y_test[target], y_pred_test)
    
    print(f"{target}:")
    print(f"  MAE:  {mae:.3f}%")
    print(f"  RMSE: {rmse:.3f}%")
    print(f"  R²:   {r2:.3f}")
    print()

# ==================================================
# STEP 7: VISUALIZE RESULTS
# ==================================================

print("Creating visualizations...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, target in enumerate(target_cols):
    ax = axes[idx]
    
    # Scatter plot: Actual vs Predicted
    ax.scatter(y_test[target], test_predictions[target], alpha=0.6, s=30)
    
    # Perfect prediction line
    min_val = min(y_test[target].min(), test_predictions[target].min())
    max_val = max(y_test[target].max(), test_predictions[target].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    
    ax.set_xlabel('Actual (%)', fontsize=10)
    ax.set_ylabel('Predicted (%)', fontsize=10)
    ax.set_title(f'{target}\nR² = {r2_score(y_test[target], test_predictions[target]):.3f}', 
                 fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_performance.png', dpi=150, bbox_inches='tight')
print("✓ Saved: model_performance.png")
print()

# ==================================================
# STEP 8: FEATURE IMPORTANCE
# ==================================================

print("Analyzing feature importance...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, target in enumerate(target_cols):
    importance = models[target].feature_importances_
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    ax = axes[idx]
    top_features = feature_importance.head(10)
    ax.barh(range(len(top_features)), top_features['importance'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance')
    ax.set_title(f'{target}\nTop 10 Features', fontweight='bold')
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
print("✓ Saved: feature_importance.png")
print()

# ==================================================
# STEP 9: SAVE MODELS
# ==================================================

print("Saving models and preprocessors...")

# Save everything needed for deployment
joblib.dump(models, 'models.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(imputer_X, 'imputer_X.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(feature_cols, 'feature_cols.pkl')

print("✓ Saved: models.pkl")
print("✓ Saved: scaler.pkl")
print("✓ Saved: imputer_X.pkl")
print("✓ Saved: label_encoder.pkl")
print()

print("=" * 60)
print("✓ MODEL TRAINING COMPLETE!")
print("=" * 60)