import pandas as pd
import numpy as np
import joblib

# Load saved models and preprocessors
models = joblib.load('models.pkl')
scaler = joblib.load('scaler.pkl')
imputer_X = joblib.load('imputer_X.pkl')
label_encoder = joblib.load('label_encoder.pkl')
feature_cols = joblib.load('feature_cols.pkl')

print("=" * 60)
print("SMARTPYRO: CONSTRAINED PYROLYSIS PREDICTOR")
print("=" * 60)
print()

def predict_yields_constrained(input_data, normalize=True):
    """
    Predict pyrolysis yields with mass balance constraint
    
    Parameters:
    - input_data: dictionary with keys matching feature names
    - normalize: if True, scale outputs to sum to 100%
    """
    
    # Convert to DataFrame
    df = pd.DataFrame([input_data])
    
    # Encode biomass species if provided
    if 'Biomass species' in df.columns:
        try:
            df['Biomass_encoded'] = label_encoder.transform(df['Biomass species'])
        except:
            df['Biomass_encoded'] = 0
    
    # Select and order features
    X = df[feature_cols].copy()
    
    # Handle missing values
    X_imputed = imputer_X.transform(X)
    
    # Scale features
    X_scaled = scaler.transform(X_imputed)
    
    # Predict each output
    predictions = {}
    raw_predictions = {}
    
    for target, model in models.items():
        pred = model.predict(X_scaled)[0]
        raw_predictions[target] = pred
        predictions[target] = pred
    
    # Apply mass balance constraint
    if normalize:
        total = sum(predictions.values())
        # Normalize to sum to 100%
        for target in predictions:
            predictions[target] = (predictions[target] / total) * 100
    
    return predictions, raw_predictions


# ==================================================
# TEST 1: Low Temperature Pyrolysis
# ==================================================

print("TEST 1: Low Temperature (400°C) - Expect More Biochar")
print("-" * 60)

test1 = {
    'M': 6.5,
    'Ash': 5.2,
    'VM': 75,
    'FC': 14,
    'C': 48,
    'H': 6.1,
    'O': 44,
    'N': 1.4,
    'PS': 0.5,
    'FT': 400,
    'HR': 10,
    'FR': 100,
    'Biomass_encoded': 0
}

constrained, raw = predict_yields_constrained(test1, normalize=True)

print("Raw predictions (unconstrained):")
print(f"  Solid phase (biochar):  {raw['Solid phase']:.2f}%")
print(f"  Liquid phase (bio-oil): {raw['Liquid phase']:.2f}%")
print(f"  Gas phase (syngas):     {raw['Gas phase']:.2f}%")
print(f"  Total:                  {sum(raw.values()):.2f}%")
print()

print("Constrained predictions (normalized to 100%):")
print(f"  Solid phase (biochar):  {constrained['Solid phase']:.2f}%")
print(f"  Liquid phase (bio-oil): {constrained['Liquid phase']:.2f}%")
print(f"  Gas phase (syngas):     {constrained['Gas phase']:.2f}%")
print(f"  Total:                  {sum(constrained.values()):.2f}%")
print()

# ==================================================
# TEST 2: Optimal Bio-oil Temperature
# ==================================================

print("=" * 60)
print("TEST 2: Optimal Temperature (550°C) - Expect Max Bio-oil")
print("-" * 60)

test2 = {
    'M': 5,
    'Ash': 4,
    'VM': 76,
    'FC': 15,
    'C': 47,
    'H': 6.3,
    'O': 45,
    'N': 1.2,
    'PS': 0.5,
    'FT': 550,
    'HR': 20,
    'FR': 100,
    'Biomass_encoded': 0
}

constrained2, raw2 = predict_yields_constrained(test2, normalize=True)

print("Constrained predictions:")
print(f"  Solid phase (biochar):  {constrained2['Solid phase']:.2f}%")
print(f"  Liquid phase (bio-oil): {constrained2['Liquid phase']:.2f}%")
print(f"  Gas phase (syngas):     {constrained2['Gas phase']:.2f}%")
print()

# ==================================================
# TEST 3: High Temperature
# ==================================================

print("=" * 60)
print("TEST 3: High Temperature (700°C) - Expect More Gas")
print("-" * 60)

test3 = {
    'M': 5,
    'Ash': 3.5,
    'VM': 78,
    'FC': 13,
    'C': 49,
    'H': 6.5,
    'O': 43,
    'N': 1.0,
    'PS': 0.5,
    'FT': 700,
    'HR': 50,
    'FR': 150,
    'Biomass_encoded': 0
}

constrained3, raw3 = predict_yields_constrained(test3, normalize=True)

print("Constrained predictions:")
print(f"  Solid phase (biochar):  {constrained3['Solid phase']:.2f}%")
print(f"  Liquid phase (bio-oil): {constrained3['Liquid phase']:.2f}%")
print(f"  Gas phase (syngas):     {constrained3['Gas phase']:.2f}%")
print()

# ==================================================
# COMPARISON TABLE
# ==================================================

print("=" * 60)
print("SUMMARY: Temperature Effect on Product Distribution")
print("=" * 60)
print()
print(f"{'Temperature':<15} {'Biochar %':<12} {'Bio-oil %':<12} {'Syngas %':<12}")
print("-" * 60)
print(f"{'400°C':<15} {constrained['Solid phase']:<12.2f} {constrained['Liquid phase']:<12.2f} {constrained['Gas phase']:<12.2f}")
print(f"{'550°C':<15} {constrained2['Solid phase']:<12.2f} {constrained2['Liquid phase']:<12.2f} {constrained2['Gas phase']:<12.2f}")
print(f"{'700°C':<15} {constrained3['Solid phase']:<12.2f} {constrained3['Liquid phase']:<12.2f} {constrained3['Gas phase']:<12.2f}")
print()

# Check trends
print("Expected trends:")
print("  ✓ Biochar should DECREASE with temperature")
print("  ✓ Bio-oil should be HIGHEST around 500-550°C")
print("  ✓ Syngas should INCREASE with temperature")
print()

biochar_trend = constrained['Solid phase'] > constrained2['Solid phase'] > constrained3['Solid phase']
biooil_peak = constrained2['Liquid phase'] > constrained['Liquid phase'] and constrained2['Liquid phase'] >= constrained3['Liquid phase']
syngas_trend = constrained3['Gas phase'] > constrained2['Gas phase'] > constrained['Gas phase']

print("Model validation:")
print(f"  Biochar decreases: {'✓ PASS' if biochar_trend else '✗ FAIL'}")
print(f"  Bio-oil peaks at 550°C: {'✓ PASS' if biooil_peak else '✗ FAIL'}")
print(f"  Syngas increases: {'✓ PASS' if syngas_trend else '✗ FAIL'}")
print()

print("=" * 60)