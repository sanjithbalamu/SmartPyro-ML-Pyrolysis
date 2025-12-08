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
print("SMARTPYRO: PYROLYSIS YIELD PREDICTOR")
print("=" * 60)
print()

def predict_yields(input_data):
    """
    Predict pyrolysis yields from input conditions
    
    Parameters:
    - input_data: dictionary with keys matching feature names
    """
    
    # Convert to DataFrame
    df = pd.DataFrame([input_data])
    
    # Encode biomass species if provided
    if 'Biomass species' in df.columns:
        try:
            df['Biomass_encoded'] = label_encoder.transform(df['Biomass species'])
        except:
            print("Warning: Unknown biomass species. Using average encoding.")
            df['Biomass_encoded'] = 0
    
    # Select and order features
    X = df[feature_cols].copy()
    
    # Handle missing values
    X_imputed = imputer_X.transform(X)
    
    # Scale features
    X_scaled = scaler.transform(X_imputed)
    
    # Predict each output
    predictions = {}
    for target, model in models.items():
        pred = model.predict(X_scaled)[0]
        predictions[target] = pred
    
    return predictions


# ==================================================
# EXAMPLE 1: Predict from manual input
# ==================================================

print("EXAMPLE 1: Custom Input")
print("-" * 60)

# Define your pyrolysis conditions
conditions = {
    'M': 5.7,           # Moisture %
    'Ash': 4.7,         # Ash %
    'VM': 74.9,         # Volatile matter %
    'FC': 14.3,         # Fixed carbon %
    'C': 47.3,          # Carbon %
    'H': 6.2,           # Hydrogen %
    'O': 44.7,          # Oxygen %
    'N': 1.5,           # Nitrogen %
    'PS': 0.5,          # Particle size (mm)
    'FT': 500,          # Final temperature (°C) ← KEY PARAMETER
    'HR': 20,           # Heating rate (°C/min)
    'FR': 100,          # Flow rate (mL/min)
    'Biomass species': 'rapeseed'
}

print("Input conditions:")
for key, value in conditions.items():
    print(f"  {key}: {value}")
print()

results = predict_yields(conditions)

print("Predicted yields:")
print(f"  Solid phase (biochar):  {results['Solid phase']:.2f}%")
print(f"  Liquid phase (bio-oil): {results['Liquid phase']:.2f}%")
print(f"  Gas phase (syngas):     {results['Gas phase']:.2f}%")
print(f"  Total:                  {sum(results.values()):.2f}%")
print()

# ==================================================
# EXAMPLE 2: Temperature sensitivity analysis
# ==================================================

print("=" * 60)
print("EXAMPLE 2: Temperature Optimization")
print("-" * 60)
print()

temperatures = [400, 450, 500, 550, 600, 650, 700]
results_by_temp = []

print("Testing different temperatures to maximize bio-oil yield...")
print()
print(f"{'Temp (°C)':<12} {'Biochar %':<12} {'Bio-oil %':<12} {'Syngas %':<12}")
print("-" * 60)

for temp in temperatures:
    test_conditions = conditions.copy()
    test_conditions['FT'] = temp
    
    preds = predict_yields(test_conditions)
    results_by_temp.append({
        'Temperature': temp,
        'Biochar': preds['Solid phase'],
        'Bio-oil': preds['Liquid phase'],
        'Syngas': preds['Gas phase']
    })
    
    print(f"{temp:<12} {preds['Solid phase']:<12.2f} {preds['Liquid phase']:<12.2f} {preds['Gas phase']:<12.2f}")

print()

# Find optimal temperature for bio-oil
best_result = max(results_by_temp, key=lambda x: x['Bio-oil'])
print(f"✓ Optimal temperature for bio-oil: {best_result['Temperature']}°C")
print(f"  Expected bio-oil yield: {best_result['Bio-oil']:.2f}%")
print()

# ==================================================
# EXAMPLE 3: Interactive prediction
# ==================================================

print("=" * 60)
print("INTERACTIVE PREDICTION MODE")
print("=" * 60)
print()
print("Enter your pyrolysis conditions:")
print("(Press Enter to use default value shown in brackets)")
print()

def get_input(prompt, default):
    user_input = input(prompt)
    return float(user_input) if user_input.strip() else default

try:
    user_conditions = {
        'M': get_input(f"Moisture % [{conditions['M']}]: ", conditions['M']),
        'Ash': get_input(f"Ash % [{conditions['Ash']}]: ", conditions['Ash']),
        'VM': get_input(f"Volatile Matter % [{conditions['VM']}]: ", conditions['VM']),
        'FC': get_input(f"Fixed Carbon % [{conditions['FC']}]: ", conditions['FC']),
        'C': get_input(f"Carbon % [{conditions['C']}]: ", conditions['C']),
        'H': get_input(f"Hydrogen % [{conditions['H']}]: ", conditions['H']),
        'O': get_input(f"Oxygen % [{conditions['O']}]: ", conditions['O']),
        'N': get_input(f"Nitrogen % [{conditions['N']}]: ", conditions['N']),
        'PS': get_input(f"Particle Size mm [{conditions['PS']}]: ", conditions['PS']),
        'FT': get_input(f"Final Temperature °C [{conditions['FT']}]: ", conditions['FT']),
        'HR': get_input(f"Heating Rate °C/min [{conditions['HR']}]: ", conditions['HR']),
        'FR': get_input(f"Flow Rate mL/min [{conditions['FR']}]: ", conditions['FR']),
        'Biomass_encoded': 0  # Default encoding
    }
    
    print()
    print("-" * 60)
    print("YOUR PREDICTION:")
    print("-" * 60)
    
    user_results = predict_yields(user_conditions)
    
    print(f"Solid phase (biochar):  {user_results['Solid phase']:.2f}%")
    print(f"Liquid phase (bio-oil): {user_results['Liquid phase']:.2f}%")
    print(f"Gas phase (syngas):     {user_results['Gas phase']:.2f}%")
    print(f"Total:                  {sum(user_results.values()):.2f}%")
    print()

except KeyboardInterrupt:
    print("\n\nPrediction cancelled.")
except Exception as e:
    print(f"\nSkipping interactive mode: {e}")

print("=" * 60)
print("✓ PREDICTION COMPLETE")
print("=" * 60)