import pandas as pd
import numpy as np
import joblib


models = joblib.load('models_improved.pkl')
scaler = joblib.load('scaler_improved.pkl')
imputer = joblib.load('imputer_improved.pkl')
feature_cols = joblib.load('feature_cols_improved.pkl')

print("=" * 60)
print("SMARTPYRO: PYROLYSIS YIELD PREDICTOR")
print("Physics-Constrained ML Model")
print("=" * 60)
print()

def predict_yields(conditions):
    """
    Predict pyrolysis yields from process conditions
    
    Returns: dict with Solid phase, Liquid phase, Gas phase (all sum to 100%)
    """
    
    df = pd.DataFrame([conditions])[feature_cols]
    
    
    X_imputed = imputer.transform(df)
    X_scaled = scaler.transform(X_imputed)
    
    
    predictions = {}
    for target, model in models.items():
        predictions[target] = model.predict(X_scaled)[0]
    
    
    total = sum(predictions.values())
    predictions_normalized = {k: (v/total)*100 for k, v in predictions.items()}
    
    return predictions_normalized




print("EXAMPLE 1: Single Prediction")
print("-" * 60)

conditions = {
    'M': 6.0,           
    'Ash': 5.0,         
    'VM': 75.0,         
    'FC': 14.0,         
    'C': 47.0,          
    'H': 6.2,           
    'O': 45.0,          
    'N': 1.5,           
    'PS': 0.5,          
    'FT': 550,          
    'HR': 20.0,         
    'FR': 100.0         
}

results = predict_yields(conditions)

print("Input conditions:")
print(f"  Temperature: {conditions['FT']}°C")
print(f"  Heating rate: {conditions['HR']}°C/min")
print(f"  Particle size: {conditions['PS']} mm")
print()

print("Predicted yields:")
print(f"  Biochar (solid):  {results['Solid phase']:.2f}%")
print(f"  Bio-oil (liquid): {results['Liquid phase']:.2f}%")
print(f"  Syngas (gas):     {results['Gas phase']:.2f}%")
print(f"  Total:            {sum(results.values()):.2f}%")
print()



print("=" * 60)
print("EXAMPLE 2: Find Optimal Temperature for Bio-oil")
print("-" * 60)
print()

base_conditions = conditions.copy()
temperatures = range(400, 751, 50)

best_temp = None
best_biooil = 0

print(f"{'Temp (°C)':<12} {'Biochar %':<12} {'Bio-oil %':<12} {'Syngas %':<12}")
print("-" * 60)

for temp in temperatures:
    test = base_conditions.copy()
    test['FT'] = temp
    preds = predict_yields(test)
    
    print(f"{temp:<12} {preds['Solid phase']:<12.2f} {preds['Liquid phase']:<12.2f} {preds['Gas phase']:<12.2f}")
    
    if preds['Liquid phase'] > best_biooil:
        best_biooil = preds['Liquid phase']
        best_temp = temp

print()
print(f"✓ Optimal temperature: {best_temp}°C")
print(f"  Expected bio-oil yield: {best_biooil:.2f}%")
print()



print("=" * 60)
print("INTERACTIVE PREDICTION")
print("=" * 60)
print()
print("Enter your conditions (press Enter for defaults):")
print()

def get_input(prompt, default):
    try:
        user_input = input(prompt)
        return float(user_input) if user_input.strip() else default
    except:
        return default

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
        'FT': get_input(f"Temperature °C [{conditions['FT']}]: ", conditions['FT']),
        'HR': get_input(f"Heating Rate °C/min [{conditions['HR']}]: ", conditions['HR']),
        'FR': get_input(f"Flow Rate mL/min [{conditions['FR']}]: ", conditions['FR'])
    }
    
    print()
    print("-" * 60)
    print("YOUR PREDICTION:")
    print("-" * 60)
    
    user_results = predict_yields(user_conditions)
    
    print(f"Biochar (solid):  {user_results['Solid phase']:.2f}%")
    print(f"Bio-oil (liquid): {user_results['Liquid phase']:.2f}%")
    print(f"Syngas (gas):     {user_results['Gas phase']:.2f}%")
    print()
    
except KeyboardInterrupt:
    print("\n\nCancelled.")
except Exception as e:
    print(f"\nInteractive mode skipped.")

print("=" * 60)
print("✓ DONE")
print("=" * 60)