# SmartPyro: ML-Driven Pyrolysis Yield Prediction

Machine learning model for predicting pyrolysis product yields (biochar, bio-oil, syngas) from process conditions.

## Model Performance
- Biochar: R² = 0.637, MAE = 2.90%
- Bio-oil: R² = 0.796, MAE = 3.56%
- Syngas: R² = 0.806, MAE = 2.83%

## Quick Start

### Installation
```bash
pip install xgboost pandas numpy scikit-learn matplotlib seaborn joblib
```

### Run Predictions
```bash
python predict_final.py
```

## Key Features
- Physics-constrained XGBoost models
- Mass balance enforcement (outputs sum to 100%)
- Temperature optimization
- 619 training samples, 75 biomass species

## Dataset
Source: Kaggle "Biomass Pyrolysis Data"

## Author
Sanjith Balamurali