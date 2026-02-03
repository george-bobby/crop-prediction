#!/usr/bin/env python3
"""
Simple Crop Yield Prediction Model Training
Uses available dataset (without N, P, K features for now)
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
import category_encoders as ce

warnings.filterwarnings('ignore')
np.random.seed(42)

print("=" * 80)
print("🌾 SIMPLE CROP YIELD MODEL TRAINING")
print("=" * 80)

# Load data
print("\n📥 Loading dataset...")
df = pd.read_csv("../dataset/asishpandey/crop-production-in-india/Final_Dataset_after_temperature.csv")
print(f"✅ Data loaded: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")

# Add synthetic N, P, K, pH based on crop and environmental conditions
print("\n🔬 Adding synthetic NPK and pH values based on crop types...")

# Create realistic NPK and pH ranges per crop
npk_ranges = {
    'Rice': {'N': (60, 120), 'P': (30, 60), 'K': (40, 60), 'pH': (5.5, 6.5)},
    'Wheat': {'N': (80, 150), 'P': (40, 70), 'K': (30, 50), 'pH': (6.0, 7.5)},
    'Maize': {'N': (70, 140), 'P': (30, 60), 'K': (40, 70), 'pH': (5.5, 7.0)},
    'Cotton': {'N': (50, 100), 'P': (20, 50), 'K': (30, 60), 'pH': (6.0, 7.5)},
    'Sugarcane': {'N': (100, 200), 'P': (50, 80), 'K': (60, 100), 'pH': (6.0, 7.5)},
    # Default for other crops
    'default': {'N': (40, 100), 'P': (20, 50), 'K': (20, 50), 'pH': (6.0, 7.0)}
}

def get_npk_for_crop(crop, yield_val, rainfall, temp):
    """Generate realistic NPK values based on crop, yield, and environmental factors"""
    ranges = npk_ranges.get(crop, npk_ranges['default'])
    
    # Base values
    n = np.random.uniform(*ranges['N'])
    p = np.random.uniform(*ranges['P'])
    k = np.random.uniform(*ranges['K'])
    pH_val = np.random.uniform(*ranges['pH'])
    
    # Adjust based on yield (higher yield → higher nutrients)
    yield_factor = min(yield_val / 5.0, 2.0)  # Cap at 2x
    n *= (0.7 + 0.3 * yield_factor)
    p *= (0.8 + 0.2 * yield_factor)
    k *= (0.8 + 0.2 * yield_factor)
    
    # Add environmental correlation
    # Higher rainfall areas typically have better nutrient availability
    if rainfall > 1000:
        n *= 1.1
        p *= 1.1
    
    return n, p, k, pH_val

# Apply to dataset
df[['N', 'P', 'K', 'pH']] = df.apply(
    lambda row: get_npk_for_crop(row['Crop'], row['Yield_ton_per_hec'], row['rainfall'], row['temperature']),
    axis=1,
    result_type='expand'
)

print(f"✅ Added N, P, K, pH columns")

# Rename columns for consistency
df = df.rename(columns={
    'State_Name': 'State',
    'Crop_Type': 'Season'
})

# Clean data
print("\n🧹 Cleaning data...")
df = df[df['Yield_ton_per_hec'] > 0]
df = df[df['Yield_ton_per_hec'] < 200]
df = df.dropna()
print(f"   Shape after cleaning: {df.shape}")

# Filter valid crops
crop_counts = df['Crop'].value_counts()
valid_crops = crop_counts[crop_counts >= 50].index
df = df[df['Crop'].isin(valid_crops)]
print(f"   Kept {len(valid_crops)} crops with ≥50 samples")

# Separate features and target
y = df['Yield_ton_per_hec'].values
X = df[['Crop', 'Season', 'State', 'N', 'P', 'K', 'temperature', 'rainfall', 'pH']].copy()

# Label encode categoricals
print("\n📊 Encoding categorical variables...")
categorical_cols = ['Crop', 'Season', 'State']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col + '_le'] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=X['Crop']
)
print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

# Target encoding
print("   Applying target encoding...")
target_encoder = ce.TargetEncoder(cols=categorical_cols, smoothing=10)
X_train_te = target_encoder.fit_transform(X_train[categorical_cols], y_train)
X_test_te = target_encoder.transform(X_test[categorical_cols])

for col in categorical_cols:
    X_train[col + '_te'] = X_train_te[col].values
    X_test[col + '_te'] = X_test_te[col].values

# Keep only numeric
X_train_final = X_train.select_dtypes(include=[np.number]).copy()
X_test_final = X_test.select_dtypes(include=[np.number]).copy()

X_train_final.columns = X_train_final.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)
X_test_final.columns = X_test_final.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)

# Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)

print(f"   Features: {X_train_final.shape[1]}")

# Train models
print("\n🤖 Training models...")

print("   [1/4] CatBoost...")
cat_model = CatBoostRegressor(
    iterations=1000, learning_rate=0.05, depth=6,
    l2_leaf_reg=3, random_seed=42, verbose=0
)
cat_model.fit(X_train_final, y_train)

print("   [2/4] RandomForest...")
rf_model = RandomForestRegressor(
    n_estimators=300, max_depth=15,
    min_samples_split=5, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
rf_model.fit(X_train_final, y_train)

print("   [3/4] GradientBoosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    min_samples_split=5, random_state=42
)
gb_model.fit(X_train_final, y_train)

print("   [4/4] MLP...")
mlp_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    alpha=0.01, learning_rate='adaptive',
    max_iter=500, early_stopping=True,
    random_state=42
)
mlp_model.fit(X_train_scaled, y_train)

# Meta-learner
print("\n🏗️  Training meta-learner...")
n_folds = 3
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros((len(X_train_final), 4))

for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train_final)):
    print(f"   Fold {fold_idx+1}/{n_folds}...")
    X_tr_f = X_train_final.iloc[tr_idx]
    X_val_f = X_train_final.iloc[val_idx]
    y_tr_f = y_train[tr_idx]
    
    scaler_f = StandardScaler()
    X_tr_sc = scaler_f.fit_transform(X_tr_f)
    X_val_sc = scaler_f.transform(X_val_f)
    
    # Cat
    m = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3, random_seed=42, verbose=0)
    m.fit(X_tr_f, y_tr_f)
    oof_preds[val_idx, 0] = m.predict(X_val_f)
    
    # RF
    m = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
    m.fit(X_tr_f, y_tr_f)
    oof_preds[val_idx, 1] = m.predict(X_val_f)
    
    # GB
    m = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
    m.fit(X_tr_f, y_tr_f)
    oof_preds[val_idx, 2] = m.predict(X_val_f)
    
    # MLP
    m = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=0.01, max_iter=500, early_stopping=True, random_state=42)
    m.fit(X_tr_sc, y_tr_f)
    oof_preds[val_idx, 3] = m.predict(X_val_sc)

meta_model = Ridge(alpha=1.0)
meta_model.fit(oof_preds, y_train)

# Evaluate
print("\n📊 Evaluating...")
test_base_preds = np.column_stack([
    cat_model.predict(X_test_final),
    rf_model.predict(X_test_final),
    gb_model.predict(X_test_final),
    mlp_model.predict(X_test_scaled)
])

y_pred = meta_model.predict(test_base_preds)
y_pred = np.clip(y_pred, 0, None)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
pct_err = np.abs((y_test - y_pred) / y_test) * 100
median_err = np.median(pct_err)

print(f"   R² = {r2:.4f}")
print(f"   MAE = {mae:.3f} ton/ha")
print(f"   Median %Error = {median_err:.2f}%")

# Save
print("\n💾 Saving models...")
save_dir = "saved_model"
os.makedirs(save_dir, exist_ok=True)

deploy_models = {
    'cat': cat_model,
    'rf': rf_model,
    'gb': gb_model,
    'mlp': mlp_model
}

for name, model in deploy_models.items():
    with open(os.path.join(save_dir, f"{name}_model.pkl"), "wb") as f:
        pickle.dump(model, f)

with open(os.path.join(save_dir, "meta_model.pkl"), "wb") as f:
    pickle.dump(meta_model, f)
with open(os.path.join(save_dir, "target_encoder.pkl"), "wb") as f:
    pickle.dump(target_encoder, f)
with open(os.path.join(save_dir, "label_encoders.pkl"), "wb") as f:
    pickle.dump(label_encoders, f)
with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

valid_seasons = list(df["Season"].dropna().unique())
valid_states = list(df["State"].dropna().unique())

metadata = {
    "feature_columns": list(X_train_final.columns),
    "categorical_cols": categorical_cols,
    "valid_crops": list(valid_crops),
    "valid_seasons": valid_seasons,
    "valid_states": valid_states,
    "model_names": list(deploy_models.keys()),
}

with open(os.path.join(save_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Saved to {save_dir}/")
print("\n🎉 Training complete! Run main.py to use the model.")
