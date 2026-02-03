#!/usr/bin/env python3
"""
Train Crop Yield Prediction Model
This script trains all models and saves them to saved_model/ directory
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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception as e:
    HAS_XGB = False
    print(f"⚠️ XGBoost not available: {e}")

try:    
    from lightgbm import LGBMRegressor
    import lightgbm as lgb_lib
    HAS_LGB = True
except Exception as e:
    HAS_LGB = False
    print(f"⚠️ LightGBM not available: {e}")
    
from catboost import CatBoostRegressor
import category_encoders as ce

warnings.filterwarnings('ignore')
np.random.seed(42)

print("=" * 80)
print("🌾 CROP YIELD MODEL TRAINING")
print("=" * 80)

# ========================
# 1. LOAD DATA
# ========================
print("\n📥 Loading dataset...")

# Try different possible data locations
data_paths = [
    "../dataset/asishpandey/crop-production-in-india/Final_Dataset_after_temperature.csv",
    "dataset/asishpandey/crop-production-in-india/Final_Dataset_after_temperature.csv",
    "../Final_Dataset_after_temperature.csv",
]

df = None
for path in data_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"✅ Data loaded from: {path}")
        print(f"   Shape: {df.shape}")
        break

if df is None:
    print("❌ Dataset not found. Please ensure the dataset is in one of these locations:")
    for p in data_paths:
        print(f"   - {p}")
    sys.exit(1)

# ========================
# 2. DATA CLEANING
# ========================
print("\n🧹 Cleaning data...")
print(f"   Initial shape: {df.shape}")

df = df.drop_duplicates()
df = df.dropna(subset=['Yield_ton_per_hec'])
df = df[df['Yield_ton_per_hec'] > 0]
df = df[df['Yield_ton_per_hec'] < 200]

numeric_cols = ['N', 'P', 'K', 'temperature', 'rainfall', 'pH']
for col in numeric_cols:
    if col in df.columns:
        df = df[df[col].notna()]
        df = df[(df[col] >= 0) & (df[col] < 1000)]

print(f"   After cleaning: {df.shape}")

# Per-crop outlier removal
print("   Removing per-crop outliers...")
df_clean = df.copy()
outlier_count = 0

for crop in df['Crop'].unique():
    crop_mask = df_clean['Crop'] == crop
    crop_data = df_clean[crop_mask]
    if len(crop_data) < 30:
        continue

    Q1 = crop_data['Yield_ton_per_hec'].quantile(0.03)
    Q3 = crop_data['Yield_ton_per_hec'].quantile(0.97)
    IQR = Q3 - Q1
    lower = Q1 - 2.5 * IQR
    upper = Q3 + 2.5 * IQR

    valid_mask = crop_mask & (df_clean['Yield_ton_per_hec'] >= lower) & (df_clean['Yield_ton_per_hec'] <= upper)
    removed = crop_mask & ~valid_mask
    outlier_count += removed.sum()
    df_clean = df_clean[~removed]

print(f"   Removed {outlier_count} outliers")
print(f"   Final shape: {df_clean.shape}")
df = df_clean.reset_index(drop=True)

# ========================
# 3. FEATURE ENGINEERING
# ========================
print("\n🔧 Creating features...")
df_features = df.copy()

# Nutrient features
df_features['NPK_sum'] = df_features['N'] + df_features['P'] + df_features['K']
df_features['NPK_product'] = np.log1p(df_features['N'] * df_features['P'] * df_features['K'])
df_features['NP_ratio'] = df_features['N'] / (df_features['P'] + 1)
df_features['NK_ratio'] = df_features['N'] / (df_features['K'] + 1)
df_features['PK_ratio'] = df_features['P'] / (df_features['K'] + 1)
df_features['N_dominance'] = df_features['N'] / (df_features['NPK_sum'] + 1)
df_features['P_dominance'] = df_features['P'] / (df_features['NPK_sum'] + 1)
df_features['K_dominance'] = df_features['K'] / (df_features['NPK_sum'] + 1)
df_features['nutrient_balance'] = 1 - (np.abs(df_features['N_dominance'] - 0.33) + np.abs(df_features['P_dominance'] - 0.33))
df_features['NPK_harmonic'] = 3 / ((1/(df_features['N']+1)) + (1/(df_features['P']+1)) + (1/(df_features['K']+1)))

# Environmental features
df_features['temp_rain'] = df_features['temperature'] * df_features['rainfall']
df_features['temp_pH'] = df_features['temperature'] * df_features['pH']
df_features['rain_pH'] = df_features['rainfall'] * df_features['pH']
df_features['moisture_index'] = df_features['rainfall'] / (df_features['temperature'] + 1)
df_features['heat_stress'] = np.where(df_features['temperature'] > 30, (df_features['temperature'] - 30)**2, 0)
df_features['drought_stress'] = np.where(df_features['rainfall'] < 500, (500 - df_features['rainfall'])**1.5, 0)
df_features['optimal_temp'] = np.exp(-((df_features['temperature'] - 25)**2) / 100)
df_features['optimal_rain'] = np.exp(-((df_features['rainfall'] - 800)**2) / 100000)

# Soil quality
df_features['pH_dist'] = np.abs(df_features['pH'] - 6.5)
df_features['soil_fert'] = df_features['NPK_sum'] * (1 - df_features['pH_dist'] / 7)

# Crop-level statistics
for col in ['N', 'P', 'K', 'temperature', 'rainfall', 'pH']:
    df_features[f'{col}_crop_mean'] = df_features.groupby('Crop')[col].transform('mean')
    df_features[f'{col}_deviation'] = df_features[col] - df_features[f'{col}_crop_mean']

df_features['crop_median_yield'] = df_features.groupby('Crop')['Yield_ton_per_hec'].transform('median')
df_features['crop_std_yield'] = df_features.groupby('Crop')['Yield_ton_per_hec'].transform('std').fillna(0)
df_features['crop_count'] = df_features.groupby('Crop')['Crop'].transform('count')

# Interactions
df_features['NPK_temp'] = df_features['NPK_sum'] * df_features['temperature']
df_features['NPK_rain'] = df_features['NPK_sum'] * df_features['rainfall']

# Polynomial
for col in ['N', 'P', 'K', 'temperature', 'rainfall']:
    df_features[f'{col}_sq'] = df_features[col] ** 2

# Log
for col in ['N', 'P', 'K', 'rainfall']:
    df_features[f'{col}_log'] = np.log1p(df_features[col])

# Binning
for col, bins in [('N',10), ('P',10), ('K',10), ('temperature',10), ('rainfall',10)]:
    df_features[f'{col}_bin'] = pd.cut(df_features[col], bins=bins, labels=False)

print(f"   Total features: {len(df_features.columns)}")

# ========================
# 4. DATA PREPARATION
# ========================
print("\n📊 Preparing data...")

# Filter rare crops
crop_counts = df_features['Crop'].value_counts()
valid_crops = crop_counts[crop_counts >= 5].index
df_filtered = df_features[df_features['Crop'].isin(valid_crops)].copy().reset_index(drop=True)
print(f"   Kept {len(valid_crops)} crops | {len(df_filtered):,} samples")

# Separate X / y
y = df_filtered['Yield_ton_per_hec'].values
X = df_filtered.drop(columns=['Yield_ton_per_hec'])

# Identify categoricals
categorical_cols = [c for c in ['Crop', 'Season', 'State', 'District'] if c in X.columns]
print(f"   Categorical columns: {categorical_cols}")

# Label-encode categoricals
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col + '_le'] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Train / Test split
print("\n   Splitting 80/20...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=X['Crop']
)
print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

# Target encoding (fit on train only)
print("   Applying target encoding...")
target_encoder = ce.TargetEncoder(cols=categorical_cols, smoothing=10)
X_train_te = target_encoder.fit_transform(X_train[categorical_cols], y_train)
X_test_te = target_encoder.transform(X_test[categorical_cols])

X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()
for col in categorical_cols:
    X_train_encoded[col + '_te'] = X_train_te[col].values
    X_test_encoded[col + '_te'] = X_test_te[col].values

# Keep only numeric columns
X_train_final = X_train_encoded.select_dtypes(include=[np.number]).copy()
X_test_final = X_test_encoded.select_dtypes(include=[np.number]).copy()

# Clean column names
X_train_final.columns = X_train_final.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)
X_test_final.columns = X_test_final.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)

# Scaled copy for MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)

print(f"   Final feature count: {X_train_final.shape[1]}")

# ========================
# 5. TRAIN BASE MODELS
# ========================
print("\n🤖 Training base models...")

# Early stopping validation set
X_tr, X_es, y_tr, y_es = train_test_split(
    X_train_final, y_train, test_size=0.10, random_state=42
)
X_es_scaled = scaler.transform(X_es)

# XGBoost (optional)
xgb_model = None
if HAS_XGB:
    print("   [1/6] XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=2000, max_depth=6, learning_rate=0.03,
        subsample=0.75, colsample_bytree=0.7,
        min_child_weight=5, gamma=0.2,
        reg_alpha=0.8, reg_lambda=1.5,
        random_state=42, n_jobs=-1, tree_method='hist',
        early_stopping_rounds=50, eval_metric='mae'
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
else:
    print("   [1/6] XGBoost... SKIPPED (not available)")

# LightGBM (optional)
lgb_model = None
if HAS_LGB:
    print("   [2/6] LightGBM...")
    lgb_model = LGBMRegressor(
        n_estimators=2000, max_depth=7, learning_rate=0.025,
        subsample=0.8, colsample_bytree=0.65,
        min_child_samples=30, reg_alpha=0.5, reg_lambda=2.0,
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_model.fit(X_tr, y_tr,
                  eval_set=[(X_es, y_es)],
                  callbacks=[lgb_lib.callback.early_stopping(50, verbose=False),
                             lgb_lib.callback.log_evaluation(0)])
else:
    print("   [2/6] LightGBM... SKIPPED (not available)")

# CatBoost
print("   [3/6] CatBoost...")
cat_model = CatBoostRegressor(
    iterations=2000, learning_rate=0.03, depth=5,
    l2_leaf_reg=5, random_seed=42,
    early_stopping_rounds=50, verbose=0
)
cat_model.fit(X_tr, y_tr, eval_set=(X_es, y_es))

# RandomForest
print("   [4/6] RandomForest...")
rf_model = RandomForestRegressor(
    n_estimators=500, max_depth=18, min_samples_split=5,
    min_samples_leaf=3, max_features='sqrt',
    random_state=42, n_jobs=-1
)
rf_model.fit(X_train_final, y_train)

# GradientBoosting
print("   [5/6] GradientBoosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=500, learning_rate=0.03, max_depth=5,
    min_samples_split=8, min_samples_leaf=4,
    subsample=0.75, random_state=42
)
gb_model.fit(X_train_final, y_train)

# MLP
print("   [6/6] MLP...")
mlp_model = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation='relu', alpha=0.01,
    learning_rate='adaptive', learning_rate_init=0.001,
    max_iter=800, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=30, random_state=42
)
mlp_model.fit(X_train_scaled, y_train)

print("✅ All base models trained!")

# ========================
# 6. STACKING META-LEARNER
# ========================
print("\n🏗️  Training stacking meta-learner...")

n_models = 5 if HAS_XGB else 5  # Always 5 models (without xgb if not available)
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros((len(X_train_final), n_models))
test_preds_folds = np.zeros((len(X_test_final), n_models, n_folds))

model_configs = []

if HAS_XGB:
    model_configs.append(
        ('XGB', lambda: XGBRegressor(n_estimators=2000, max_depth=6, learning_rate=0.03,
                                      subsample=0.75, colsample_bytree=0.7,
                                      min_child_weight=5, gamma=0.2,
                                      reg_alpha=0.8, reg_lambda=1.5,
                                      random_state=42, n_jobs=-1, tree_method='hist',
                                      early_stopping_rounds=50, eval_metric='mae'), 'xgb')
    )

if HAS_LGB:
    model_configs.append(
        ('LGB', lambda: LGBMRegressor(n_estimators=2000, max_depth=7, learning_rate=0.025,
                                       subsample=0.8, colsample_bytree=0.65,
                                       min_child_samples=30, reg_alpha=0.5, reg_lambda=2.0,
                                       random_state=42, n_jobs=-1, verbose=-1), 'lgb')
    )

model_configs.extend([
    ('CAT', lambda: CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=5,
                                       l2_leaf_reg=5, random_seed=42,
                                       early_stopping_rounds=50, verbose=0), 'cat'),
    ('RF',  lambda: RandomForestRegressor(n_estimators=500, max_depth=18,
                                           min_samples_split=5, min_samples_leaf=3,
                                           max_features='sqrt', random_state=42, n_jobs=-1), 'rf'),
    ('GB',  lambda: GradientBoostingRegressor(n_estimators=500, learning_rate=0.03,
                                              max_depth=5, min_samples_split=8,
                                              min_samples_leaf=4, subsample=0.75,
                                              random_state=42), 'gb'),
    ('MLP', lambda: MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                                  alpha=0.01, learning_rate='adaptive',
                                  learning_rate_init=0.001, max_iter=800,
                                  early_stopping=True, validation_fraction=0.1,
                                  n_iter_no_change=30, random_state=42), 'mlp'),
])

n_models = len(model_configs)
print(f"   Training with {n_models} models: {[name for name, _, _ in model_configs]}")

oof_preds = np.zeros((len(X_train_final), n_models))
test_preds_folds = np.zeros((len(X_test_final), n_models, n_folds))

for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train_final)):
    print(f"   Fold {fold_idx+1}/{n_folds}...", flush=True)
    X_tr_f = X_train_final.iloc[tr_idx]
    X_val_f = X_train_final.iloc[val_idx]
    y_tr_f = y_train[tr_idx]

    es_split = int(len(X_tr_f) * 0.9)
    X_tr_es = X_tr_f.iloc[:es_split]; y_tr_es = y_tr_f[:es_split]
    X_es_f = X_tr_f.iloc[es_split:]; y_es_f = y_tr_f[es_split:]

    scaler_f = StandardScaler()
    X_tr_sc = scaler_f.fit_transform(X_tr_f)
    X_val_sc = scaler_f.transform(X_val_f)
    X_test_sc = scaler_f.transform(X_test_final)

    for m_idx, (name, make_model, kind) in enumerate(model_configs):
        model = make_model()

        if kind == 'xgb':
            model.fit(X_tr_es, y_tr_es, eval_set=[(X_es_f, y_es_f)], verbose=False)
        elif kind == 'lgb':
            model.fit(X_tr_es, y_tr_es, eval_set=[(X_es_f, y_es_f)],
                      callbacks=[lgb_lib.callback.early_stopping(50, verbose=False),
                                 lgb_lib.callback.log_evaluation(0)])
        elif kind == 'cat':
            model.fit(X_tr_es, y_tr_es, eval_set=(X_es_f, y_es_f))
        elif kind in ('rf', 'gb'):
            model.fit(X_tr_f, y_tr_f)
        elif kind == 'mlp':
            model.fit(X_tr_sc, y_tr_f)

        if kind == 'mlp':
            oof_preds[val_idx, m_idx] = model.predict(X_val_sc)
            test_preds_folds[:, m_idx, fold_idx] = model.predict(X_test_sc)
        else:
            oof_preds[val_idx, m_idx] = model.predict(X_val_f)
            test_preds_folds[:, m_idx, fold_idx] = model.predict(X_test_final)

test_preds_base = test_preds_folds.mean(axis=2)

# Fit Ridge meta-learner
print("\n   Fitting Ridge meta-learner...")
meta_model = Ridge(alpha=1.0, fit_intercept=True)
meta_model.fit(oof_preds, y_train)

print("✅ Stacking complete!")

# ========================
# 7. EVALUATE
# ========================
print("\n📊 Evaluating on test set...")
y_pred_final = meta_model.predict(test_preds_base)
y_pred_final = np.clip(y_pred_final, 0, None)

r2 = r2_score(y_test, y_pred_final)
mae = mean_absolute_error(y_test, y_pred_final)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))

pct_err = np.abs((y_test - y_pred_final) / y_test) * 100
median_err = np.median(pct_err)

w10 = (pct_err <= 10).mean() * 100
w20 = (pct_err <= 20).mean() * 100

print(f"\n   R²              = {r2:.4f}")
print(f"   MAE             = {mae:.3f} ton/ha")
print(f"   RMSE            = {rmse:.3f} ton/ha")
print(f"   Median % Error  = {median_err:.2f}%")
print(f"   Within 10%      = {w10:.1f}%")
print(f"   Within 20%      = {w20:.1f}%")

# ========================
# 8. SAVE MODELS
# ========================
print("\n💾 Saving models...")
save_dir = "saved_model"
os.makedirs(save_dir, exist_ok=True)

# Retrain models on full training set
print("   Retraining on full train set...")

deploy_models = {}

if HAS_XGB and xgb_model is not None:
    xgb_deploy = XGBRegressor(
        n_estimators=xgb_model.best_iteration, max_depth=6, learning_rate=0.03,
        subsample=0.75, colsample_bytree=0.7, min_child_weight=5, gamma=0.2,
        reg_alpha=0.8, reg_lambda=1.5, random_state=42, n_jobs=-1, tree_method='hist'
    )
    xgb_deploy.fit(X_train_final, y_train)
    deploy_models['xgb'] = xgb_deploy

if HAS_LGB and lgb_model is not None:
    lgb_deploy = LGBMRegressor(
        n_estimators=lgb_model.best_iteration_, max_depth=7, learning_rate=0.025,
        subsample=0.8, colsample_bytree=0.65, min_child_samples=30,
        reg_alpha=0.5, reg_lambda=2.0, random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_deploy.fit(X_train_final, y_train)
    deploy_models['lgb'] = lgb_deploy

cat_deploy = CatBoostRegressor(
    iterations=cat_model.best_iteration_, learning_rate=0.03, depth=5,
    l2_leaf_reg=5, random_seed=42, verbose=0
)
cat_deploy.fit(X_train_final, y_train)

deploy_models.update({
    'cat': cat_deploy,
    'rf': rf_model,
    'gb': gb_model,
    'mlp': mlp_model,
})

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

# Get valid values from dataset
valid_seasons = list(df_filtered["Season"].dropna().unique()) if "Season" in df_filtered.columns else ["Kharif", "Rabi", "Whole Year", "Summer", "Winter", "Autumn"]
valid_states = list(df_filtered["State"].dropna().unique()) if "State" in df_filtered.columns else []

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

total_kb = sum(os.path.getsize(os.path.join(save_dir, fn)) for fn in os.listdir(save_dir)) / 1024
print(f"\n✅ All models saved to {save_dir}/")
print(f"   Total size: {total_kb:.1f} KB")
print(f"   Files: {len(os.listdir(save_dir))}")

print("\n🎉 Training complete! You can now run main.py to use the models.")
