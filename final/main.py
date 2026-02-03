#!/usr/bin/env python3
"""
🌾 Crop Intelligence Suite
Combined Crop Yield, Crop Disease Detection, and Soil Health in one Gradio interface.
No retraining required — this script loads existing artifacts if available.
"""

import os
import sys
import json
import pickle
import warnings

# Add parent directory to path to access models folder
parent_dir = os.path.dirname(os.path.abspath(os.getcwd()))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go

from PIL import Image
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge

try:
    from xgboost import XGBRegressor
except Exception as e:
    XGBRegressor = None
    print(f"⚠️ xgboost unavailable: {e}")
try:
    from lightgbm import LGBMRegressor
except Exception as e:
    LGBMRegressor = None
    print(f"⚠️ lightgbm unavailable: {e}")
try:
    from catboost import CatBoostRegressor
except Exception as e:
    CatBoostRegressor = None
    print(f"⚠️ catboost unavailable: {e}")
try:
    import category_encoders as ce
except Exception as e:
    ce = None
    print(f"⚠️ category_encoders unavailable: {e}")

warnings.filterwarnings("ignore")

WORKSPACE_DIR = os.path.abspath(os.getcwd())
print(f"✅ Workspace: {WORKSPACE_DIR}")

# =========================
# AUTO-TRAIN IF NEEDED
# =========================
def check_and_train_models():
    """Check if models exist, if not, train them automatically"""
    save_dir_candidates = [
        os.path.join(WORKSPACE_DIR, "saved_model"),
        os.path.join(WORKSPACE_DIR, "final", "saved_model"),
        os.path.join(os.path.dirname(WORKSPACE_DIR), "saved_model"),
    ]
    
    # Check if any saved_model directory exists with models
    model_exists = False
    for save_dir in save_dir_candidates:
        if os.path.exists(save_dir) and os.path.exists(os.path.join(save_dir, "metadata.json")):
            model_exists = True
            break
    
    if not model_exists:
        print("\n" + "="*80)
        print("⚠️  NO TRAINED MODELS FOUND")
        print("="*80)
        print("Starting automatic model training...")
        print("This will take 5-10 minutes. Please wait...\n")
        
        # Run training script
        train_script = os.path.join(WORKSPACE_DIR, "train_simple.py")
        if os.path.exists(train_script):
            import subprocess
            result = subprocess.run([sys.executable, train_script], 
                                  capture_output=False, text=True)
            if result.returncode == 0:
                print("\n✅ Model training completed successfully!")
            else:
                print("\n⚠️ Model training encountered issues but continuing...")
        else:
            print(f"⚠️ Training script not found: {train_script}")
            print("   Continuing in DEMO mode...")

# Auto-train if needed (comment out this line to skip auto-training)
check_and_train_models()

warnings.filterwarnings("ignore")


# =========================
# 1) CROP DISEASE DETECTOR
# =========================

SKIP_MODEL_LOAD = os.environ.get("SKIP_MODEL_LOAD", "0") == "1"
DISEASE_MODEL_CANDIDATES = [
    os.path.join(WORKSPACE_DIR, "plant_disease_detector.keras"),
    os.path.join(WORKSPACE_DIR, "final_plant_disease_model.h5"),
    os.path.join(WORKSPACE_DIR, "final", "plant_disease_detector.keras"),
    os.path.join(WORKSPACE_DIR, "final", "final_plant_disease_model.h5"),
    os.path.join(WORKSPACE_DIR, "models", "disease_detection_model.h5"),
    os.path.join(os.path.dirname(WORKSPACE_DIR), "models", "disease_detection_model.h5"),
]

CLASS_INDEX_CANDIDATES = [
    os.path.join(WORKSPACE_DIR, "class_indices.json"),
    os.path.join(os.path.dirname(WORKSPACE_DIR), "class_indices.json"),
]

DISEASE_MODEL = None
DISEASE_CLASS_NAMES = None
DISEASE_MODEL_PATH = None
DISEASE_LOADED = False

if not SKIP_MODEL_LOAD:
    for path in DISEASE_MODEL_CANDIDATES:
        if os.path.exists(path):
            DISEASE_MODEL_PATH = path
            break
else:
    DISEASE_MODEL_PATH = None

if DISEASE_MODEL_PATH:
    DISEASE_MODEL = tf.keras.models.load_model(DISEASE_MODEL_PATH)
    DISEASE_LOADED = True

    for cpath in CLASS_INDEX_CANDIDATES:
        if os.path.exists(cpath):
            with open(cpath, "r") as f:
                class_indices = json.load(f)
            # class_indices maps class_name -> index
            DISEASE_CLASS_NAMES = {v: k for k, v in class_indices.items()}

    print(f"✅ Disease model loaded: {DISEASE_MODEL_PATH}")
    if DISEASE_CLASS_NAMES:
        print(f"✅ Class indices loaded: {len(DISEASE_CLASS_NAMES)} classes")
    else:
        print("⚠️ class_indices.json not found — labels will be shown as class indices")
else:
    print("⚠️ Disease model not found. Place model file and class_indices.json in the workspace.")


def predict_top_disease(image):
    """Return the most confident disease prediction"""
    if not DISEASE_LOADED:
        return "❌ Disease model not loaded. Add model files and restart the kernel."

    if image is None:
        return "❌ Please upload an image first."

    try:
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0

        if len(img_array.shape) == 2:  # grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]

        img_array = np.expand_dims(img_array, axis=0)
        predictions = DISEASE_MODEL.predict(img_array, verbose=0)[0]

        top_idx = int(np.argmax(predictions))
        confidence = float(predictions[top_idx])
        disease_name = DISEASE_CLASS_NAMES.get(top_idx, f"Class #{top_idx}") if DISEASE_CLASS_NAMES else f"Class #{top_idx}"

        # Parse disease name
        if "___" in disease_name:
            plant, condition = disease_name.split("___", 1)
            condition = condition.replace("_", " ")
        else:
            plant, condition = "Plant", disease_name

        is_healthy = "healthy" in disease_name.lower()

        if is_healthy:
            result = f"""
# 🟢 **HEALTHY PLANT DETECTED**

## **{plant}**

📊 **Confidence:** {confidence:.1%}

✅ **Status:** No disease detected. Plant appears healthy.
"""
        else:
            result = f"""
# 🔴 **DISEASE DETECTED**

## **{plant}**
### **{condition}**

📊 **Confidence:** {confidence:.1%}

⚠️ **Status:** Disease detected. Immediate attention recommended.
"""

        if not DISEASE_CLASS_NAMES:
            result += "\n\n⚠️ Class labels are missing (class_indices.json not found)."

        return result

    except Exception as e:
        return f"❌ Error: {str(e)}"


# =========================
# 2) CROP YIELD PREDICTOR
# =========================

SKIP_MODEL_LOAD = os.environ.get("SKIP_MODEL_LOAD", "0") == "1"
SAVE_DIR_CANDIDATES = [
    os.path.join(WORKSPACE_DIR, "saved_model"),
    os.path.join(WORKSPACE_DIR, "final", "saved_model"),
    os.path.join(os.path.dirname(WORKSPACE_DIR), "saved_model"),
    os.path.join(os.path.dirname(WORKSPACE_DIR), "models"),
]

SAVE_DIR = None if SKIP_MODEL_LOAD else next((p for p in SAVE_DIR_CANDIDATES if os.path.exists(p)), None)

DEFAULT_META = {
    "valid_crops": [
        "Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Jute", "Bajra",
        "Jowar", "Ragi", "Groundnut", "Soybean", "Sunflower", "Sesame",
        "Arhar/Tur", "Gram", "Masoor", "Moong", "Urad", "Rapeseed & Mustard"
    ],
    "valid_seasons": ["Kharif", "Rabi", "Whole Year", "Summer", "Winter", "Autumn"],
    "valid_states": [
        "Andhra Pradesh", "Karnataka", "Kerala", "Tamil Nadu", "Maharashtra",
        "Gujarat", "Rajasthan", "Punjab", "Haryana", "Uttar Pradesh",
        "Madhya Pradesh", "West Bengal", "Bihar", "Odisha"
    ],
    "model_names": ["xgb", "lgb", "cat", "rf", "gb", "mlp"],
}

META = DEFAULT_META.copy()
FEATURE_COLS = []
CAT_COLS = ["Crop", "Season", "State"]

YIELD_MODEL_LOADED = False

if SAVE_DIR and os.path.exists(os.path.join(SAVE_DIR, "metadata.json")):
    try:
        with open(os.path.join(SAVE_DIR, "metadata.json"), "r") as f:
            loaded_meta = json.load(f)

        for key in ["valid_crops", "valid_seasons", "valid_states", "model_names"]:
            if key in loaded_meta and loaded_meta[key]:
                META[key] = loaded_meta[key]

        with open(os.path.join(SAVE_DIR, "target_encoder.pkl"), "rb") as f:
            TE = pickle.load(f)
        with open(os.path.join(SAVE_DIR, "label_encoders.pkl"), "rb") as f:
            LE = pickle.load(f)
        with open(os.path.join(SAVE_DIR, "scaler.pkl"), "rb") as f:
            SCALER = pickle.load(f)
        with open(os.path.join(SAVE_DIR, "meta_model.pkl"), "rb") as f:
            META_MODEL = pickle.load(f)

        BASE_MODELS = {}
        for name in META["model_names"]:
            with open(os.path.join(SAVE_DIR, f"{name}_model.pkl"), "rb") as f:
                BASE_MODELS[name] = pickle.load(f)

        FEATURE_COLS = loaded_meta.get("feature_columns", [])
        CAT_COLS = [col for col in loaded_meta.get("categorical_cols", CAT_COLS) if col != "District"]

        YIELD_MODEL_LOADED = True
        print(f"✅ Yield model loaded from: {SAVE_DIR}")
    except Exception as e:
        print(f"⚠️ Yield model load failed: {e}")
else:
    print("⚠️ Yield saved_model not found. Running in DEMO mode.")


def create_features(df):
    d = df.copy()

    # Nutrient features
    d["NPK_sum"] = d["N"] + d["P"] + d["K"]
    d["NPK_product"] = np.log1p(d["N"] * d["P"] * d["K"])
    d["NP_ratio"] = d["N"] / (d["P"] + 1)
    d["NK_ratio"] = d["N"] / (d["K"] + 1)
    d["PK_ratio"] = d["P"] / (d["K"] + 1)
    d["N_dominance"] = d["N"] / (d["NPK_sum"] + 1)
    d["P_dominance"] = d["P"] / (d["NPK_sum"] + 1)
    d["K_dominance"] = d["K"] / (d["NPK_sum"] + 1)
    d["nutrient_balance"] = 1 - (np.abs(d["N_dominance"] - 0.33) + np.abs(d["P_dominance"] - 0.33))
    d["NPK_harmonic"] = 3 / ((1/(d["N"]+1)) + (1/(d["P"]+1)) + (1/(d["K"]+1)))

    # Climate features
    d["temp_rain"] = d["temperature"] * d["rainfall"]
    d["temp_pH"] = d["temperature"] * d["pH"]
    d["rain_pH"] = d["rainfall"] * d["pH"]
    d["moisture_index"] = d["rainfall"] / (d["temperature"] + 1)
    d["heat_stress"] = np.where(d["temperature"] > 30, (d["temperature"] - 30)**2, 0)
    d["drought_stress"] = np.where(d["rainfall"] < 500, (500 - d["rainfall"])**1.5, 0)
    d["optimal_temp"] = np.exp(-((d["temperature"] - 25)**2) / 100)
    d["optimal_rain"] = np.exp(-((d["rainfall"] - 800)**2) / 100000)

    # Soil features
    d["pH_dist"] = np.abs(d["pH"] - 6.5)
    d["soil_fert"] = d["NPK_sum"] * (1 - d["pH_dist"] / 7)

    # Crop aggregates placeholders
    for col in ["N", "P", "K", "temperature", "rainfall", "pH"]:
        d[f"{col}_crop_mean"] = d[col]
        d[f"{col}_deviation"] = 0.0
    d["crop_median_yield"] = 0.0
    d["crop_std_yield"] = 0.0
    d["crop_count"] = 100

    # Interactions
    d["NPK_temp"] = d["NPK_sum"] * d["temperature"]
    d["NPK_rain"] = d["NPK_sum"] * d["rainfall"]

    # Polynomial
    for col in ["N", "P", "K", "temperature", "rainfall"]:
        d[f"{col}_sq"] = d[col] ** 2

    # Log
    for col in ["N", "P", "K", "rainfall"]:
        d[f"{col}_log"] = np.log1p(d[col])

    # Binning
    for col, bins in [("N",10), ("P",10), ("K",10), ("temperature",10), ("rainfall",10)]:
        val = d[col].iloc[0]
        d[f"{col}_bin"] = min(int(val / (val + 1) * bins), bins - 1)

    return d


def predict_real(crop, season, state, N, P, K, temp, rain, pH):
    try:
        row = pd.DataFrame([{
            "Crop": crop,
            "Season": season,
            "State": state,
            "N": float(N),
            "P": float(P),
            "K": float(K),
            "temperature": float(temp),
            "rainfall": float(rain),
            "pH": float(pH)
        }])

        d = create_features(row)

        # Label encode
        for col in CAT_COLS:
            if col in LE:
                val = str(d[col].iloc[0])
                if val not in set(LE[col].classes_):
                    d[col] = LE[col].classes_[0]
                d[col + "_le"] = LE[col].transform(d[col].astype(str))

        # Target encode
        te_out = TE.transform(d[CAT_COLS])
        for col in CAT_COLS:
            d[col + "_te"] = te_out[col].values

        d_num = d.select_dtypes(include=[np.number]).copy()
        d_num.columns = d_num.columns.str.replace("[^A-Za-z0-9_]", "_", regex=True)
        d_num = d_num.reindex(columns=FEATURE_COLS, fill_value=0)

        d_scaled = SCALER.transform(d_num)
        base_preds = np.zeros((1, len(META["model_names"])))

        for i, name in enumerate(META["model_names"]):
            model = BASE_MODELS[name]
            if name == "mlp":
                base_preds[0, i] = model.predict(d_scaled)[0]
            else:
                base_preds[0, i] = model.predict(d_num)[0]

        base_preds_clipped = np.clip(base_preds, 0, None)

        try:
            meta_pred = META_MODEL.predict(base_preds_clipped)[0]
            if meta_pred > 0.1:
                final = float(meta_pred)
                method = "Stacked (Meta-Model)"
            else:
                positive_preds = base_preds_clipped[0][base_preds_clipped[0] > 0]
                if len(positive_preds) > 0:
                    final = float(np.median(positive_preds))
                    method = "Median (Fallback)"
                else:
                    final = 0.5
                    method = "Default (All predictions negative)"
        except Exception:
            positive_preds = base_preds_clipped[0][base_preds_clipped[0] > 0]
            if len(positive_preds) > 0:
                final = float(np.median(positive_preds))
                method = "Median (Error Fallback)"
            else:
                final = 0.5
                method = "Default (Prediction Error)"

        breakdown = "🤖 Individual Model Predictions:\n\n"
        for i, name in enumerate(META["model_names"]):
            pred_val = base_preds[0, i]
            clipped_val = base_preds_clipped[0, i]
            if pred_val != clipped_val:
                breakdown += f"  {name.upper():>4s}: {pred_val:>7.3f} → {clipped_val:>7.3f} ton/ha (clipped)\n"
            else:
                breakdown += f"  {name.upper():>4s}: {pred_val:>7.3f} ton/ha\n"

        breakdown += f"\n📊 Final ({method}): {final:.3f} ton/ha"
        breakdown += "\n\n✅ Real prediction using trained model"

        return round(final, 3), breakdown

    except Exception as e:
        return 0.0, f"❌ Prediction Error: {str(e)}"


def predict_demo(crop, season, state, N, P, K, temp, rain, pH):
    np.random.seed(hash(f"{crop}{season}{state}") % 2**32)

    base = (N * 0.02 + P * 0.015 + K * 0.018) / 3
    temp_f = 1 - abs(temp - 25) / 50
    rain_f = min(rain / 1000, 1.5)
    pH_f = 1 - abs(pH - 6.5) / 5

    mult = {
        "Rice": 3.5, "Wheat": 3.0, "Maize": 4.0, "Cotton": 1.5,
        "Sugarcane": 70, "Jute": 2.0, "Bajra": 1.8, "Jowar": 2.2,
        "Ragi": 2.5, "Groundnut": 1.2, "Soybean": 2.0
    }.get(crop, 2.5)

    final = base * temp_f * rain_f * pH_f * mult * np.random.uniform(0.85, 1.15)

    breakdown = "🤖 Individual Model Predictions (DEMO):\n\n"
    for name in ["XGB", "LGB", "CAT", "RF", "GB", "MLP"]:
        pred = final * np.random.uniform(0.9, 1.1)
        breakdown += f"  {name:>4s}: {pred:>7.3f} ton/ha\n"
    breakdown += f"\n📊 Final (Stacked): {final:.3f} ton/ha"
    breakdown += "\n\n⚠️ DEMO MODE - Load model for real predictions"

    return round(final, 3), breakdown


predict_yield = predict_real if YIELD_MODEL_LOADED else predict_demo


# =========================
# 3) SOIL HEALTH ANALYSIS
# =========================

def analyze_soil_health(Clay, OM, CEC, pH, V, exP, exK, exCa, exMg):
    """Comprehensive soil health analysis with ratings"""

    scores = {}
    ratings = {}
    recommendations = []

    # pH
    if 6.0 <= pH <= 7.0:
        scores['pH'] = 100
        ratings['pH'] = "✅ Excellent"
        recommendations.append("pH is in optimal range for most crops")
    elif 5.5 <= pH < 6.0 or 7.0 < pH <= 7.5:
        scores['pH'] = 70
        ratings['pH'] = "⚠️ Moderate"
        if pH < 6.0:
            recommendations.append("Soil is slightly acidic. Consider small lime application")
        else:
            recommendations.append("Soil is slightly alkaline. May need sulfur amendment")
    else:
        scores['pH'] = 30
        ratings['pH'] = "❌ Poor"
        if pH < 5.5:
            recommendations.append("Strongly acidic soil. Lime application required for most crops")
        else:
            recommendations.append("Strongly alkaline soil. Sulfur or gypsum needed")

    # Organic matter
    if OM >= 30:
        scores['OM'] = 100
        ratings['OM'] = "✅ Excellent"
        recommendations.append("High organic matter - good for soil structure and nutrients")
    elif 20 <= OM < 30:
        scores['OM'] = 70
        ratings['OM'] = "⚠️ Moderate"
        recommendations.append("Organic matter is adequate but could be improved with compost")
    else:
        scores['OM'] = 40
        ratings['OM'] = "❌ Low"
        recommendations.append("Low organic matter. Add compost, manure, or green manure crops")

    # Nutrients
    nutrient_status = []

    if exP >= 20:
        scores['P'] = 100
        ratings['P'] = "✅ Sufficient"
    elif 10 <= exP < 20:
        scores['P'] = 65
        ratings['P'] = "⚠️ Marginal"
        nutrient_status.append("Phosphorus")
    else:
        scores['P'] = 30
        ratings['P'] = "❌ Deficient"
        nutrient_status.append("Phosphorus")

    if exK >= 2.5:
        scores['K'] = 100
        ratings['K'] = "✅ Sufficient"
    elif 1.5 <= exK < 2.5:
        scores['K'] = 65
        ratings['K'] = "⚠️ Marginal"
        nutrient_status.append("Potassium")
    else:
        scores['K'] = 30
        ratings['K'] = "❌ Deficient"
        nutrient_status.append("Potassium")

    if exCa >= 40:
        scores['Ca'] = 100
        ratings['Ca'] = "✅ Sufficient"
    elif 20 <= exCa < 40:
        scores['Ca'] = 65
        ratings['Ca'] = "⚠️ Marginal"
        nutrient_status.append("Calcium")
    else:
        scores['Ca'] = 30
        ratings['Ca'] = "❌ Deficient"
        nutrient_status.append("Calcium")

    if exMg >= 20:
        scores['Mg'] = 100
        ratings['Mg'] = "✅ Sufficient"
    elif 10 <= exMg < 20:
        scores['Mg'] = 65
        ratings['Mg'] = "⚠️ Marginal"
    else:
        scores['Mg'] = 30
        ratings['Mg'] = "❌ Deficient"

    # CEC
    if CEC >= 80:
        scores['CEC'] = 100
        ratings['CEC'] = "✅ High"
        recommendations.append("High CEC - good nutrient holding capacity")
    elif 50 <= CEC < 80:
        scores['CEC'] = 75
        ratings['CEC'] = "⚠️ Moderate"
    else:
        scores['CEC'] = 40
        ratings['CEC'] = "❌ Low"
        recommendations.append("Low CEC - may need frequent fertilization")

    # Base saturation
    if V >= 80:
        scores['V'] = 100
        ratings['V'] = "✅ High"
    elif 60 <= V < 80:
        scores['V'] = 70
        ratings['V'] = "⚠️ Adequate"
    else:
        scores['V'] = 40
        ratings['V'] = "❌ Low"
        recommendations.append("Low base saturation - soil may be acidic")

    weights = {
        'pH': 0.25,
        'OM': 0.20,
        'P': 0.15,
        'K': 0.15,
        'Ca': 0.10,
        'CEC': 0.10,
        'V': 0.05
    }

    overall_score = sum(scores[param] * weights.get(param, 0) for param in weights)

    if overall_score >= 80:
        health_rating = "Excellent"
        health_color = "green"
    elif overall_score >= 60:
        health_rating = "Good"
        health_color = "orange"
    elif overall_score >= 40:
        health_rating = "Fair"
        health_color = "yellow"
    else:
        health_rating = "Poor"
        health_color = "red"

    return {
        'overall_score': round(overall_score, 1),
        'health_rating': health_rating,
        'health_color': health_color,
        'ratings': ratings,
        'scores': scores,
        'recommendations': recommendations,
        'nutrient_deficiencies': nutrient_status
    }


def get_crop_recommendations(soil_summary, soil_params):
    crops_database = {
        'Rice': {'min_pH': 4.5, 'max_pH': 6.5, 'min_OM': 15, 'min_P': 10, 'min_K': 1.5, 'category': 'Staple'},
        'Potato': {'min_pH': 4.8, 'max_pH': 5.5, 'min_OM': 20, 'min_P': 25, 'min_K': 3.0, 'category': 'Vegetable'},
        'Tea': {'min_pH': 4.5, 'max_pH': 5.5, 'min_OM': 25, 'min_P': 15, 'min_K': 2.0, 'category': 'Cash'},
        'Coffee': {'min_pH': 4.5, 'max_pH': 6.0, 'min_OM': 30, 'min_P': 15, 'min_K': 2.5, 'category': 'Cash'},
        'Pineapple': {'min_pH': 4.5, 'max_pH': 5.5, 'min_OM': 20, 'min_P': 15, 'min_K': 2.0, 'category': 'Fruit'},
        'Maize': {'min_pH': 5.5, 'max_pH': 7.0, 'min_OM': 20, 'min_P': 15, 'min_K': 2.0, 'category': 'Staple'},
        'Tomato': {'min_pH': 5.5, 'max_pH': 6.8, 'min_OM': 25, 'min_P': 20, 'min_K': 2.5, 'min_Ca': 30, 'category': 'Vegetable'},
        'Wheat': {'min_pH': 6.0, 'max_pH': 7.5, 'min_OM': 25, 'min_P': 20, 'min_K': 2.5, 'min_Ca': 30, 'category': 'Staple'},
        'Soybean': {'min_pH': 6.0, 'max_pH': 7.0, 'min_OM': 20, 'min_P': 15, 'min_K': 2.0, 'category': 'Legume'},
        'Sugarcane': {'min_pH': 5.5, 'max_pH': 7.5, 'min_OM': 30, 'min_P': 25, 'min_K': 3.0, 'category': 'Cash'},
        'Banana': {'min_pH': 5.5, 'max_pH': 7.0, 'min_OM': 25, 'min_P': 20, 'min_K': 3.0, 'category': 'Fruit'},
        'Groundnut': {'min_pH': 5.5, 'max_pH': 6.5, 'min_OM': 15, 'min_P': 12, 'min_K': 1.5, 'category': 'Legume'},
        'Castor': {'min_pH': 5.5, 'max_pH': 7.0, 'min_OM': 15, 'min_P': 10, 'min_K': 1.5, 'category': 'Oilseed'},
    }

    recommendations = []

    for crop, requirements in crops_database.items():
        suitability = 100

        if not (requirements['min_pH'] <= soil_params['pH'] <= requirements['max_pH']):
            suitability -= 40
        if soil_params['OM'] < requirements['min_OM']:
            suitability -= 30
        if 'min_P' in requirements and soil_params['exP'] < requirements['min_P']:
            suitability -= 20
        if 'min_K' in requirements and soil_params['exK'] < requirements['min_K']:
            suitability -= 15
        if 'min_Ca' in requirements and soil_params['exCa'] < requirements['min_Ca']:
            suitability -= 10

        if suitability >= 50:
            recommendations.append({
                'Crop': crop,
                'Suitability': suitability,
                'Category': requirements['category'],
                'Rating': 'Excellent' if suitability >= 85 else
                         'Good' if suitability >= 70 else
                         'Fair' if suitability >= 60 else 'Marginal'
            })

    recommendations.sort(key=lambda x: x['Suitability'], reverse=True)
    return recommendations[:10]


def create_soil_health_radar(soil_summary):
    params = ['pH', 'OM', 'P', 'K', 'Ca', 'CEC']
    values = [soil_summary['scores'].get(param, 50) for param in params]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=params,
        fill='toself',
        name='Soil Health',
        line_color='blue',
        fillcolor='rgba(135, 206, 250, 0.5)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=11))
        ),
        showlegend=False,
        height=350,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig


def create_nutrient_bars(soil_params):
    nutrients = ['P (mg/kg)', 'K (mg/kg)', 'Ca (mg/kg)', 'Mg (mg/kg)']
    values = [soil_params['exP'], soil_params['exK'], soil_params['exCa'], soil_params['exMg']]

    optimal_min = [15, 2.0, 30, 15]
    optimal_max = [40, 4.0, 60, 30]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=nutrients,
        y=values,
        name='Current Level',
        marker_color=['#FF6B6B' if v < optimal_min[i] else '#4ECDC4' if v <= optimal_max[i] else '#FFD166' for i, v in enumerate(values)]
    ))

    for i, nutrient in enumerate(nutrients):
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=i-0.4,
            x1=i+0.4,
            y0=optimal_min[i],
            y1=optimal_max[i],
            fillcolor="rgba(78, 205, 196, 0.2)",
            line_width=0,
            layer="below"
        )

    fig.update_layout(
        title="Nutrient Levels vs Optimal Range",
        yaxis_title="Concentration",
        height=300,
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig


def analyze_and_recommend(Clay, OM, CEC, pH, V, exP, exK, exCa, exMg):
    soil_params = {
        'Clay': Clay, 'OM': OM, 'CEC': CEC, 'pH': pH, 'V': V,
        'exP': exP, 'exK': exK, 'exCa': exCa, 'exMg': exMg
    }

    soil_summary = analyze_soil_health(Clay, OM, CEC, pH, V, exP, exK, exCa, exMg)
    crop_recommendations = get_crop_recommendations(soil_summary, soil_params)

    health_html = f"""
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid {soil_summary['health_color']};'>
        <h2 style='color: {soil_summary['health_color']}; margin-top: 0;'>🌱 Soil Health: {soil_summary['health_rating']}</h2>
        <div style='font-size: 24px; font-weight: bold; color: {soil_summary['health_color']};'>
            Overall Score: {soil_summary['overall_score']}/100
        </div>
    </div>

    <h3>📊 Parameter Analysis</h3>
    <table style='width: 100%; border-collapse: collapse;'>
        <tr style='background-color: #e9ecef;'>
            <th style='padding: 10px; text-align: left;'>Parameter</th>
            <th style='padding: 10px; text-align: left;'>Rating</th>
            <th style='padding: 10px; text-align: left;'>Score</th>
        </tr>
    """

    for param, rating in soil_summary['ratings'].items():
        if param in soil_summary['scores']:
            health_html += f"""
            <tr style='border-bottom: 1px solid #dee2e6;'>
                <td style='padding: 10px;'><strong>{param}</strong></td>
                <td style='padding: 10px;'>{rating}</td>
                <td style='padding: 10px;'>{soil_summary['scores'][param]}/100</td>
            </tr>
            """

    health_html += "</table>"

    health_html += """
    <h3>💡 Recommendations</h3>
    <div style='background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107;'>
    """

    for rec in soil_summary['recommendations'][:5]:
        health_html += f"<p>• {rec}</p>"

    if soil_summary['nutrient_deficiencies']:
        health_html += f"<p><strong>Nutrient Deficiencies:</strong> {', '.join(soil_summary['nutrient_deficiencies'])}</p>"

    health_html += "</div>"

    crops_html = "<h2>🌾 Recommended Crops</h2>"

    if crop_recommendations:
        crops_html += """
        <table style='width: 100%; border-collapse: collapse; margin-top: 20px;'>
            <tr style='background-color: #28a745; color: white;'>
                <th style='padding: 12px; text-align: left;'>Rank</th>
                <th style='padding: 12px; text-align: left;'>Crop</th>
                <th style='padding: 12px; text-align: left;'>Category</th>
                <th style='padding: 12px; text-align: left;'>Suitability</th>
                <th style='padding: 12px; text-align: left;'>Rating</th>
            </tr>
        """

        for i, crop in enumerate(crop_recommendations[:8], 1):
            color_map = {
                'Excellent': '#28a745',
                'Good': '#ffc107',
                'Fair': '#fd7e14',
                'Marginal': '#dc3545'
            }

            crops_html += f"""
            <tr style='border-bottom: 1px solid #dee2e6;'>
                <td style='padding: 10px;'>{i}</td>
                <td style='padding: 10px; font-weight: bold;'>{crop['Crop']}</td>
                <td style='padding: 10px;'>{crop['Category']}</td>
                <td style='padding: 10px;'>{crop['Suitability']}/100</td>
                <td style='padding: 10px; color: {color_map.get(crop['Rating'], '#000')}; font-weight: bold;'>{crop['Rating']}</td>
            </tr>
            """

        crops_html += "</table>"
    else:
        crops_html += """
        <div style='padding: 20px; background-color: #f8d7da; border-radius: 5px; border-left: 4px solid #dc3545;'>
            <p><strong>⚠️ No suitable crops found.</strong> Your soil may require significant amendments before planting.</p>
            <p>Consider addressing the issues identified in the soil health analysis first.</p>
        </div>
        """

    return (
        health_html,
        crops_html,
        create_soil_health_radar(soil_summary),
        create_nutrient_bars(soil_params)
    )


# =========================
# 4) UNIFIED GRADIO APP
# =========================

def create_app():
    with gr.Blocks(title="🌾 Crop Intelligence Suite", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🌾 Crop Intelligence Suite")
        gr.Markdown("Unified interface for **Crop Yield**, **Crop Disease Detection**, and **Soil Health**.")

        with gr.Tabs():
            # -----------------------
            # Crop Yield Tab
            # -----------------------
            with gr.TabItem("🌾 Crop Yield"):
                if YIELD_MODEL_LOADED:
                    gr.Markdown("✅ **Yield model loaded** — real predictions enabled.")
                else:
                    gr.Markdown("⚠️ **Yield DEMO mode** — add saved_model/ for real predictions.")

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 🏷️ Categorical Information")

                        with gr.Row():
                            crop = gr.Dropdown(
                                label="Crop Type",
                                choices=sorted(META["valid_crops"]),
                                value=META["valid_crops"][0] if META["valid_crops"] else "Rice"
                            )
                            season = gr.Dropdown(
                                label="Growing Season",
                                choices=sorted(META["valid_seasons"]),
                                value=META["valid_seasons"][0] if META["valid_seasons"] else "Kharif"
                            )

                        state = gr.Dropdown(
                            label="State",
                            choices=sorted(META["valid_states"]),
                            value=META["valid_states"][0] if META["valid_states"] else "West Bengal"
                        )

                        gr.Markdown("---")
                        gr.Markdown("### 🧪 Soil Nutrients (NPK)")

                        with gr.Row():
                            N = gr.Slider(label="Nitrogen (N) - kg/ha", minimum=0, maximum=150, value=50, step=1)
                            P = gr.Slider(label="Phosphorus (P) - kg/ha", minimum=0, maximum=150, value=40, step=1)
                            K = gr.Slider(label="Potassium (K) - kg/ha", minimum=0, maximum=150, value=45, step=1)

                        gr.Markdown("---")
                        gr.Markdown("### 🌡️ Climate Conditions")

                        with gr.Row():
                            temp = gr.Slider(label="Temperature (°C)", minimum=5, maximum=50, value=25, step=0.5)
                            pH_val = gr.Slider(label="Soil pH", minimum=3.5, maximum=9.5, value=6.5, step=0.1)

                        rain = gr.Slider(label="Rainfall (mm)", minimum=0, maximum=3000, value=800, step=10)

                        gr.Markdown("---")
                        yield_btn = gr.Button("🌱 Predict Crop Yield", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Prediction Results")
                        yield_out = gr.Number(label="Predicted Yield (ton/ha)", precision=3)
                        breakdown_out = gr.Textbox(label="Model Breakdown", lines=14)

                        gr.Markdown("""
                        ---
                        ### 💡 Optimal Ranges
                        - **N**: 40-80 kg/ha
                        - **P**: 30-60 kg/ha
                        - **K**: 30-60 kg/ha
                        - **Temp**: 20-30°C
                        - **Rain**: 600-1200mm
                        - **pH**: 6.0-7.0
                        """)

                yield_btn.click(
                    fn=predict_yield,
                    inputs=[crop, season, state, N, P, K, temp, rain, pH_val],
                    outputs=[yield_out, breakdown_out]
                )

                gr.Markdown("---")
                gr.Markdown("### 🎯 Quick Start Examples")
                gr.Examples(
                    examples=[
                        ["Rice", "Kharif", "West Bengal", 80, 40, 45, 28, 1200, 6.2],
                        ["Wheat", "Rabi", "Punjab", 90, 55, 60, 22, 500, 7.0],
                        ["Maize", "Kharif", "Karnataka", 70, 35, 50, 30, 900, 6.8],
                        ["Cotton", "Kharif", "Maharashtra", 60, 30, 40, 32, 700, 6.5],
                        ["Sugarcane", "Whole Year", "Uttar Pradesh", 100, 60, 70, 28, 1500, 6.8],
                    ],
                    inputs=[crop, season, state, N, P, K, temp, rain, pH_val],
                    outputs=[yield_out, breakdown_out],
                    fn=predict_yield
                )

            # -----------------------
            # Crop Disease Tab
            # -----------------------
            with gr.TabItem("🍃 Crop Disease"):
                if DISEASE_LOADED:
                    gr.Markdown("✅ **Disease model loaded** — predictions enabled.")
                else:
                    gr.Markdown("⚠️ **Disease model not found** — add model files and restart the kernel.")

                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="📷 Upload Leaf Image",
                            type="pil",
                            height=300
                        )

                        analyze_btn = gr.Button(
                            "🔍 Analyze Disease",
                            variant="primary",
                            size="lg"
                        )

                        gr.Markdown("""
                        **📝 Tips for best results:**
                        1. Clear photo of single leaf
                        2. Good lighting
                        3. Avoid blurry images
                        """)

                    with gr.Column(scale=2):
                        result_output = gr.Markdown(
                            label="Detection Result",
                            value="### 👈 Upload an image to get started"
                        )

                analyze_btn.click(
                    fn=predict_top_disease,
                    inputs=image_input,
                    outputs=result_output
                )

                image_input.change(
                    fn=predict_top_disease,
                    inputs=image_input,
                    outputs=result_output
                )

            # -----------------------
            # Soil Health Tab
            # -----------------------
            with gr.TabItem("🧪 Soil Health"):
                gr.Markdown("### 🌱 Soil Health Analysis & Crop Recommendation")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📝 Soil Parameters")

                        with gr.Group():
                            gr.Markdown("**Physical Properties**")
                            Clay = gr.Slider(100, 600, value=350, step=10, label="Clay Content (g/kg)")
                            OM = gr.Slider(10, 50, value=25, step=1, label="Organic Matter (g/kg)")
                            CEC = gr.Slider(30, 150, value=80, step=5, label="CEC (cmol₊/kg)")

                        with gr.Group():
                            gr.Markdown("**Chemical Properties**")
                            pH = gr.Slider(4.0, 8.0, value=5.5, step=0.1, label="pH (4.0 = Acidic, 7.0 = Neutral)")
                            V = gr.Slider(20, 100, value=65, step=5, label="Base Saturation V (%)")

                        with gr.Group():
                            gr.Markdown("**Nutrient Levels**")
                            with gr.Row():
                                exP = gr.Slider(5, 100, value=20, step=5, label="Available P (mg/kg)")
                                exK = gr.Slider(0.5, 15, value=3.0, step=0.5, label="Available K (mg/kg)")
                            with gr.Row():
                                exCa = gr.Slider(10, 100, value=35, step=5, label="Available Ca (mg/kg)")
                                exMg = gr.Slider(5, 60, value=18, step=2, label="Available Mg (mg/kg)")

                        gr.Markdown("### 📋 Example Soil Profiles")
                        gr.Examples(
                            examples=[
                                [406, 32, 63, 5.3, 76, 15, 4.1, 31, 13],
                                [250, 35, 95, 6.5, 85, 35, 4.5, 50, 25],
                                [400, 15, 45, 4.8, 50, 8, 1.2, 15, 8]
                            ],
                            inputs=[Clay, OM, CEC, pH, V, exP, exK, exCa, exMg],
                            label="Click to load examples"
                        )

                        soil_btn = gr.Button("🔍 Analyze Soil & Get Recommendations", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Analysis Results")
                        with gr.Tabs():
                            with gr.TabItem("1️⃣ Soil Health Analysis"):
                                soil_health_html = gr.HTML(label="Soil Health Report")
                            with gr.TabItem("2️⃣ Recommended Crops"):
                                crop_recommendations_html = gr.HTML(label="Crop Recommendations")
                            with gr.TabItem("📈 Visualizations"):
                                with gr.Row():
                                    radar_plot = gr.Plot(label="Soil Health Radar Chart")
                                    nutrient_plot = gr.Plot(label="Nutrient Levels")

                soil_btn.click(
                    fn=analyze_and_recommend,
                    inputs=[Clay, OM, CEC, pH, V, exP, exK, exCa, exMg],
                    outputs=[soil_health_html, crop_recommendations_html, radar_plot, nutrient_plot]
                )

        gr.Markdown("---")
        gr.Markdown("Made with 💚 — unified app (no retraining required)")

    return app


if __name__ == "__main__":
    print("🚀 Launching Crop Intelligence Suite...")
    app = create_app()
    app.launch(share=False, debug=True, inbrowser=True)
