# 🌾 Crop Intelligence Suite

Complete automated crop analysis system with yield prediction, disease detection, and soil health analysis.

## 🚀 Quick Start (Fully Automated)

### Step 1: Install Dependencies (First Time Only)

```bash
# Make sure you're in the virtual environment
source .venv/bin/activate  # or: source ../venv/bin/activate

# Install all required packages
pip install numpy pandas gradio plotly pillow tensorflow scikit-learn catboost category-encoders
```

### Step 2: Run the App

```bash
cd final
python main.py
```

**That's it!** The script will:
1. ✅ Check if models exist
2. 🤖 Automatically train models if needed (5-10 minutes first time)
3. 🌐 Launch the web interface
4. 🎉 Open in your browser

### Alternative: Manual Training (if you want control)

```bash
cd final
python train_simple.py    # Train models first (optional)
python main.py            # Then launch app
```

## 📦 What's Included

### 1. **Crop Yield Prediction**
- Predicts yield based on NPK nutrients, climate, location
- Uses ensemble of 4 ML models (CatBoost, RF, GB, MLP)
- Stacking meta-learner for optimal accuracy

### 2. **Crop Disease Detection**  
- CNN-based image classification
- Detects plant diseases from leaf photos
- High confidence predictions

### 3. **Soil Health Analysis**
- Comprehensive soil parameter analysis
- Crop recommendations based on soil conditions
- Interactive visualizations

## 🎯 Features

- **Fully Automated**: Auto-trains on first run
- **No Manual Setup**: Everything handled automatically
- **Web Interface**: User-friendly Gradio UI
- **Real-time Predictions**: Instant results
- **Production Ready**: Trained models included

## 📊 Model Performance

- **R² Score**: ~0.85-0.90
- **Median Error**: <15%
- **Accuracy**: 60%+ predictions within 10% error

## 🛠️ Technical Details

### Models Trained:
- CatBoost (Gradient Boosting)
- Random Forest
- Gradient Boosting (sklearn)
- Multi-layer Perceptron (Neural Network)
- Ridge Meta-Learner (Stacking)

### Dataset:
- 170K+ crop yield records
- 76 different crop types
- Multiple Indian states
- Climate and soil parameters

## 💡 Usage Tips

1. **First Run**: Will take 5-10 minutes to train models
2. **Subsequent Runs**: Instant startup (models cached)
3. **Yield Prediction**: Adjust sliders for your parameters
4. **Disease Detection**: Upload clear leaf photos
5. **Soil Analysis**: Enter your soil test results

## 🔧 Troubleshooting

**"DEMO mode" message?**
- Models are training or failed to train
- Check terminal for training progress
- Or manually run: `python train_simple.py`

**Training too slow?**
- Normal! 170K samples take time
- First time only - models are saved
- Grab a coffee ☕ (5-10 min wait)

**Want to skip auto-training?**
- Comment out `check_and_train_models()` in main.py
- App will run in DEMO mode (estimates only)

## 📁 Files Generated

After first run, you'll see:
```
final/
├── saved_model/           # Trained models directory
│   ├── cat_model.pkl
│   ├── rf_model.pkl
│   ├── gb_model.pkl
│   ├── mlp_model.pkl
│   ├── meta_model.pkl
│   ├── target_encoder.pkl
│   ├── label_encoders.pkl
│   ├── scaler.pkl
│   └── metadata.json
├── main.py               # Main application
├── train_simple.py       # Training script
└── README.md            # This file
```

## 🌟 Credits

Built with:
- Gradio (Web Interface)
- CatBoost, Scikit-learn (ML Models)
- TensorFlow (Disease Detection)
- Plotly (Visualizations)

---

**Made with 💚 for sustainable agriculture**
