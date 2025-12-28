# 🌾 Crop Yield Prediction using Machine Learning

## Overview

This project predicts crop production quantities using Machine Learning. It automatically downloads the Crop Production in India dataset from Kaggle, trains multiple regression models, and provides a web interface for predictions.

## Features

- **Automated Data Pipeline**: Downloads dataset from Kaggle using kagglehub
- **Data Preprocessing**: Handles missing values, scales numerical features, and encodes categorical features
- **Model Training**: Trains and compares 6 regression models
- **Web Interface**: Flask-based web app for easy predictions
- **Production Ready**: Calculates both total production and yield per hectare

## Project Structure

```
├── app.py                          # Flask web application
├── requirements.txt                # Python dependencies
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # Downloads data from Kaggle
│   │   ├── data_transformation.py # Preprocessing pipeline
│   │   └── model_trainer.py       # Model training & evaluation
│   ├── pipeline/
│   │   ├── train_pipeline.py      # Training workflow
│   │   └── prediction_pipeline.py # Prediction workflow
│   ├── exception.py               # Custom exception handling
│   ├── logger.py                  # Logging configuration
│   └── utils.py                   # Utility functions
├── templates/
│   └── index.html                 # Web interface
└── dataset/                       # Generated after training
    ├── model.pkl                  # Trained model
    └── preprocessor.pkl           # Preprocessing pipeline
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd Crop_prediction_ml_pipeline
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Step 1: Train the Model

Run the training pipeline to download the dataset and train models:

```bash
python -m src.pipeline.train_pipeline
```

This will:

- Download the dataset from Kaggle automatically
- Preprocess the data
- Train 6 regression models
- Save the best model and preprocessor to `dataset/`

**Note**: First-time execution will download ~7MB of data from Kaggle.

#### Step 2: Run the Web Application

Start the Flask server:

```bash
python app.py
```

Open your browser and navigate to:

```
http://localhost:5000
```

### Making Predictions

The web interface requires the following inputs:

- **N, P, K**: Nutrient levels (Nitrogen, Phosphorus, Potassium)
- **pH**: Soil pH level
- **Rainfall**: Annual rainfall in mm
- **Temperature**: Average temperature in °C
- **Area in hectares**: Cultivation area
- **State Name**: Indian state
- **Crop Type**: kharif, rabi, whole year, or summer
- **Crop**: Specific crop name

The model returns:

- **Predicted Crop Production** (in tons)
- **Predicted Yield** (tons per hectare)

## Data Pipeline

**Data Source**: The [Crop Production in India dataset](https://www.kaggle.com/datasets/asishpandey/crop-production-in-india) is automatically downloaded from Kaggle using `kagglehub`.

**Features**:

- Nutrient levels: N, P, K
- pH level
- Rainfall (mm)
- Temperature (°C)
- Area in hectares
- State Name
- Crop Type (kharif/rabi/summer/whole year)
- Crop name
- Target: Production in tons

## Data Transformation

**Preprocessing Pipeline**:

1. **Imputation**

   - Numerical features: Median imputation
   - Categorical features: Most frequent value

2. **Encoding**

   - Categorical features: OrdinalEncoder with predefined categories
   - 33 states, 4 crop types, 54 crop varieties

3. **Scaling**
   - StandardScaler applied to all features

## Model Training

**Models Evaluated**:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet Regression
5. Decision Tree Regressor
6. Random Forest Regressor

**Evaluation Metric**: R² Score

The best-performing model is automatically selected and saved.

## Requirements

```
numpy==1.26.4
pandas==2.2.2
flask==3.0.3
scikit-learn==1.4.2
seaborn==0.13.2
datetime==5.5
kagglehub
```

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'kagglehub'`

- **Solution**: Run `pip install -r requirements.txt`

**Issue**: Model/preprocessor not found

- **Solution**: Run the training pipeline first: `python -m src.pipeline.train_pipeline`

**Issue**: Port 5000 already in use

- **Solution**: Change the port in `app.py`: `app.run(debug=True, port=8080)`

## Dataset Credit

Dataset: [Crop Production in India](https://www.kaggle.com/datasets/asishpandey/crop-production-in-india) by Asish Pandey on Kaggle
