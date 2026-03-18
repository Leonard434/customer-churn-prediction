# Customer Churn Prediction

A machine learning project that predicts customer churn for a bank using a Random Forest classifier. The project includes a complete ML pipeline for data ingestion, preprocessing, feature engineering, model training, and evaluation, along with a Flask web API and interactive web interface for making predictions.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Running the Full ML Pipeline](#running-the-full-ml-pipeline)
  - [Starting the Flask API](#starting-the-flask-api)
  - [Making Predictions](#making-predictions)
- [Project Components](#project-components)
- [Data](#data)
- [Model Details](#model-details)
- [API Endpoints](#api-endpoints)
- [Web Interface](#web-interface)
- [Results](#results)
- [Directory Structure](#directory-structure)

## Overview

This project implements an end-to-end machine learning solution for predicting customer churn in a banking context. It processes customer data through a sophisticated preprocessing and feature engineering pipeline, trains a Random Forest classifier with cross-validation, and exposes predictions through both a REST API and an interactive web dashboard.

**Key Capabilities:**
- Single customer predictions via REST API
- Batch predictions for multiple customers
- Interactive web interface for manual predictions
- Complete ML pipeline with cross-validation
- Production-ready model artifact serialization

## Project Structure

```
customer_churn_prediction/
├── src/                          # Source code for ML pipeline
│   ├── config.py                # Configuration and settings
│   ├── data_ingestion.py         # Data loading utilities
│   ├── data_preprocessing.py     # Data cleaning and preprocessing
│   ├── feature_engineering.py    # Feature creation and engineering
│   ├── train_model.py            # Model training with cross-validation
│   ├── model_evaluation.py       # Model evaluation metrics
│   ├── predict.py                # Prediction utilities
│   └── main.py                   # Main pipeline orchestration
├── api/
│   └── app.py                    # Flask web API
├── models/
│   └── churn_model.pkl           # Trained model (serialized)
├── data/
│   ├── raw/                      # Raw data files
│   │   └── Bank Customer Churn Prediction.csv
│   └── processed/                # Processed/cleaned data
│       └── cleaned_churn.csv
├── notebooks/
│   └── eda.ipynb                 # Exploratory Data Analysis
├── templates/
│   └── index.html                # Web interface UI
├── static/                       # Static assets (CSS, JS, images)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Features

### ML Pipeline
- **Data Ingestion:** Abstract base classes for flexible data loading
- **Preprocessing:** Handling missing values, outliers, and data validation
- **Feature Engineering:**
  - Creation of new derived features (e.g., balance-to-salary ratio)
  - Automatic age group categorization
  - Engagement score calculation
- **Model Training:** Random Forest with hyperparameter tuning and cross-validation
- **Evaluation:** Comprehensive metrics including accuracy, precision, recall, F1-score, and ROC-AUC

### API & Web Interface
- **REST API:** JSON-based predictions for single customers and batch processing
- **Web UI:** Interactive form with real-time validation and visual feedback
- **Risk Levels:** Predictions categorized as Low, Medium, or High churn risk
- **Health Check:** API health monitoring endpoint

## Requirements

Python 3.8+

### Dependencies
- **Data Processing:** pandas, numpy
- **ML & Preprocessing:** scikit-learn
- **Visualization:** matplotlib, seaborn
- **Web Framework:** Flask
- **Server:** gunicorn
- **Model Serialization:** joblib
- **Notebooks:** jupyterlab

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd customer_churn_prediction
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Run the Complete ML Pipeline
Train the model and evaluate it end-to-end:

```bash
python src/main.py
```

This will:
1. Ingest data from the raw CSV file
2. Preprocess and clean the data
3. Engineer features
4. Split into train/test sets
5. Train the Random Forest model with cross-validation
6. Evaluate on the test set
7. Save the model to `models/churn_model.pkl`

### Start the Flask API
Launch the web API server:

```bash
python api/app.py
```

The API will be accessible at `http://localhost:5000`

Access the interactive web interface at:
```
http://localhost:5000/
```

## Usage

### Running the Full ML Pipeline

The main pipeline orchestrates all steps:

```bash
python src/main.py
```

**Pipeline Steps:**
1. Data Ingestion → Load CSV file
2. Data Preprocessing → Clean and validate
3. Feature Engineering → Create derived features
4. Train-Test Split → 80/20 split
5. Model Training → Train with cross-validation
6. Model Evaluation → Calculate metrics

Logs will show progress for each step.

### Starting the Flask API

```bash
python api/app.py
```

**Default:** Runs on `http://localhost:5000` in debug mode

**Production deployment:**
```bash
gunicorn -w 4 -b 0.0.0.0:5000 api.app:app
```

### Making Predictions

#### Option 1: Interactive Web Interface
Navigate to `http://localhost:5000/` in your browser and fill out the form with customer details.

#### Option 2: REST API - Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "credit_score": 650,
    "age": 35,
    "tenure": 3,
    "balance": 80000,
    "products_number": 2,
    "estimated_salary": 55000,
    "country": "France",
    "gender": "Male",
    "credit_card": 1,
    "active_member": 1
  }'
```

**Response:**
```json
{
  "status": "success",
  "prediction": {
    "churn_probability": 0.25,
    "risk_level": "Low",
    "recommendation": "Schedule routine check-in in 90 days"
  }
}
```

#### Option 3: REST API - Batch Predictions
```bash
curl -X POST http://localhost:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {"credit_score": 650, "age": 35, ...},
      {"credit_score": 720, "age": 45, ...}
    ]
  }'
```

#### Option 4: Python Script
```python
import sys, os
sys.path.insert(0, 'src')
from predict import predict_single_customer

customer_data = {
    "credit_score": 650,
    "age": 35,
    "tenure": 3,
    "balance": 80000,
    "products_number": 2,
    "estimated_salary": 55000,
    "country": "France",
    "gender": "Male",
    "credit_card": 1,
    "active_member": 1
}

result = predict_single_customer(customer_data)
print(result)
```

## Project Components

### `src/config.py`
Central configuration file containing:
- File paths for data and models
- Model hyperparameters (Random Forest settings)
- Feature names (numerical and categorical)
- Cross-validation fold count
- Train-test split ratio

### `src/data_ingestion.py`
Handles data loading with abstract base class pattern:
- `DataIngestion` (ABC) - Interface definition
- `CSVDataIngestion` - Concrete implementation for CSV files

### `src/data_preprocessing.py`
Data cleaning and standardization:
- Missing value handling
- Outlier detection and treatment
- Type conversion
- Data validation
- Generates cleaned data file

### `src/feature_engineering.py`
Advanced feature creation:
- **New numerical features:**
  - `balance_to_salary_ratio` - Account balance relative to salary
  - `has_zero_balance` - Binary flag for zero balance
  - `engagement_score` - Customer engagement metric
  - `tenure_years` - Tenure in years
- **Categorical binning:**
  - Age group categorization (Young, Middle-aged, Senior)
- Train-test split (80/20 with stratification)

### `src/train_model.py`
Model training pipeline:
- Builds scikit-learn Pipeline with preprocessing and classifier
- Numerical features: StandardScaler
- Categorical features: OneHotEncoder
- Classifier: Random Forest with balanced class weights
- Cross-validation (5-fold) with ROC-AUC scoring
- Model serialization to pickle format

**Hyperparameters:**
```
n_estimators: 200
max_depth: 10
min_samples_split: 5
min_samples_leaf: 2
class_weight: balanced
random_state: 42
```

### `src/model_evaluation.py`
Comprehensive evaluation metrics:
- Accuracy, Precision, Recall
- F1-score
- ROC-AUC score
- Classification report
- Confusion matrix

### `src/predict.py`
Prediction utilities:
- `predict_single_customer()` - Predict for one customer
- `predict_batch()` - Predict for multiple customers
- Loads pre-trained model from pickle
- Returns churn probability and risk level (Low/Medium/High)

### `src/main.py`
Orchestrates the complete ML pipeline:
1. Data preprocessing
2. Feature engineering
3. Train-test split
4. Model training
5. Model evaluation

### `api/app.py`
Flask REST API with endpoints:
- `GET /` - Serve web interface
- `POST /predict` - Single prediction
- `POST /predict_batch` - Batch predictions
- `GET /health` - Health check
- Error handling and logging

### `templates/index.html`
Interactive web interface featuring:
- Form inputs for all features (8 numeric inputs + 2 toggles)
- Client-side validation
- Real-time feedback
- Result display with risk visualization
- Color-coded risk badges (Green/Yellow/Red)
- Advice based on risk level
- Responsive design

## Data

### Data Source
**File:** `data/raw/Bank Customer Churn Prediction.csv`

### Features (Input Variables)
| Feature | Type | Description |
|---------|------|-------------|
| credit_score | Numeric | Customer credit score (300-850) |
| age | Numeric | Customer age in years |
| tenure | Numeric | Years as customer |
| balance | Numeric | Account balance ($) |
| products_number | Numeric | Number of products owned |
| estimated_salary | Numeric | Annual estimated salary ($) |
| country | Categorical | Country (France, Germany, Spain) |
| gender | Categorical | Gender (Male, Female) |
| credit_card | Binary | Has credit card (0/1) |
| active_member | Binary | Is active member (0/1) |

### Target Variable
| Field | Description |
|-------|-------------|
| churn | Whether customer churned (0 = No, 1 = Yes) |

### Data Processing
1. **Raw Data** → `data/raw/Bank Customer Churn Prediction.csv`
2. **Cleaned Data** → `data/processed/cleaned_churn.csv`

## Model Details

### Algorithm
**Random Forest Classifier** with:
- 200 trees
- Max depth of 10
- Balanced class weights for imbalanced data
- 5-fold cross-validation

### Training Approach
- Stratified train-test split (80/20)
- Cross-validation for robust performance estimation
- ROC-AUC metric for evaluation
- Hyperparameter tuning via config

### Features Used
**Numerical Features:**
- credit_score, age, tenure, balance, products_number
- estimated_salary, balance_to_salary_ratio
- has_zero_balance, engagement_score

**Categorical Features:**
- country (One-Hot Encoded)
- gender (One-Hot Encoded)
- age_group (One-Hot Encoded)

### Pipeline Architecture
```
Input Data
    ↓
ColumnTransformer
├─ Numeric: StandardScaler
└─ Categorical: OneHotEncoder
    ↓
RandomForestClassifier
    ↓
Prediction (probability + class)
```

## API Endpoints

### 1. Web Interface
```
GET /
```
Serves the interactive prediction form.

**Response:** HTML page

---

### 2. Single Prediction
```
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "credit_score": 650,
  "age": 35,
  "tenure": 3,
  "balance": 80000,
  "products_number": 2,
  "estimated_salary": 55000,
  "country": "France",
  "gender": "Male",
  "credit_card": 1,
  "active_member": 1
}
```

**Success Response (200):**
```json
{
  "status": "success",
  "prediction": {
    "churn_probability": 0.25,
    "risk_level": "Low"
  }
}
```

**Error Response (400/500):**
```json
{
  "status": "error",
  "message": "Error description"
}
```

---

### 3. Batch Predictions
```
POST /predict_batch
Content-Type: application/json
```

**Request Body:**
```json
{
  "customers": [
    {"credit_score": 650, "age": 35, ...},
    {"credit_score": 720, "age": 45, ...}
  ]
}
```

**Success Response (200):**
```json
{
  "status": "success",
  "predictions": [
    {"churn_probability": 0.25, "risk_level": "Low"},
    {"churn_probability": 0.75, "risk_level": "High"}
  ]
}
```

---

### 4. Health Check
```
GET /health
```

**Response (200):**
```json
{
  "status": "healthy"
}
```

---

## Web Interface

### Features
- **Responsive Design:** Works on desktop and mobile
- **Form Validation:** Client-side input validation
- **Interactive Fields:**
  - Text/number inputs for numeric features
  - Dropdowns for categorical features
  - Toggle switches for binary flags
- **Result Display:**
  - Churn probability percentage
  - Risk level (Low/Medium/High)
  - Color-coded badge (Green/Yellow/Red)
  - Actionable recommendations
  - Loading indicator during prediction
- **User Actions:**
  - Submit prediction request
  - Reset form for new customer
  - Smooth scrolling to results

### Risk Levels & Recommendations

| Risk Level | Probability Range | Recommendation |
|-----------|------------------|-----------------|
| **Low** | 0-33% | Schedule routine check-in in 90 days |
| **Medium** | 33-67% | Consider personal outreach and loyalty offer |
| **High** | 67-100% | Escalate to retention team for account review |

---

## Results

### Model Performance
The Random Forest model is trained with 5-fold cross-validation:
- **Metric:** ROC-AUC Score
- **Cross-validation:** 5 folds
- **Class Balance:** Handled with balanced weights

### Output Artifacts
1. **Model File:** `models/churn_model.pkl` (~9.4 MB)
2. **Cleaned Data:** `data/processed/cleaned_churn.csv`
3. **Evaluation Metrics:** Printed to console during training

### Key Statistics
- Training set: ~80% of data
- Test set: ~20% of data
- Features after engineering: 13+
- Random state: 42 (reproducibility)

---

## Directory Structure

```
customer_churn_prediction/
├── src/
│   ├── __init__.py
│   ├── config.py                      # Configuration
│   ├── data_ingestion.py              # Data loading
│   ├── data_preprocessing.py          # Data cleaning
│   ├── feature_engineering.py         # Feature creation
│   ├── train_model.py                 # Model training
│   ├── model_evaluation.py            # Evaluation metrics
│   ├── predict.py                     # Prediction logic
│   └── main.py                        # Pipeline orchestration
├── api/
│   └── app.py                         # Flask application
├── data/
│   ├── raw/
│   │   └── Bank Customer Churn Prediction.csv
│   └── processed/
│       └── cleaned_churn.csv
├── models/
│   └── churn_model.pkl                # Trained model
├── notebooks/
│   └── eda.ipynb                      # Exploratory analysis
├── templates/
│   └── index.html                     # Web UI
├── static/                            # Static assets
├── requirements.txt                   # Dependencies
└── README.md                          # Documentation
```

---

## Troubleshooting

### Issue: Model file not found
**Solution:** Run `python src/main.py` to train the model first.

### Issue: Port 5000 already in use
**Solution:** Change port in `api/app.py` (line 57):
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Issue: Predictions look wrong
**Solution:**
1. Verify input data format matches schema
2. Ensure model is latest version
3. Check preprocessing matches training pipeline

### Issue: Missing data files
**Solution:** Ensure raw data CSV exists at `data/raw/Bank Customer Churn Prediction.csv`

---

## Development

### Adding New Features
1. Add feature creation logic in `src/feature_engineering.py`
2. Update feature lists in `src/config.py`
3. Retrain the model: `python src/main.py`

### Modifying Model Hyperparameters
Edit `RF_PARAMS` in `src/config.py`:
```python
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    # ... other parameters
}
```

### Extending the API
Add new routes in `api/app.py`:
```python
@app.route('/new-endpoint', methods=['POST'])
def new_endpoint():
    # Implementation
    return jsonify({'result': ...})
```

---

## Version History

- **v1.0.0** - Initial release with complete ML pipeline and web API

---

## License

[Specify your license here]

---

## Contact & Support

For issues or questions, please create an issue in the repository or contact the development team.

---

**Built with:** Python, scikit-learn, Flask, Random Forest, pandas, numpy
