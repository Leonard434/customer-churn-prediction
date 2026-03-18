import os
from pathlib import Path
import os

#-- paths ----------------------------------------------------------------------
# project root
BASE_DIR = Path(__file__).resolve().parent.parent   

# data paths
DATA_RAW    = BASE_DIR / "data" / "raw"    / "Bank Customer Churn Prediction.csv"
DATA_CLEAN  = BASE_DIR / "data" / "processed" / "cleaned_churn.csv"
MODEL_PATH  = BASE_DIR / "models" / "churn_model.pkl"

print("BASE_DIR =", BASE_DIR)
print("DATA_RAW =", DATA_RAW)

#-- MODEL SETTINGS ----------------------------------------------------------------
TARGET_VAR = 'churn'
RANDOM_STATE = 42
TEST_SIZE = 0.2
COLUMN_TO_DROP = ['customer_id']
CV_FOLDS = 5

#-- MODEL HYPERPARAMETERS ----------------------------------------------------------------
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth' : 10,
    'min_samples_split' : 5,
    'min_samples_leaf' : 2,
    'random_state' : RANDOM_STATE,
    'class_weight' : 'balanced',
    'n_jobs' : -1
}

# FEATURE ENGINEERING SETTINGS ----------------------------------------------------------------
FEATURES_TO_CREATE = ['total_charges_per_month', 'tenure_years']

NUMERICAL_FEATURES = ['credit_score', 
                      'age', 
                      'tenure', 
                      'balance', 
                      'products_number', 
                      'estimated_salary'
]


CATEGORICAL_FEATURES = ['country', 
                        'gender'
]