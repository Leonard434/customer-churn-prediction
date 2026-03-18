import joblib
from config import MODEL_PATH
import pandas as pd
from feature_engineering import create_new_features
import logging
logging.getLogger(__name__)

# load the model and make predictions
def load_pipeline():
    '''Load the trained model pipeline from disk.'''
    return joblib.load(MODEL_PATH)

_pipeline = None
def predict_single_customer(customer_data: dict):
    '''Predict churn for a single customer.'''
    global _pipeline
    if _pipeline is None:
        _pipeline = load_pipeline()
    df = pd.DataFrame([customer_data])
    df = df.drop(columns=['churn', 'customer_id'], errors='ignore')
    df = create_new_features(df)
    logging.info("Making prediction for a single customer...")
    proba = _pipeline.predict_proba(df)[:,1][0]
    risk = 'High' if proba >= 0.7 else 'Medium' if proba >= 0.4 else 'Low'
    logging.info(f"Predicted churn probability: {proba:.4f}, Risk level: {risk}")
    return {'will_churn': bool(proba >= 0.5), 'churn_probability': round(float(proba), 4), 'risk_level': risk}

# predict for a batch of customers
def predict_batch(df: pd.DataFrame):
    '''Predict churn for a batch of customers.'''
    global _pipeline
    if _pipeline is None:
        _pipeline = load_pipeline()
        customers_df = create_new_features(df.copy())
    logging.info("Making predictions for a batch of customers...")
    customers_df = customers_df.drop(columns=['churn', 'customer_id'], errors='ignore')
    proba = _pipeline.predict_proba(customers_df)[:,1]
    results_df = df.copy()
    results_df['churn_probability'] = proba
    results_df['will_churn'] = results_df['churn_probability'] >= 0.5
    results_df['risk_level'] = results_df['churn_probability'].apply(lambda x: 'High' if x >= 0.7 else 'Medium' if x >= 0.4 else 'Low')
    logging.info("Batch predictions completed.")
    return results_df

if __name__ == "__main__":
    # Example usage
    sample_customer = {
        'customer_id': 12345,
        'credit_score': 600,
        'country': 'France',
        'gender': 'Male',
        'age': 40,
        'tenure': 3,
        'balance': 50000.0,
        'products_number': 2,
        'credit_card': 1,
        'active_member': 1,
        'estimated_salary': 60000.0
    }    
    prediction = predict_single_customer(sample_customer)
    print(prediction)