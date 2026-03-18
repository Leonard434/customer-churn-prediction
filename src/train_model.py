from config import RF_PARAMS, MODEL_PATH, CV_FOLDS
from feature_engineering import create_new_features, prepare_train_test_split
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import logging
logging.getLogger(__name__)
logging.info("Starting model training...")

# pipeline
def build_sklearn_pipeline():
    ''' Build a sklearn pipeline for the model. '''
    numeric_features = ['credit_score', 'age', 'tenure', 'balance', 'products_number',
       'credit_card', 'active_member', 'estimated_salary',
       'balance_to_salary_ratio', 'has_zero_balance', 'engagement_score']
    
    categorical_features = ['country', 'gender', 'age_group']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])
    
    categorical_transdormer =   Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transdormer, categorical_features),
    ], remainder='drop')

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(**RF_PARAMS))
    ])
    return pipeline

def train_model(df=None):
    '''Train the model using the provided pipeline and training data.'''
    if df is None:
        from config import DATA_CLEAN
        df = pd.read_csv(DATA_CLEAN)
        df = create_new_features(df)
        X_train, _, y_train, _ = prepare_train_test_split(df)
        pipeline = build_sklearn_pipeline()

    # Fit the model
    logging.info("Fitting the model...")
    cross_val = cross_val_score(pipeline, 
                                X_train, 
                                y_train, 
                                cv=CV_FOLDS, 
                                scoring='roc_auc')
    logging.info(f"Cross-validation AUC scores: {cross_val}")
    print(f"Mean CV AUC: {cross_val.mean():.4f} ± {cross_val.std():.4f}")
    pipeline.fit(X_train, y_train)
    logging.info("Model training completed.")
    # Save the model
    MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)  # Ensure the directory exists
    joblib.dump(pipeline, MODEL_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")
    return pipeline, X_train, y_train

if __name__ == "__main__":
    train_model()