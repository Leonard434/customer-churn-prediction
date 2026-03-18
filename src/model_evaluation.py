import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from config import MODEL_PATH, DATA_CLEAN
from feature_engineering import create_new_features, prepare_train_test_split
import logging
logging.getLogger(__name__)

# Evaluate the model
def evaluate_model(pipeline=None, X_test=None, y_test=None):
    '''Evaluate the trained model on the test set.'''
    if pipeline is None:
        pipeline = joblib.load(MODEL_PATH)
    if X_test is None or y_test is None:
        df = pd.read_csv(DATA_CLEAN)
        df = create_new_features(df)
        _, X_test, _, y_test = prepare_train_test_split(df)

    logging.info("Evaluating the model...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:,1]
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred)
    print(f"AUC Score: {auc_score:.4f}")
    print(classification_report(y_test, y_pred, target_names=['stayed', 'exited']))
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}, True Negatives: {tn}")
    logging.info("Model evaluation completed.")
    # Return features importance
    rf_model = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    num_features = list(preprocessor.transformers_[0][2])
    hot_encoder = preprocessor.transformers_[1][1].named_steps['onehot']
    cat_features = hot_encoder.get_feature_names_out(preprocessor.transformers_[1][2])
    importance_df = pd.DataFrame({
        'feature': num_features + list(cat_features),
        'importance': rf_model.feature_importances_
    }).sort_values(by='importance', ascending=False).head(10)
    print("Top 10 Feature Importances:")
    print(importance_df.to_string(index=False))
    return auc_score, importance_df

if __name__ == "__main__":
    evaluate_model()