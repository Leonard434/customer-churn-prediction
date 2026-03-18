import logging
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_preprocessing import run_preprocessing_pipeline
from feature_engineering import create_features, prepare_train_test_split
from train_model import train
from model_evaluation import evaluate

logging.basicConfig(level=logging.INFO,
format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
def run_full_pipeline():
    logger.info("Starting the full pipeline...")
    # Step 1: Data Preprocessing
    logger.info("Running data preprocessing...")
    df_clean = run_preprocessing_pipeline()
    
    # Step 2: Feature Engineering
    logger.info("Creating features...")
    df_features = create_features(df_clean)
    
    # Step 3: Train-Test Split
    logger.info("Preparing train-test split...")
    X_train, X_test, y_train, y_test = prepare_train_test_split(df_features)
    
    # Step 4: Train the Model
    logger.info("Training the model...")
    model_pipeline = train(X_train, y_train)
    
    # Step 5: Evaluate the Model
    logger.info("Evaluating the model...")
    evaluate(model_pipeline, X_test, y_test)
    
    logger.info("Pipeline execution completed.")


if __name__ == "__main__":
    run_full_pipeline()
