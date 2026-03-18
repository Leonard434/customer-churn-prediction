import pandas as pd
import numpy as np
import logging
from config import DATA_RAW, COLUMN_TO_DROP, DATA_CLEAN, TARGET_VAR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(path=DATA_RAW):
    '''Load raw data from the specified path'''
    try:
        df = pd.read_csv(path)
        logger.info(f"data loaded rows:{df.shape[0]} columns:{df.shape[1]}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def drop_unnecessary_columns(df, columns_to_drop=COLUMN_TO_DROP):
    '''Drop unnecessary columns from the dataframe'''
    df = df.drop(columns=columns_to_drop)
    logger.info(f"Dropped columns: {columns_to_drop}")
    return df

def handle_missing_values(df: pd.DataFrame)  -> pd.DataFrame:
    """ Handle missing values in the dataframe by filling them with the median for numeric columns and mode for categorical columns.
    Args:
        df (pd.DataFrame): Input dataframe with potential missing values.
    Returns:
        pd.DataFrame: Dataframe with missing values handled.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in numeric_cols:
        missing = df.isnull().sum()[col]
        if missing > 0:
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            logger.info(f"Filled {missing} missing values in numeric column '{col}' with median: {median_value}")

    for col in categorical_cols:
        missing = df.isnull().sum()[col]
        if missing > 0:
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)
            logger.info(f"Filled {missing} missing values in categorical column '{col}' with mode: {mode_value}")

    return df

def run_preprocessing_pipeline():
    logger.info("Starting data preprocessing...")
    df = (load_data()
          .pipe(drop_unnecessary_columns)
          .pipe(handle_missing_values)
    )
    # save cleaned data
    DATA_CLEAN.parent.mkdir(exist_ok=True, parents=True)  # Ensure the directory exists
    df.to_csv(DATA_CLEAN, index=False)
    logger.info(f"Data preprocessing completed. Cleaned data saved to {DATA_CLEAN}")
    logger.info(f"churn rate: {df[TARGET_VAR].mean():.2%}")

    return df

if __name__ == "__main__":
    run_preprocessing_pipeline()


