import pandas as pd
from config import TEST_SIZE, RANDOM_STATE, DATA_CLEAN
from sklearn.model_selection import train_test_split
import logging

logging.getLogger(__name__)

# FEATURE ENGINEERING ----------------------------------------------------------------
def create_new_features(df: pd.DataFrrame) -> pd.DataFrame:
    """ Create new features based on existing ones. For example, we can create a feature that represents the average monthly charges by dividing total charges by tenure.
    Args:
        df (pd.DataFrame): Input dataframe with existing features.
    Returns:
        pd.DataFrame: Dataframe with new features added.
    """
    df = df.copy()  # Avoid modifying the original dataframe
    logging.info("Creating new features...")

    df['balance_to_salary_ratio'] = df['balance'] / (df['estimated_salary'] + 1)  # Adding 1 to avoid division by zero
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle-aged', 'Senior']).astype(str)
    df['has_zero_balance'] = (df['balance'] == 0).astype(int)
    df['engagement_score'] = df['products_number'] * df['estimated_salary']
    logging.info("New features created.")
    return df

def prepare_train_test_split(df):
    ''' Prepare the train-test split of the dataset. This function will separate the features and target variable, and then split the data into training and testing sets.
    Args:
        df (pd.DataFrame): Input dataframe with features and target variable.
    Returns:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target variable.
        y_test (pd.Series): Testing target variable.
    '''
    logging.info("Preparing train-test split...")
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    logging.info("Train-test split completed.")
    print(f"Train set rate: {y_train.mean():.2f}, \nTest set rate: {y_test.mean():.2f}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = pd.read_csv(DATA_CLEAN)
    df = create_new_features(df)
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)
    print(df.head())