import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path

class DataIngestion(ABC):
    '''Abstract base class for data ingestion.'''
    @abstractmethod
    def ingest_data(self) -> pd.DataFrame:
        pass    

class CSVDataIngestion(DataIngestion):
    '''Data ingestion class for CSV files.'''
    def __init__(self, file_path: str):
        self.file_path = file_path

    def ingest_data(self) -> pd.DataFrame:
        try:
            data = pd.read_csv(self.file_path)
            print(f"Data successfully ingested from {self.file_path}")
            return data
        except Exception as e:
            print(f"Error ingesting data from {self.file_path}: {e}")
            return pd.DataFrame()


if __name__ == "__main__":    # Example usage
    project_root = Path(__file__).resolve().parent.parent
    data_file_path = project_root / "Data" / "raw" / "Bank Customer Churn Prediction.csv"
    csv_ingestion = CSVDataIngestion(data_file_path)
    data = csv_ingestion.ingest_data()
    print(data.head())