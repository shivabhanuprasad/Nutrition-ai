# Importing required libraries 
import os
import sys
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

# --- Ensure package imports work even when running this file directly ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.exception import CustomException   # Custom exception class
from src.logger import logging              # Project logger

# Importing pipeline components (your folder is 'Components' with capital C)
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig


# -------------------- CONFIG --------------------
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str  = os.path.join('artifacts', "test.csv")
    raw_data_path: str   = os.path.join('artifacts', "data.csv")


# -------------------- INGESTION --------------------
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion.")
        try:
            # Prefer project-root/notebook/NutriPlan_AI_Dataset.csv
            data_csv_path = PROJECT_ROOT / "notebook" / "NutriPlan_AI_Dataset.csv"
            if not data_csv_path.exists():
                raise FileNotFoundError(f"Input data file not found: {data_csv_path}")

            df = pd.read_csv(data_csv_path)
            logging.info(f"Read dataset: shape={df.shape}")

            # Ensure artifacts folder exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            # Train/test split
            logging.info("Performing 80/20 train-test split...")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data ingestion completed.")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)


# -------------------- MAIN ENTRY --------------------
if __name__ == "__main__":
    # 1) Ingest
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print(f"Train data saved to: {train_data}")
    print(f"Test data saved to:  {test_data}")

    # 2) Transform
    data_transformation = DataTransformation()
    train_arr, test_arr, preproc_path = data_transformation.initiate_data_transformation(train_data, test_data)
    print(f"Preprocessor saved to: {preproc_path}")

    # 3) Train model
    model_trainer = ModelTrainer()
    final_r2 = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"✅ Final Test Accuracy: {final_r2:.4f}")
