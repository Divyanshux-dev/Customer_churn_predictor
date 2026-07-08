import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    source_data_path: str = os.path.join("data", "processed", "telco_churn_cleaned.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Entered the data ingestion component")

        try:
            logger.info(f"Reading cleaned dataset from {self.ingestion_config.source_data_path}")
            df = pd.read_csv(self.ingestion_config.source_data_path)

            logger.info(f"Dataset loaded successfully with shape: {df.shape}")

            # Create artifacts directory if it does not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw copy into artifacts
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logger.info(f"Raw dataset saved at {self.ingestion_config.raw_data_path}")

            logger.info("Performing train-test split")
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df["Churn"]
            )

            logger.info(f"Train shape: {train_set.shape}")
            logger.info(f"Test shape: {test_set.shape}")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logger.info(f"Train file saved at {self.ingestion_config.train_data_path}")
            logger.info(f"Test file saved at {self.ingestion_config.test_data_path}")
            logger.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logger.error("Exception occurred during data ingestion")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print("Train path:", train_path)
    print("Test path:", test_path)