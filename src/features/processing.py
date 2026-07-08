import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logger
from src.exception import CustomException


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns the preprocessing object for numerical and categorical features.
        """
        try:
            logger.info("Creating preprocessing pipelines for numerical and categorical features")

            # IMPORTANT:
            # These columns must match your final cleaned dataset exactly.
            # If you dropped PaymentMethod in EDA, remove it from categorical_columns.

            numerical_columns = [
                "tenure",
                "MonthlyCharges",
                "TotalCharges"
            ]

            categorical_columns = [
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            logger.info(f"Numerical columns: {numerical_columns}")
            logger.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            logger.info("Preprocessor object created successfully")
            return preprocessor

        except Exception as e:
            logger.error("Error while creating preprocessing object")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads train and test CSV files, fits preprocessing on training data,
        transforms both train and test data, and saves the preprocessor object.
        """
        try:
            logger.info("Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info(f"Train dataframe shape: {train_df.shape}")
            logger.info(f"Test dataframe shape: {test_df.shape}")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Churn"

            logger.info("Splitting input features and target column")

            # Drop target from X
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            input_feature_test_df = test_df.drop(columns=[target_column_name])

            # Convert target from Yes/No to 1/0
            target_feature_train_df = (
                train_df[target_column_name]
                .map({"No": 0, "Yes": 1})
                .astype(int)
            )

            target_feature_test_df = (
                test_df[target_column_name]
                .map({"No": 0, "Yes": 1})
                .astype(int)
            )

            logger.info("Fitting preprocessing object on training data")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)

            logger.info("Transforming test data")
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logger.info("Saving preprocessing object")
            os.makedirs(
                os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path),
                exist_ok=True
            )
            joblib.dump(
                preprocessing_obj,
                self.data_transformation_config.preprocessor_obj_file_path
            )

            # Append target as the last column
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logger.info("Data transformation completed successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logger.error("Exception occurred during data transformation")
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.data.data_ingestion import DataIngestion

    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
        train_path,
        test_path
    )

    print("Train array shape:", train_arr.shape)
    print("Test array shape:", test_arr.shape)
    print("Preprocessor saved at:", preprocessor_path)