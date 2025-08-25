import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")
    label_encoder_path: str = os.path.join('artifacts', "label_encoder.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def _build_preprocessor(self, df: pd.DataFrame, target_col: str):
        try:
            # Identify features
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()

            # Remove target
            if target_col in num_cols:
                num_cols.remove(target_col)
            if target_col in cat_cols:
                cat_cols.remove(target_col)

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ])

            logging.info(f"Numerical features: {num_cols}")
            logging.info(f"Categorical features: {cat_cols}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, num_cols),
                    ("cat", cat_pipeline, cat_cols),
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self, train_path: str, test_path: str, target_column_name: str = "Dish_Name"
    ):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f"Loaded train/test: {train_df.shape} / {test_df.shape}")

            # Preprocessor for features
            preprocessor = self._build_preprocessor(train_df, target_col=target_column_name)

            # Label encode target
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(train_df[target_column_name])
            y_test = label_encoder.transform(test_df[target_column_name])

            X_train = train_df.drop(columns=[target_column_name], axis=1)
            X_test = test_df.drop(columns=[target_column_name], axis=1)

            logging.info("Fitting preprocessor and transforming data...")
            X_train_t = preprocessor.fit_transform(X_train)
            X_test_t = preprocessor.transform(X_test)

            train_arr = np.hstack((X_train_t.toarray() if hasattr(X_train_t, "toarray") else X_train_t, y_train.reshape(-1, 1)))
            test_arr  = np.hstack((X_test_t.toarray() if hasattr(X_test_t, "toarray") else X_test_t, y_test.reshape(-1, 1)))

            # Save artifacts
            save_object(self.config.preprocessor_obj_file_path, preprocessor)
            save_object(self.config.label_encoder_path, label_encoder)

            logging.info(f"Saved preprocessor to {self.config.preprocessor_obj_file_path}")
            logging.info(f"Saved label encoder to {self.config.label_encoder_path}")

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
