import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import load_object
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        # Paths to saved artifacts
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.label_encoder_path = os.path.join("artifacts", "label_encoder.pkl")
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.dataset_path = os.path.join("artifacts", "data.csv")  # Raw dataset for lookups

        # Load artifacts
        self.preprocessor = load_object(self.preprocessor_path)
        self.label_encoder = load_object(self.label_encoder_path)
        self.model = load_object(self.model_path)

        # Load the raw dataset to get column information and for lookups
        if os.path.exists(self.dataset_path):
            self.dataset = pd.read_csv(self.dataset_path)
            # Define the feature columns used during training (all columns except the target 'Dish_Name')
            self.training_columns = self.dataset.drop(columns=["Dish_Name"], errors='ignore').columns.tolist()
        else:
            self.dataset = None
            self.training_columns = None
            logging.error(f"Raw dataset not found at {self.dataset_path}. Cannot perform predictions or lookups.")


    def predict(self, features: pd.DataFrame, top_k=5):
        """
        Makes predictions on the input features DataFrame.
        It aligns the input columns with the training columns before prediction.
        """
        try:
            if self.training_columns is None:
                raise CustomException("Training columns not defined because raw dataset is missing.", sys)

            logging.info("Starting prediction...")
            
            # --- FIX: Align input DataFrame with the columns expected by the preprocessor ---
            # Create a new DataFrame with the exact columns and order from training
            # This ensures that if a column like 'Meal_ID' is missing, it gets added as NaN
            aligned_features = pd.DataFrame(columns=self.training_columns)
            for col in self.training_columns:
                 if col in features.columns:
                     aligned_features[col] = features[col]
            
            logging.info("Aligned input features to match training data structure.")
            # --- END FIX ---

            # Transform the aligned data using the loaded preprocessor
            data_transformed = self.preprocessor.transform(aligned_features)

            # Get prediction probabilities if the model supports it
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(data_transformed)[0]
                # Get the indices of the top_k predictions
                top_indices = np.argsort(proba)[-top_k:][::-1]
                pred_classes = self.label_encoder.inverse_transform(top_indices)
                pred_probs = proba[top_indices]
            else:
                # Handle models that don't output probabilities
                pred_class = self.model.predict(data_transformed)
                pred_classes = self.label_encoder.inverse_transform(pred_class)
                pred_probs = [1.0] * len(pred_classes)

            # Prepare the final results with a calorie lookup from the raw dataset
            results = []
            if self.dataset is not None:
                for dish, prob in zip(pred_classes, pred_probs):
                    # Get all rows for a given dish to find calorie info
                    rows = self.dataset[self.dataset["Dish_Name"] == dish]
                    calories = rows["Calories_per_Serving"].iloc[0] if not rows.empty else None
                    
                    results.append({
                        "Dish_Name": dish,
                        "Confidence": round(float(prob), 3),
                        "Calories_per_Serving": calories
                    })

                # Sort results by confidence score in descending order
                results = sorted(results, key=lambda x: x["Confidence"], reverse=True)

            return results

        except Exception as e:
            # Wrap any exception in our custom exception for detailed logging
            raise CustomException(e, sys)

