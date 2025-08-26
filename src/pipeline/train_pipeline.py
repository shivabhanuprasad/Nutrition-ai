import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from joblib import dump

# Import your custom exception handler (optional)
from src.exception import CustomException

try:
    # Load dataset
    df = pd.read_csv("data/meal_data.csv")   # <-- change path if needed

    # Drop Meal_ID (not useful for prediction)
    if "Meal_ID" in df.columns:
        df = df.drop(columns=["Meal_ID"])

    # Split features & target
    X = df.drop(columns=["Dish_Name"])
    y = df["Dish_Name"]

    # Separate categorical & numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Preprocessor
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    # Full pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=42))
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    pipeline.fit(X_train, y_train)

    # Save pipeline
    os.makedirs("artifacts", exist_ok=True)
    dump(pipeline, "artifacts/model.pkl")

    print("✅ Training completed and model saved as artifacts/model.pkl")

except Exception as e:
    raise CustomException(e, sys)
