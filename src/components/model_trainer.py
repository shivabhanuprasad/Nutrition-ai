import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting arrays into X/y for training and testing...")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # -------------------- Candidate models --------------------
            models = {
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
            }

            # -------------------- Hyperparameters --------------------
            params = {
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20],
                },
                "Random Forest": {
                    "n_estimators": [64, 128],
                    "max_depth": [None, 10, 20],
                },
            }

            # -------------------- Train & Evaluate --------------------
            best_model_name, best_model, best_score = None, None, -1

            for name, model in models.items():
                logging.info(f"🔄 Training {name}...")

                gs = GridSearchCV(model, params[name], cv=3, n_jobs=-1, verbose=1)
                gs.fit(X_train, y_train)

                candidate = gs.best_estimator_
                y_pred = candidate.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                logging.info(f"✅ {name} finished | Accuracy = {acc:.4f}")

                if acc > best_score:
                    best_score = acc
                    best_model_name = name
                    best_model = candidate

            if best_model is None:
                raise CustomException("No model was successfully trained.", sys)

            logging.info(f"Best model: {best_model_name} | Accuracy = {best_score:.4f}")

            # -------------------- Save best model --------------------
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logging.info(
                f"Saved best model ({best_model_name}) to {self.model_trainer_config.trained_model_file_path}"
            )

            return best_score  # return test accuracy

        except Exception as e:
            raise CustomException(e, sys)
