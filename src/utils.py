# utils.py
import os
import sys
import pickle
from sklearn.metrics import accuracy_score   # ✅ classification metric
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for name, model in models.items():
            print(f"🔄 Training {name}...")   # live progress

            # Use smaller CV and fewer parameter choices
            param_grid = param.get(name, {})
            if len(param_grid) > 0:
                gs = GridSearchCV(
                    model,
                    param_grid,
                    cv=2,         # smaller CV (faster)
                    n_jobs=1,     # avoid CPU overload
                    verbose=1
                )
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model

            y_test_pred = best_model.predict(X_test)
            score = accuracy_score(y_test, y_test_pred)   # ✅ accuracy instead of R²
            report[name] = score

            print(f"✅ {name} finished | Accuracy = {score:.4f}")

        return report
    except Exception as e:
        raise CustomException(e, sys)
