import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


def train_model():
    print(" Loading dataset...")
    df = pd.read_csv("data/processed/labeled_dataset.csv")

    # Define the exact 4 features the dashboard uses
    features = ["pm2_5", "no2", "so2", "o3"]
    X = df[features]
    
    # Keeping pollution_source as string labels so Dashboard prints actual words
    y = df["pollution_source"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(" Dataset Split Completed")

    print("\n Hyperparameter Tuning (Random Forest)...")

    rf_params = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5]
    }

    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params,
        cv=3,
        scoring="accuracy"
    )

    rf_grid.fit(X_train, y_train)

    rf_model = rf_grid.best_estimator_

    print(" Best Parameters:", rf_grid.best_params_)

    print("\n Cross Validation...")
    cv_scores = cross_val_score(rf_model, X, y, cv=5)
    print("CV Scores:", cv_scores)
    print("Average CV Score:", cv_scores.mean())

    print("\n Model Evaluation")
    predictions = rf_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))
    
    # Save to root directly so dashboard.py natively finds it
    joblib.dump(rf_model, "model.pkl")
    print("\n Model saved successfully as model.pkl")


if __name__ == "__main__":
    train_model()