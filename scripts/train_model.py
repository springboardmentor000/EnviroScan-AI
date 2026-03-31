import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def train_model():

    print(" Loading dataset...")

    df = pd.read_csv("data/processed/final_dataset.csv")

    label_encoder = LabelEncoder()
    df["pollution_source"] = label_encoder.fit_transform(df["pollution_source"])

    X = df.select_dtypes(include=["number"])

    y = df["pollution_source"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("✔ Dataset Split Completed")

    print("\n🔧 Hyperparameter Tuning...")

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

    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)

    print("\n Cross Validation...")

    cv_scores = cross_val_score(rf_model, X, y, cv=5)

    print("CV Scores:", cv_scores)
    print("Average CV Score:", cv_scores.mean())

    print("\n Model Evaluation")

    predictions = rf_model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))

    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, predictions))

    os.makedirs("models", exist_ok=True)

    joblib.dump(rf_model, "models/pollution_model.pkl")

    print("\n Model saved → models/pollution_model.pkl")