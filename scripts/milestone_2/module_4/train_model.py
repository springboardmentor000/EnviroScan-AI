import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


df = pd.read_csv("../../../data/processed/final_dataset.csv")

print("Dataset shape:", df.shape)

X = df.drop(columns=["pollution_source", "city"])
y = df["pollution_source"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


dt = DecisionTreeClassifier(class_weight="balanced")

dt_params = {
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5]
}

dt_grid = GridSearchCV(dt, dt_params, cv=3, scoring="f1_weighted")
dt_grid.fit(X_train, y_train)

best_dt = dt_grid.best_estimator_


rf = RandomForestClassifier(class_weight="balanced")

rf_params = {
    "n_estimators": [50, 100],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5]
}

rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring="f1_weighted")
rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_


def evaluate(model, name):
    y_pred = model.predict(X_test)

    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


evaluate(best_dt, "Decision Tree")
evaluate(best_rf, "Random Forest")


joblib.dump(best_rf, "../../../model/final_model.pkl")

print("\nModel saved as final_model.pkl")