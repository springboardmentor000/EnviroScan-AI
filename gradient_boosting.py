import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ================================
# Create folders
# ================================
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ================================
# Load Dataset
# ================================
df = pd.read_csv("vijayawada_labelled_dataset.csv")

print("Dataset Loaded:", df.shape)

# ================================
# Features
# ================================
features = [
    'pm2_5','pm10','no2','o3','so2','co',
    'temperature_c','humidity','pressure_hpa',
    'wind_speed_ms','wind_direction'
]

X = df[features]
y = df["pollution_source"]

print("Features used:", features)

# ================================
# Label Encoding
# ================================
le = LabelEncoder()
y = le.fit_transform(y)

# ================================
# SMOTE Balancing
# ================================
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print("After SMOTE:")
print(pd.Series(y_res).value_counts())

# ================================
# Train Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# ================================
# Gradient Boosting BEFORE Tuning
# ================================
print("\n===== Gradient Boosting BEFORE Tuning =====")

gb = GradientBoostingClassifier(random_state=42)

gb.fit(X_train, y_train)

# Predictions
train_pred = gb.predict(X_train)
test_pred = gb.predict(X_test)

# Accuracies
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)
model_accuracy = accuracy_score(y_test, test_pred)

print("\nModel Accuracy:", round(model_accuracy*100,2), "%")
print("Train Accuracy:", round(train_accuracy*100,2), "%")
print("Test Accuracy:", round(test_accuracy*100,2), "%")

print("\nClassification Report:")
print(classification_report(y_test, test_pred, target_names=le.classes_))

# ================================
# Hyperparameter Tuning
# ================================
print("\n===== Gradient Boosting AFTER Tuning =====")

param_grid = {
    "n_estimators":[100,200],
    "learning_rate":[0.05,0.1,0.2],
    "max_depth":[1]
}

grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_gb = grid.best_estimator_

print("Best Parameters:", grid.best_params_)

# ================================
# Evaluate Tuned Model
# ================================
train_pred = best_gb.predict(X_train)
test_pred = best_gb.predict(X_test)

train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)
model_accuracy = accuracy_score(y_test, test_pred)

print("\nModel Accuracy:", round(model_accuracy*100,2), "%")
print("Train Accuracy:", round(train_accuracy*100,2), "%")
print("Test Accuracy:", round(test_accuracy*100,2), "%")

print("\nClassification Report:")
print(classification_report(y_test, test_pred, target_names=le.classes_))

# ================================
# Cross Validation
# ================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    best_gb,
    X_res,
    y_res,
    cv=skf,
    scoring="f1_weighted"
)

print("\nCross Validation F1 Scores:", cv_scores)
print("Average F1:", np.mean(cv_scores))

# ================================
# Confusion Matrix Plot
# ================================
cm = confusion_matrix(y_test, test_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap="Blues")

plt.title("Confusion Matrix - Gradient Boosting")
plt.ylabel("Actual")
plt.xlabel("Predicted")

plt.tight_layout()
plt.savefig("plots/gradient_boost_confusion_matrix.png")
plt.show()

print("Confusion matrix saved.")

# ================================
# Feature Importance Plot
# ================================
importance = best_gb.feature_importances_

importance_df = pd.Series(importance, index=features).sort_values()

plt.figure(figsize=(8,6))
importance_df.plot(kind="barh")

plt.title("Feature Importance - Gradient Boosting")
plt.xlabel("Importance Score")

plt.tight_layout()
plt.savefig("plots/gradient_boost_feature_importance.png")
plt.show()

print("Feature importance plot saved.")

# ================================
# Save Model
# ================================
joblib.dump(best_gb,"models/gradient_boost_pollution_model.pkl")

print("\nModel saved successfully.") 