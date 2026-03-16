

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

print("Starting Module 4: Model Training & Prediction...\n")

# =========================
# Step 1: Load Labeled Dataset
# =========================
df = pd.read_csv("module3_labeled_dataset.csv")
print("✅ Loaded dataset with", len(df), "rows")

# Features (pollutants, weather, proximity) and target (pollution_source)
feature_cols = ['PM2_5', 'PM10', 'NO2', 'CO', 'SO2', 'O3',
                'temperature_C', 'humidity_%', 'wind_speed_mps',
                'dist_to_road', 'dist_to_industry', 'dist_to_farmland']
X = df[feature_cols]
y = df['pollution_source']  # target variable (string labels)

# =========================
# Step 2: Split Train/Test
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("✅ Split data: Training =", len(X_train), "Test =", len(X_test))

# =========================
# Step 2a: Encode Target for XGBoost
# =========================
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# =========================
# Step 3: Train Models
# =========================
# 3a. Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
print("✅ Random Forest trained")

# 3b. Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
print("✅ Decision Tree trained")

# 3c. XGBoost
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train_enc)
print("✅ XGBoost trained")

# =========================
# Step 4: Hyperparameter Tuning for Random Forest (Optional)
# =========================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    param_grid, cv=3, scoring='f1_macro')
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("✅ Best Random Forest found via GridSearchCV")
print("Best Parameters:", grid.best_params_)

# =========================
# Step 5: Model Evaluation
# =========================
y_pred = best_model.predict(X_test)

print("\n===== Random Forest Evaluation =====")
print("Accuracy: ", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred, average='macro'), 3))
print("Recall:   ", round(recall_score(y_test, y_pred, average='macro'), 3))
print("F1 Score: ", round(f1_score(y_test, y_pred, average='macro'), 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# XGBoost evaluation
y_pred_xgb = xgb_model.predict(X_test)
y_pred_xgb = le.inverse_transform(y_pred_xgb)  # convert back to string labels

print("\n===== XGBoost Evaluation =====")
print("Accuracy: ", round(accuracy_score(y_test, y_pred_xgb), 3))
print("Precision:", round(precision_score(y_test, y_pred_xgb, average='macro'), 3))
print("Recall:   ", round(recall_score(y_test, y_pred_xgb, average='macro'), 3))
print("F1 Score: ", round(f1_score(y_test, y_pred_xgb, average='macro'), 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

# =========================
# Step 6: Save the Trained Model
# =========================
joblib.dump(best_model, 'pollution_source_model.pkl')
print("\n✅ Trained Random Forest model saved as pollution_source_model.pkl")

# =========================
# Step 7: Optional: Predict on New Data
# =========================
# Example new location
new_data = pd.DataFrame([{
    'PM2_5': 60, 'PM10': 80, 'NO2': 45, 'CO': 0.8, 'SO2': 12, 'O3': 30,
    'temperature_C': 32, 'humidity_%': 40, 'wind_speed_mps': 2,
    'dist_to_road': 50, 'dist_to_industry': 200, 'dist_to_farmland': 300
}])

predicted_source = best_model.predict(new_data)
print("\nExample Prediction for New Data (Random Forest): ", predicted_source[0])

print("\nModule 4 Completed Successfully!")