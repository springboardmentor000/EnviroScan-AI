# ==========================================
# MODULE 4: MODEL TRAINING & SOURCE PREDICTION
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv("labeled_environment_dataset.csv")

print("Dataset Shape:", df.shape)


# ==============================
# SELECT FEATURES
# ==============================

features = [
    "pm25","pm10","no2","co","so2","o3",
    "temperature","humidity","pressure","wind_speed",
    "dist_to_road","dist_to_industry","dist_to_dump"
]

X = df[features]

# Target variable
y = df["source_label"]


# ==============================
# ENCODE LABEL
# ==============================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)


# ==============================
# TRAIN TEST SPLIT (80/20)
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

print("Training Size:", X_train.shape)
print("Testing Size:", X_test.shape)


# ==================================
# DECISION TREE MODEL
# ==================================

dt_model = DecisionTreeClassifier(max_depth=5)

dt_model.fit(X_train, y_train)


# ==================================
# RANDOM FOREST MODEL
# ==================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

rf_model.fit(X_train, y_train)


# ==================================
# XGBOOST MODEL
# ==================================

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    eval_metric="mlogloss"
)

xgb_model.fit(X_train, y_train)


# ==================================
# EVALUATION FUNCTION
# ==================================

def evaluate(model, name):

    predictions = model.predict(X_test)

    print("\n==========================")
    print(name)
    print("==========================")

    acc = accuracy_score(y_test, predictions)

    print("Accuracy:", acc)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))


# ==============================
# EVALUATE MODELS
# ==============================

evaluate(dt_model, "Decision Tree")

evaluate(rf_model, "Random Forest")

evaluate(xgb_model, "XGBoost")


# ==================================
# SAVE BEST MODEL (Random Forest)
# ==================================

joblib.dump(rf_model, "pollution_source_model.pkl")

joblib.dump(encoder, "label_encoder.pkl")

print("\nModel saved successfully!")