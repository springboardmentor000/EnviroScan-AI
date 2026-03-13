# ==========================================
# MODULE 4: MODEL TRAINING & SOURCE PREDICTION
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


# ==============================
# LOAD LABELED DATASET
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

y = df["source_label"]


# ==============================
# ENCODE TARGET LABEL
# ==============================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)


# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42
)

print("Training Size:", X_train.shape)
print("Testing Size:", X_test.shape)


# ==========================================
# 1️⃣ DECISION TREE MODEL
# ==========================================

dt = DecisionTreeClassifier()

dt_params = {
    "max_depth":[3,5,10],
    "min_samples_split":[2,5,10]
}

dt_grid = GridSearchCV(dt, dt_params, cv=3)

dt_grid.fit(X_train, y_train)

best_dt = dt_grid.best_estimator_

print("\nBest Decision Tree Params:", dt_grid.best_params_)


# ==========================================
# 2️⃣ RANDOM FOREST MODEL
# ==========================================

rf = RandomForestClassifier()

rf_params = {
    "n_estimators":[50,100],
    "max_depth":[5,10,None]
}

rf_grid = GridSearchCV(rf, rf_params, cv=3)

rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_

print("\nBest Random Forest Params:", rf_grid.best_params_)


# ==========================================
# 3️⃣ XGBOOST MODEL
# ==========================================

xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")

xgb_params = {
    "n_estimators":[50,100],
    "max_depth":[3,6],
    "learning_rate":[0.1,0.3]
}

xgb_grid = GridSearchCV(xgb, xgb_params, cv=3)

xgb_grid.fit(X_train, y_train)

best_xgb = xgb_grid.best_estimator_

print("\nBest XGBoost Params:", xgb_grid.best_params_)


# ==========================================
# 4️⃣ GRADIENT BOOSTING MODEL
# ==========================================

gb = GradientBoostingClassifier()

gb_params = {
    "n_estimators":[50,100],
    "learning_rate":[0.05,0.1],
    "max_depth":[3,5]
}

gb_grid = GridSearchCV(gb, gb_params, cv=3)

gb_grid.fit(X_train, y_train)

best_gb = gb_grid.best_estimator_

print("\nBest Gradient Boosting Params:", gb_grid.best_params_)


# ==========================================
# MODEL EVALUATION FUNCTION
# ==========================================

def evaluate_model(model, name):

    predictions = model.predict(X_test)

    print("\n==============================")
    print(name)
    print("==============================")

    print("Accuracy:", accuracy_score(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=encoder.classes_))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))


# ==============================
# EVALUATE ALL MODELS
# ==============================

evaluate_model(best_dt, "Decision Tree")

evaluate_model(best_rf, "Random Forest")

evaluate_model(best_xgb, "XGBoost")

evaluate_model(best_gb, "Gradient Boosting")


# ==========================================
# SAVE BEST MODEL
# ==========================================

# Here we assume Random Forest performed best
best_model = best_rf

joblib.dump(best_model, "pollution_source_model.pkl")

joblib.dump(encoder, "label_encoder.pkl")

print("\nModel saved successfully!")