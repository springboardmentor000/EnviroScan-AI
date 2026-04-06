import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ==============================
#  PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "labeled_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

print("\n Module 4: Model Training Started")

# ==============================
#  LOAD DATA
# ==============================
df = pd.read_csv(DATA_PATH)

# ==============================
#  FEATURES & TARGET
# ==============================
features = [
    'lat','lon',
    'pm2_5','pm10','no2','co','so2','o3',
    'temp','humidity','wind',
    'dist_road','dist_industry','dist_dump','dist_farm'
]

X = df[features]
y = df['pollution_source']

# ==============================
#  LABEL ENCODING
# ==============================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ==============================
#  TRAIN-TEST SPLIT (80/20)
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"Training Data: {X_train.shape}")
print(f"Testing Data: {X_test.shape}")

# ==============================
#  TRAIN FUNCTION
# ==============================
def train_model(name, model, params):
    print(f"\n🔹 Training {name}...")

    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),  # normalization
        ('model', model)
    ])

    grid = GridSearchCV(
        pipeline,
        params,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # Predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    cv_score = cross_val_score(
        best_model, X, y_encoded,
        cv=5, scoring='f1_weighted'
    ).mean()

    print(f" Best Params: {grid.best_params_}")
    print(f" Train Accuracy: {train_acc:.4f}")
    print(f" Test Accuracy: {test_acc:.4f}")
    print(f" CV Score: {cv_score:.4f}")

    print("\n Classification Report:")
    print(classification_report(
        y_test, y_test_pred,
        target_names=le.classes_,
        zero_division=0
    ))

    print(" Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    # Save model
    model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(best_model, model_path)
    print(f" Saved: {model_path}")

    return best_model, test_acc, cv_score


# ==============================
#  DECISION TREE (CONTROL OVERFITTING)
# ==============================
dt_params = {
    'model__max_depth': [3, 4],
    'model__min_samples_split': [10, 15, 20],
    'model__min_samples_leaf': [4, 6, 8],
    'model__class_weight': ['balanced']
}

dt_model, dt_acc, dt_cv = train_model(
    "decision_tree",
    DecisionTreeClassifier(random_state=42),
    dt_params
)

# ==============================
#  RANDOM FOREST
# ==============================
rf_params = {
    'model__n_estimators': [120, 150],
    'model__max_depth': [4, 5],
    'model__min_samples_split': [10],
    'model__min_samples_leaf': [4, 6],
    'model__max_features': ['sqrt'],
    'model__class_weight': ['balanced']
}

rf_model, rf_acc, rf_cv = train_model(
    "random_forest",
    RandomForestClassifier(random_state=42),
    rf_params
)

# ==============================
#  XGBOOST
# ==============================
xgb_params = {
    'model__n_estimators': [80, 100],
    'model__max_depth': [2, 3],
    'model__learning_rate': [0.05],
    'model__subsample': [0.7],
    'model__colsample_bytree': [0.7],
    'model__reg_alpha': [1, 2],
    'model__reg_lambda': [1, 2]
}

xgb_model, xgb_acc, xgb_cv = train_model(
    "xgboost",
    XGBClassifier(
        objective='multi:softmax',
        num_class=len(le.classes_),
        eval_metric='mlogloss',
        random_state=42
    ),
    xgb_params
)

# ==============================
#  SELECT BEST MODEL (CV BASED)
# ==============================
models = {
    "decision_tree": (dt_model, dt_cv),
    "random_forest": (rf_model, rf_cv),
    "xgboost": (xgb_model, xgb_cv)
}

best_name = max(models, key=lambda x: models[x][1])
best_model = models[best_name][0]

print(f"\n Best Model: {best_name}")

# Save best model
best_path = os.path.join(MODELS_DIR, "best_model.pkl")
joblib.dump(best_model, best_path)
print(f" Best Model Saved: {best_path}")

# ==============================
#  SAVE PREDICTIONS
# ==============================
y_pred = best_model.predict(X)
df['predicted_source'] = le.inverse_transform(y_pred)

output_path = os.path.join(OUTPUTS_DIR, "predictions.csv")
df.to_csv(output_path, index=False)

print(f" Predictions saved: {output_path}")

print("\n Module 4 Completed Successfully")