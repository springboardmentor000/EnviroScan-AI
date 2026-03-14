import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import numpy as np

# ==========================================
# LOAD DATASET
# ==========================================
def load_dataset(path):
    df = pd.read_csv(path)
    print("Dataset Loaded:", df.shape)
    return df

# ==========================================
# GINI INDEX
# ==========================================
def gini_index(y):
    class_counts = y.value_counts()
    total = len(y)
    gini = 1.0
    for count in class_counts:
        prob = count / total
        gini -= prob ** 2
    return gini

# ==========================================
# FEATURE ENGINEERING
# ==========================================
def prepare_features(df):
    target = "pollution_source"
    features = [
        "pm2_5","pm10","no2","o3","so2","co",
        "temperature_c","humidity","pressure_hpa","wind_speed_ms","wind_direction"
    ]

    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df[target]

    print("Features used:", features)

    return X, y, features

# ==========================================
# TRAIN TEST SPLIT + SMOTE
# ==========================================
def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(k_neighbors=3, random_state=42)

    X_train, y_train = smote.fit_resample(X_train, y_train)

    print("After SMOTE:")
    print(pd.Series(y_train).value_counts())

    return X_train, X_test, y_train, y_test


# ==========================================
# LABEL ENCODING
# ==========================================
def encode_labels(y_train, y_test):

    le = LabelEncoder()

    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    return y_train_enc, y_test_enc, le


# ==========================================
# TRAIN XGBOOST
# ==========================================
def train_xgboost(X_train, y_train, num_class):

    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=num_class,
        n_estimators=300,
        max_depth=1,
        learning_rate=0.1,
        eval_metric='mlogloss',
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


# ==========================================
# HYPERPARAMETER TUNING
# ==========================================
def tune_xgboost(X_train, y_train, num_class):

    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=num_class,
        eval_metric='mlogloss',
        random_state=42
    )

    param_grid = {

        "n_estimators":[200,300,400],
        "max_depth":[1],
        "learning_rate":[0.05,0.1,0.2]

    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)

    return grid.best_estimator_


# ==========================================
# MODEL EVALUATION (TEST REPORT ONLY)
# ==========================================
def evaluate_model(model, X_train, y_train, X_test, y_test, le):

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print("\nAccuracy:", round(test_acc*100,2), "%")

    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=le.classes_))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_pred))

    print("\nTrain Accuracy:", round(train_acc*100,2), "%")
    print("Test Accuracy:", round(test_acc*100,2), "%")

    return train_acc, test_acc


# ==========================================
# CROSS VALIDATION
# ==========================================
def cross_validate_model(model, X, y):

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=skf, scoring="f1_weighted")

    print("\nCross Validation F1 Scores:", scores)
    print("Average F1:", scores.mean())


# ==========================================
# FEATURE IMPORTANCE
# ==========================================
def plot_feature_importance(model, features):

    importance = pd.Series(model.feature_importances_, index=features).sort_values()

    plt.figure(figsize=(8,6))

    importance.plot(kind="barh")

    plt.title("XGBoost Feature Importance")
    plt.xlabel("Importance Score")

    os.makedirs("plots", exist_ok=True)

    plt.savefig("plots/xgb_feature_importance.png", bbox_inches="tight")

    plt.close()

    print("Feature importance plot saved.")


# ==========================================
# CONFUSION MATRIX
# ==========================================
def plot_confusion_matrix(model, X_test, y_test, le):

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    os.makedirs("plots", exist_ok=True)

    plt.savefig("plots/xgb_confusion_matrix.png", bbox_inches="tight")

    plt.close()

    print("Confusion matrix plot saved.")


# ==========================================
# SAVE MODEL
# ==========================================
def save_model(model, filename):

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, os.path.join("models", filename))

    print("Model saved at models/", filename)


# ==========================================
# MAIN PIPELINE
# ==========================================
if __name__ == "__main__":

    df = load_dataset("vijayawada_labelled_dataset.csv")

    X, y, features = prepare_features(df)

    print("\nDataset Gini Index:", gini_index(y))

    X_train, X_test, y_train, y_test = split_data(X, y)

    y_train_enc, y_test_enc, le = encode_labels(y_train, y_test)

    num_classes = len(le.classes_)


    # BEFORE TUNING
    print("\n===== XGBoost BEFORE Tuning =====")

    base_model = train_xgboost(X_train, y_train_enc, num_classes)

    evaluate_model(base_model, X_train, y_train_enc, X_test, y_test_enc, le)


    # AFTER TUNING
    print("\n===== XGBoost AFTER Tuning =====")

    tuned_model = tune_xgboost(X_train, y_train_enc, num_classes)

    evaluate_model(tuned_model, X_train, y_train_enc, X_test, y_test_enc, le)


    # CROSS VALIDATION
    cross_validate_model(tuned_model, X, le.transform(y))


    # FEATURE IMPORTANCE
    plot_feature_importance(tuned_model, features)


    # CONFUSION MATRIX
    plot_confusion_matrix(tuned_model, X_test, y_test_enc, le)


    # SAVE MODEL
    save_model(tuned_model, "xgboost_pollution_model.pkl")