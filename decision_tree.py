import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import plot_tree, DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV, learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE


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
# FEATURES
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
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    smote = SMOTE(k_neighbors=5, random_state=42)

    X_train, y_train = smote.fit_resample(X_train, y_train)

    print("After SMOTE:")

    print(y_train.value_counts())

    return X_train, X_test, y_train, y_test


# ==========================================
# BASE DECISION TREE
# ==========================================
def train_decision_tree(X_train, y_train):

    model = DecisionTreeClassifier(
        random_state=40,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    print("Decision Tree trained")

    return model


# ==========================================
# HYPERPARAMETER TUNING
# ==========================================
def tune_decision_tree(X_train, y_train):

    dt = DecisionTreeClassifier(random_state=42, class_weight="balanced")

    param_dist = {
        "criterion": ["gini", "entropy"],
        "max_depth": [4,5,6,7],
        "min_samples_split": [5, 10, 15],
        "min_samples_leaf": [2, 3, 5]
    }

    random_search = RandomizedSearchCV(
        dt,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring="f1_weighted",
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    print("Best Parameters:", random_search.best_params_)

    return random_search.best_estimator_


# ==========================================
# MODEL EVALUATION
# ==========================================
def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)

    print("\nAccuracy:", round(acc*100,2), "%")

    print("\nClassification Report:")

    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")

    print(confusion_matrix(y_test, predictions))

    return acc


# ==========================================
# CROSS VALIDATION
# ==========================================
def cross_validate_model(model, X, y):

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(
        model,
        X,
        y,
        cv=skf,
        scoring="f1_weighted"
    )

    print("\nCross Validation F1 Scores:", scores)

    print("Average F1:", scores.mean())

# ==========================================
# FEATURE IMPORTANCE PLOT FOR DECISION TREE
# ==========================================
def plot_decision_tree_feature_importance(model, features):
    importance = pd.Series(model.feature_importances_, index=features).sort_values()
    
    plt.figure(figsize=(8,6))
    importance.plot(kind="barh", color="skyblue")
    plt.title("Decision Tree Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/decision_tree_feature_importance.png", bbox_inches="tight")
    plt.show()
    print("Decision Tree feature importance plot saved")
# ==========================================
# TREE RULES
# ==========================================
def print_tree_rules(model, features):

    rules = export_text(model, feature_names=features)

    print("\nDecision Tree Rules:\n")

    print(rules)

# ==========================================
# TREE VISUALIZATION
# ==========================================
def visualize_decision_tree(model, features):

    plt.figure(figsize=(20,10))

    plot_tree(
        model,
        feature_names=features,
        class_names=model.classes_,
        filled=True,
        rounded=True,
        max_depth=4
    )

    os.makedirs("plots", exist_ok=True)

    plt.savefig("plots/decision_tree.png", bbox_inches="tight")

    plt.close()

    print("Decision Tree saved")

# ==========================================
# CONFUSION MATRIX
# ==========================================
def plot_confusion_matrix(model, X_test, y_test):

    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    os.makedirs("plots", exist_ok=True)

    plt.savefig("plots/confusion_matrix_dt.png")

    plt.close()

    print("Confusion matrix saved")


# ==========================================
# SAVE MODEL
# ==========================================
def save_model(model, filename):

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, os.path.join("models", filename))

    print("Model saved")


# ==========================================
# MAIN PIPELINE
# ==========================================
if __name__ == "__main__":

    df = load_dataset("vijayawada_labelled_dataset.csv")

    X, y, features = prepare_features(df)

    print("\nDataset Gini Index:", gini_index(y))

    X_train, X_test, y_train, y_test = split_data(X, y)

    # BEFORE TUNING
    print("\n===== Decision Tree BEFORE Tuning =====")

    base_model = train_decision_tree(X_train, y_train)

    train_pred = base_model.predict(X_train)
    test_pred = base_model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print("\nTrain Accuracy:", round(train_acc*100,2), "%")
    print("Test Accuracy:", round(test_acc*100,2), "%")

    evaluate_model(base_model, X_test, y_test)

    # AFTER TUNING
    print("\n===== Decision Tree AFTER Tuning =====")

    tuned_model = tune_decision_tree(X_train, y_train)

    train_pred = tuned_model.predict(X_train)
    test_pred = tuned_model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print("\nTrain Accuracy:", round(train_acc*100,2), "%")
    print("Test Accuracy:", round(test_acc*100,2), "%")

    evaluate_model(tuned_model, X_test, y_test)
    cross_validate_model(tuned_model, X, y)
    plot_decision_tree_feature_importance(tuned_model, features)
    visualize_decision_tree(tuned_model, features)
    print_tree_rules(tuned_model, features)
    plot_confusion_matrix(tuned_model, X_test, y_test)
    save_model(tuned_model, "decision_tree_pollution_model.pkl")