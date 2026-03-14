import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier
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

    # Base features
    features = [
        "pm2_5","pm10","no2","o3","so2","co",
        "temperature_c","humidity","pressure_hpa","wind_speed_ms","wind_direction"  
    ]

    # Keep only existing features
    features = [f for f in features if f in df.columns]
    X = df[features]
    y = df[target]
    print("Features used:", features)
    return X, y, features

# ==========================================
# TRAIN-TEST SPLIT 
# ==========================================
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    smote = SMOTE(k_neighbors=3,random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("After SMOTE:")
    print(y_train.value_counts())
    return X_train, X_test, y_train, y_test

# ==========================================
# RANDOM FOREST TRAIN
# ==========================================
def train_random_forest(X_train, y_train):
    model = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)   # ← THIS LINE IS REQUIRED

    return model
# ==========================================
# HYPERPARAMETER TUNING
# ==========================================

def tune_random_forest(X_train, y_train):

    rf = ExtraTreesClassifier(
        random_state=42,
        class_weight="balanced"
    )

    param_grid = {
        "n_estimators": [200,300,400],
        "max_depth": [10,15],
        "min_samples_split": [5,7],
        "min_samples_leaf": [2,5]
    }

    grid = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)

    return grid.best_estimator_

# ==========================================
# MODEL EVALUATION
# ==========================================
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print("\nAccuracy:", round(accuracy_score(y_test, predictions)*100,2), "%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    return accuracy_score(y_test, predictions)

# ==========================================
# CROSS VALIDATION 
# ==========================================
from sklearn.model_selection import StratifiedKFold

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
# FEATURE IMPORTANCE
# ==========================================
def show_feature_importance(model, features):
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nFeature Importance:\n", importance)
    return importance

def plot_feature_importance(model, features):
    importance = pd.Series(model.feature_importances_, index=features).sort_values()
    plt.figure(figsize=(8,6))
    importance.plot(kind="barh")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance Score")
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/RF_feature_importance.png", bbox_inches="tight")
    plt.close()
    print("RF_Feature importance plot saved")

# ==========================================
# VISUALIZATIONS & BAGGING
# ==========================================
def visualize_random_forest_tree(model, features, max_depth=3):
    tree = model.estimators_[0]
    plt.figure(figsize=(20,10))
    plot_tree(tree, feature_names=features, class_names=model.classes_,
              filled=True, rounded=True, max_depth=max_depth)
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/random_forest_tree.png", bbox_inches="tight")
    plt.close()
    print("Random forest tree saved")

def demonstrate_bagging(X_train, y_train):
    bagging_model = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=10,
        bootstrap=True,
        random_state=42
    )
    bagging_model.fit(X_train, y_train)
    print("Bagging model trained with", len(bagging_model.estimators_), "trees")

# ==========================================
# CONFUSION MATRIX
# ==========================================
def plot_confusion_matrix(model, X_test, y_test):
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/RF_confusion_matrix.png", bbox_inches="tight")
    plt.close()
    print(" RF_Confusion matrix saved")

# ==========================================
# SAVE & LOAD MODEL
# ==========================================
def save_model(model, filename):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, os.path.join("models", filename))
    print("Model saved at models/", filename)

def load_model(filename):
    model = joblib.load(os.path.join("models", filename))
    print("Model loaded successfully")
    return model

# ==========================================
# SAVE MODEL REPORT
# ==========================================
def save_model_report(model, X_test, y_test):
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    os.makedirs("models", exist_ok=True)
    with open("models/model_report.txt", "w") as f:
        f.write(report)
    print("Model report saved")

# ==========================================
# MAIN PIPELINE
# ==========================================
if __name__ == "__main__":

    df = load_dataset("vijayawada_labelled_dataset.csv")
    X, y, features = prepare_features(df)

    print("\nDataset Gini Index:", gini_index(y))

    X_train, X_test, y_train, y_test = split_data(X, y)

    # ======================================
    # BEFORE TUNING
    # ======================================
    print("\n===== Random Forest BEFORE Tuning =====")

    base_model = train_random_forest(X_train, y_train)

    base_test_acc = evaluate_model(base_model, X_test, y_test)

    # TRAIN ACCURACY BEFORE TUNING
    base_train_pred = base_model.predict(X_train)
    base_train_acc = accuracy_score(y_train, base_train_pred)

    print("\nTrain Accuracy:", round(base_train_acc*100,2), "%")
    print("Test Accuracy:", round(base_test_acc*100,2), "%")
    # ======================================
    # AFTER TUNING
    # ======================================
    print("\n===== Random Forest AFTER Tuning =====")

    tuned_model = tune_random_forest(X_train, y_train)

    tuned_test_acc = evaluate_model(tuned_model, X_test, y_test)

    # TRAIN ACCURACY AFTER TUNING
    tuned_train_pred = tuned_model.predict(X_train)
    tuned_train_acc = accuracy_score(y_train, tuned_train_pred)

    print("\nTrain Accuracy:", round(tuned_train_acc*100,2), "%")
    print("Test Accuracy:", round(tuned_test_acc*100,2), "%")
    # ======================================
    # CROSS VALIDATION
    # ======================================
    cross_validate_model(tuned_model, X, y)
    # ======================================
    # FEATURE IMPORTANCE
    # ======================================
    show_feature_importance(tuned_model, features)
    plot_feature_importance(tuned_model, features)
    # ======================================
    # VISUALIZATIONS
    # ======================================
    visualize_random_forest_tree(tuned_model, features)
    demonstrate_bagging(X_train, y_train)
    plot_confusion_matrix(tuned_model, X_test, y_test)
    # ======================================
    # SAVE MODEL
    # ======================================
    save_model(tuned_model, "random_forest_pollution_model.pkl")
    save_model_report(tuned_model, X_test, y_test)