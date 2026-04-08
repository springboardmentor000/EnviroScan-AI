import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os
import joblib

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 60)
print("  EnviroScan — Module 4: Model Training & Source Prediction")
print("=" * 60)

# -------------------------------------------------
# STEP 1: Load labeled data
# -------------------------------------------------
print("\n STEP 1: Loading labeled data...")
df = pd.read_csv("Hyderabad_pollution_labeled.csv")
print(f"   Rows: {len(df)}  |  Columns: {len(df.columns)}")

# -------------------------------------------------
# STEP 2: Prepare features and target
# -------------------------------------------------
print("\n  STEP 2: Preparing features and target variable...")

FEATURES = [
    "pm25", "pm10", "no2", "so2", "o3", "co",
    "temp", "humidity", "wind_speed", "wind_direction",
    "road_count", "has_major_road",
    "industrial_zone_count", "near_industrial_zone",
    "dump_site_count", "near_dump_site",
    "agricultural_count", "near_agricultural_area",
    "hour", "month"
]

TARGET = "pollution_source"

# Fill any remaining NaN with median
for col in FEATURES:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

X = df[FEATURES]
y = df[TARGET]

print(f"   Features : {len(FEATURES)}")
print(f"   Target   : {TARGET}")
print(f"   Classes  : {list(y.unique())}")

# -------------------------------------------------
# STEP 3: Encode target labels
# -------------------------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\n   Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# -------------------------------------------------
# STEP 4: Train / Test split (80/20)
# -------------------------------------------------
print("\n  STEP 4: Splitting dataset 80% train / 20% test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"   Training samples : {len(X_train)}")
print(f"   Testing samples  : {len(X_test)}")

# -------------------------------------------------
# STEP 5: Train models
# -------------------------------------------------
print("\n STEP 5: Training models...\n")

results = {}

# ── Random Forest ──
print("    Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_cv   = cross_val_score(rf, X, y_encoded, cv=5, scoring="accuracy").mean()
results["Random Forest"] = {
    "model": rf, "pred": rf_pred,
    "accuracy":  round(accuracy_score(y_test, rf_pred), 4),
    "precision": round(precision_score(y_test, rf_pred, average="weighted", zero_division=0), 4),
    "recall":    round(recall_score(y_test, rf_pred, average="weighted", zero_division=0), 4),
    "f1":        round(f1_score(y_test, rf_pred, average="weighted", zero_division=0), 4),
    "cv_score":  round(rf_cv, 4),
}
print(f"    Random Forest — Accuracy: {results['Random Forest']['accuracy']*100:.1f}%  CV: {rf_cv*100:.1f}%")

# ── XGBoost ──
print("\n    Training XGBoost...")
try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=100, random_state=42,
                        max_depth=6, learning_rate=0.1,
                        eval_metric="mlogloss", verbosity=0)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_cv   = cross_val_score(xgb, X, y_encoded, cv=5, scoring="accuracy").mean()
    results["XGBoost"] = {
        "model": xgb, "pred": xgb_pred,
        "accuracy":  round(accuracy_score(y_test, xgb_pred), 4),
        "precision": round(precision_score(y_test, xgb_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_test, xgb_pred, average="weighted", zero_division=0), 4),
        "f1":        round(f1_score(y_test, xgb_pred, average="weighted", zero_division=0), 4),
        "cv_score":  round(xgb_cv, 4),
    }
    print(f"    XGBoost — Accuracy: {results['XGBoost']['accuracy']*100:.1f}%  CV: {xgb_cv*100:.1f}%")
except ImportError:
    print("     XGBoost not installed — skipping (run: py -m pip install xgboost)")

# ── Decision Tree ──
print("\n    Training Decision Tree...")
dt = DecisionTreeClassifier(random_state=42, max_depth=8)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_cv   = cross_val_score(dt, X, y_encoded, cv=5, scoring="accuracy").mean()
results["Decision Tree"] = {
    "model": dt, "pred": dt_pred,
    "accuracy":  round(accuracy_score(y_test, dt_pred), 4),
    "precision": round(precision_score(y_test, dt_pred, average="weighted", zero_division=0), 4),
    "recall":    round(recall_score(y_test, dt_pred, average="weighted", zero_division=0), 4),
    "f1":        round(f1_score(y_test, dt_pred, average="weighted", zero_division=0), 4),
    "cv_score":  round(dt_cv, 4),
}
print(f"    Decision Tree — Accuracy: {results['Decision Tree']['accuracy']*100:.1f}%  CV: {dt_cv*100:.1f}%")

# -------------------------------------------------
# STEP 6: Compare model performance
# -------------------------------------------------
print("\n STEP 6: Model Performance Comparison\n")
print(f"   {'Model':<18} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'CV Score':>10}")
print(f"   {'-'*68}")
for name, r in results.items():
    print(f"   {name:<18} {r['accuracy']*100:>9.1f}% {r['precision']*100:>9.1f}% {r['recall']*100:>9.1f}% {r['f1']*100:>9.1f}% {r['cv_score']*100:>9.1f}%")

# -------------------------------------------------
# STEP 7: Best model
# -------------------------------------------------
best_name = max(results, key=lambda x: results[x]["f1"])
best      = results[best_name]
print(f"\n Best Model: {best_name} (F1: {best['f1']*100:.1f}%)")

# -------------------------------------------------
# STEP 8: Classification report for best model
# -------------------------------------------------
print(f"\n STEP 8: Classification Report — {best_name}\n")
print(classification_report(y_test, best["pred"], target_names=le.classes_))

# -------------------------------------------------
# STEP 9: Save all models
# -------------------------------------------------
print("\n STEP 9: Saving models...")
for name, r in results.items():
    fname = name.lower().replace(" ", "_")
    joblib.dump(r["model"], f"models/{fname}.pkl")
    print(f"    Saved → models/{fname}.pkl")

joblib.dump(le, "models/label_encoder.pkl")
print("    Saved → models/label_encoder.pkl")

# -------------------------------------------------
# STEP 10: Visualizations
# -------------------------------------------------
print("\n STEP 10: Generating visualizations...")

fig = plt.figure(figsize=(18, 12))
fig.suptitle("EnviroScan — Module 4: Model Training Results", fontsize=16, fontweight="bold")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

colors     = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6"]
bar_colors = ["#3498DB", "#E74C3C", "#2ECC71"]

# ── Plot 1: Accuracy comparison ──
ax1    = fig.add_subplot(gs[0, 0])
names  = list(results.keys())
accs   = [results[n]["accuracy"] * 100 for n in names]
bars   = ax1.bar(names, accs, color=bar_colors, edgecolor="white", width=0.5)
ax1.set_title("Model Accuracy Comparison", fontweight="bold")
ax1.set_ylabel("Accuracy (%)")
ax1.set_ylim(0, 110)
for bar, val in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{val:.1f}%", ha="center", fontweight="bold", fontsize=10)
ax1.tick_params(axis="x", rotation=10)

# ── Plot 2: F1 Score comparison ──
ax2  = fig.add_subplot(gs[0, 1])
f1s  = [results[n]["f1"] * 100 for n in names]
bars2 = ax2.bar(names, f1s, color=bar_colors, edgecolor="white", width=0.5)
ax2.set_title("F1-Score Comparison", fontweight="bold")
ax2.set_ylabel("F1-Score (%)")
ax2.set_ylim(0, 110)
for bar, val in zip(bars2, f1s):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{val:.1f}%", ha="center", fontweight="bold", fontsize=10)
ax2.tick_params(axis="x", rotation=10)

# ── Plot 3: CV Score comparison ──
ax3  = fig.add_subplot(gs[0, 2])
cvs  = [results[n]["cv_score"] * 100 for n in names]
bars3 = ax3.bar(names, cvs, color=bar_colors, edgecolor="white", width=0.5)
ax3.set_title("Cross-Validation Score (5-Fold)", fontweight="bold")
ax3.set_ylabel("CV Accuracy (%)")
ax3.set_ylim(0, 110)
for bar, val in zip(bars3, cvs):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{val:.1f}%", ha="center", fontweight="bold", fontsize=10)
ax3.tick_params(axis="x", rotation=10)

# ── Plot 4: Confusion matrix for best model ──
ax4 = fig.add_subplot(gs[1, 0:2])
cm  = confusion_matrix(y_test, best["pred"])
im  = ax4.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax4.set_title(f"Confusion Matrix — {best_name}", fontweight="bold")
ax4.set_xticks(range(len(le.classes_)))
ax4.set_yticks(range(len(le.classes_)))
ax4.set_xticklabels(le.classes_, rotation=20, fontsize=9)
ax4.set_yticklabels(le.classes_, fontsize=9)
ax4.set_xlabel("Predicted Label")
ax4.set_ylabel("True Label")
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax4.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=11, fontweight="bold")
plt.colorbar(im, ax=ax4)

# ── Plot 5: Feature importance (best model if RF or DT) ──
ax5 = fig.add_subplot(gs[1, 2])
if hasattr(best["model"], "feature_importances_"):
    importances = best["model"].feature_importances_
    indices     = np.argsort(importances)[::-1][:10]
    top_features = [FEATURES[i] for i in indices]
    top_values   = [importances[i] for i in indices]
    ax5.barh(top_features[::-1], top_values[::-1], color="#3498DB", edgecolor="white")
    ax5.set_title(f"Top 10 Feature Importance\n({best_name})", fontweight="bold")
    ax5.set_xlabel("Importance Score")

plt.savefig("outputs/module4_model_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved → outputs/module4_model_results.png")

# -------------------------------------------------
# STEP 11: Predict on real Hyderabad stations
# -------------------------------------------------
print("\n STEP 11: Predicting on real Hyderabad stations...\n")

real_df   = df[df["station_id"].astype(str).str.startswith("SIM") == False].copy()
X_real    = real_df[FEATURES].fillna(real_df[FEATURES].median())
real_pred = le.inverse_transform(best["model"].predict(X_real))
real_prob = best["model"].predict_proba(X_real).max(axis=1)

print(f"   {'Station':<45} {'Predicted Source':<16} {'Confidence'}")
print(f"   {'-'*75}")
for i, row in real_df.iterrows():
    idx  = list(real_df.index).index(i)
    pred = real_pred[idx]
    conf = round(real_prob[idx] * 100, 1)
    print(f"   {row['station_name'][:45]:<45} {pred:<16} {conf}%")

print(f"\n{'='*60}")
print(" Module 4 Complete!")
print("   Models saved in  : models/")
print("   Chart saved in   : outputs/module4_model_results.png")
print(f"{'='*60}")