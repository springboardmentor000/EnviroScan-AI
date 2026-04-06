import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# ==============================
#  STYLE SETTINGS
# ==============================
plt.style.use('ggplot')
sns.set(font_scale=1.1)

# ==============================
#  PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "labeled_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "module4_graph.png")

os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

print("\n Module 4 Visualization Started")

# ==============================
#  LOAD DATA
# ==============================
df = pd.read_csv(DATA_PATH)

#  FIX: Include lat, lon
features = [
    'lat','lon',
    'pm2_5','pm10','no2','co','so2','o3',
    'temp','humidity','wind',
    'dist_road','dist_industry','dist_dump','dist_farm'
]

X = df[features]
y = df['pollution_source']

# Label Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ==============================
#  LOAD MODELS
# ==============================
models = {
    "Decision Tree": joblib.load(os.path.join(MODELS_DIR, "decision_tree.pkl")),
    "Random Forest": joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl")),
    "XGBoost": joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl"))
}

results = []

# ==============================
#  CALCULATE METRICS
# ==============================
for name, model in models.items():
    y_pred = model.predict(X)

    acc = accuracy_score(y_encoded, y_pred)
    f1 = f1_score(y_encoded, y_pred, average='weighted')
    cv = cross_val_score(model, X, y_encoded, cv=5, scoring='f1_weighted').mean()

    results.append({
        "model": name,
        "accuracy": acc,
        "f1": f1,
        "cv": cv
    })

results_df = pd.DataFrame(results)

print("\n Model Metrics:")
print(results_df)

# ==============================
#  COLORS
# ==============================
colors = ['#FF6B6B', '#4ECDC4', '#FFD93D']

# ==============================
#  LABEL FUNCTION
# ==============================
def add_labels(ax, values):
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v*100:.1f}%", ha='center',
                fontsize=11, fontweight='bold')

# ==============================
#  PLOTTING
# ==============================
plt.figure(figsize=(18,12))

#  Accuracy
ax1 = plt.subplot(2,3,1)
ax1.bar(results_df['model'], results_df['accuracy'], color=colors)
ax1.set_title("Model Accuracy", fontsize=14, fontweight='bold')
ax1.set_ylim(0, 1.1)
add_labels(ax1, results_df['accuracy'])

#  F1 Score
ax2 = plt.subplot(2,3,2)
ax2.bar(results_df['model'], results_df['f1'], color=colors)
ax2.set_title("F1 Score", fontsize=14, fontweight='bold')
ax2.set_ylim(0, 1.1)
add_labels(ax2, results_df['f1'])

#  CV Score
ax3 = plt.subplot(2,3,3)
ax3.bar(results_df['model'], results_df['cv'], color=colors)
ax3.set_title("Cross Validation Score", fontsize=14, fontweight='bold')
ax3.set_ylim(0, 1.1)
add_labels(ax3, results_df['cv'])

# ==============================
#  CONFUSION MATRIX
# ==============================
best_model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
y_pred_best = best_model.predict(X)

cm = confusion_matrix(y_encoded, y_pred_best)

plt.subplot(2,3,4)
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            linewidths=1, linecolor='black')
plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
plt.xlabel("Predicted")
plt.ylabel("Actual")

# ==============================
#  FEATURE IMPORTANCE
# ==============================
rf_model = models["Random Forest"]

if hasattr(rf_model.named_steps['model'], "feature_importances_"):
    importances = rf_model.named_steps['model'].feature_importances_

    plt.subplot(2,3,5)
    sns.barplot(x=importances, y=features)
    plt.title("Feature Importance", fontsize=14, fontweight='bold')

# ==============================
#  SAVE GRAPH
# ==============================
plt.suptitle("AI EnviroScan Model Performance Dashboard",
             fontsize=18, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_PATH)

print(f"\n Graph saved at: {OUTPUT_PATH}")
print("https://meet.google.com/qhu-gvnt-auy Module 4 Visualization Completed Successfully")