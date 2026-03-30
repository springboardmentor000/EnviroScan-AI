import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("../../../data/processed/final_dataset.csv")

# -----------------------------
# Split features & target
# -----------------------------
X = df.drop(columns=["pollution_source", "city"], errors="ignore")
y = df["pollution_source"]

# Keep only numeric columns (safety)
X = X.select_dtypes(include=["int64", "float64"])

# -----------------------------
# Same train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("../../../model/final_model.pkl")

# -----------------------------
# Make predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation Metrics
# -----------------------------
print("\nModel Evaluation Results\n")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))