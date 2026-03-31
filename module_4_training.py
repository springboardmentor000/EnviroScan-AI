# ==============================
# MODULE 4: MODEL TRAINING (VS CODE VERSION)
# ==============================

import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(" Starting Module 4: Model Training\n")

try:
   
    file_path = "Infosys_dataset.csv"  

    if not os.path.exists(file_path):
        raise Exception(" labeled_dataset.csv not found in folder!")

    df = pd.read_csv(file_path)

    print(f" Dataset Loaded: {df.shape}")

  
    df = df.drop_duplicates()

  
    features = [
        "PM2_5", "PM10", "NO2", "CO", "SO2", "O3",
        "AQI_score",
        "temperature_C", "humidity_%", "wind_speed_mps"
    ]

 
    features = [col for col in features if col in df.columns]

    print(" Using Features:", features)

    
    target = "source_domain"

    if target not in df.columns:
        raise Exception(" source_domain column missing. Run Module 3!")


    df = df.dropna(subset=features + [target])

    X = df[features]
    y = df[target]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f" Training: {len(X_train)} | Testing: {len(X_test)}")

   
    print("\n🔍 Training Random Forest (with tuning)...")

    rf = RandomForestClassifier()

    params = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10]
    }

    grid = GridSearchCV(rf, params, cv=3)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    print(" Best Parameters:", grid.best_params_)

  
    print("\n Training Decision Tree...")

    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X_train, y_train)

    # ==============================
    # EVALUATION
    # ==============================
    print("\n MODEL EVALUATION (Random Forest):")

    preds = best_model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, preds))

    # ==============================
    # SAVE MODEL
    # ==============================
    joblib.dump(best_model, "pollution_model.pkl")

    print("\n Model saved as pollution_model.pkl")

except Exception as e:
    print(" ERROR:", e)

print("\n Module 4 Completed Successfully!")