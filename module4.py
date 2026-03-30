# ==========================================
# MODULE 4: FINAL SYSTEM (SMART INPUT)
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# OPTIONAL XGBOOST
try:
    from xgboost import XGBClassifier
    use_xgb = True
except:
    use_xgb = False


# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv("labeled_environment_dataset.csv")
df = df.drop_duplicates()

# REMOVE RARE CLASSES
df = df[df["source_label"].map(df["source_label"].value_counts()) > 2]

# ==============================
# ENCODE CITY
# ==============================

city_encoder = LabelEncoder()
df["city_encoded"] = city_encoder.fit_transform(df["city"])

# ==============================
# FEATURES
# ==============================

features = [
    "pm25","pm10","no2","co","so2","o3",
    "temperature","humidity","pressure","wind_speed",
    "dist_to_road","dist_to_industry","dist_to_dump",
    "city_encoded"
]

X = df[features]
y = df["source_label"]

# ENCODE TARGET
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ==============================
# SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ==============================
# MODELS
# ==============================

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42)
}

if use_xgb:
    models["XGBoost"] = XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="mlogloss"
    )

trained_models = {}

# ==============================
# TRAINING
# ==============================

for name, model in models.items():
    print(f"\n🔹 {name}")

    cv = cross_val_score(model, X_train, y_train, cv=5)
    print("CV Mean:", round(cv.mean(), 4))

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

    print(classification_report(y_test, y_pred))

    trained_models[name] = model

# ==============================
# BEST MODEL
# ==============================

best_model = max(trained_models, key=lambda m: accuracy_score(y_test, trained_models[m].predict(X_test)))
model = trained_models[best_model]

print("\n🔥 Best Model:", best_model)

# SAVE
joblib.dump(model, "pollution_source_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(city_encoder, "city_encoder.pkl")


# ==============================
# SMART USER INPUT (LESS INPUT)
# ==============================

print("\n🌍 ENTER KEY POLLUTION VALUES:")

def get_val(name):
    while True:
        try:
            val = float(input(f"{name} (0 to 1): "))
            if 0 <= val <= 1:
                return val
        except:
            pass
        print("❌ Invalid input")

pm25 = get_val("PM2.5")
no2 = get_val("NO2")
so2 = get_val("SO2")
co = get_val("CO")

dist_road = get_val("Distance to Road")
dist_industry = get_val("Distance to Industry")
dist_dump = get_val("Distance to Dump")

# AUTO-FILL OTHER FEATURES (AVERAGE VALUES)
pm10 = df["pm10"].mean()
o3 = df["o3"].mean()
temperature = df["temperature"].mean()
humidity = df["humidity"].mean()
pressure = df["pressure"].mean()
wind_speed = df["wind_speed"].mean()

# CITY
city_input = input("Enter City: ").strip().title()
try:
    city_encoded = city_encoder.transform([city_input])[0]
except:
    city_encoded = 0

# FINAL INPUT
user_data = [
    pm25, pm10, no2, co, so2, o3,
    temperature, humidity, pressure, wind_speed,
    dist_road, dist_industry, dist_dump,
    city_encoded
]

user_df = pd.DataFrame([user_data], columns=features)

# ==============================
# PREDICTION
# ==============================

pred = model.predict(user_df)
result = label_encoder.inverse_transform(pred)

print("\n🌟 Predicted Source:", result[0])

# ==============================
# ALERT SYSTEM
# ==============================

print("\n🚨 ALERTS:")

if pm25 > 0.7:
    print("🔴 High Pollution")
elif pm25 > 0.4:
    print("🟠 Moderate Pollution")
else:
    print("🟢 Safe")

if no2 > 0.6:
    print("⚠ Traffic Pollution Detected")

if so2 > 0.6:
    print("⚠ Industrial Pollution Detected")

# ==============================
# SUGGESTIONS
# ==============================

print("\n💡 SUGGESTIONS:")

if result[0] == "Vehicular":
    print("Reduce vehicle usage")

elif result[0] == "Industrial":
    print("Control industrial emissions")

elif result[0] == "Burning":
    print("Avoid waste burning")

elif result[0] == "Agricultural":
    print("Avoid stubble burning")

else:
    print("Environment is stable")