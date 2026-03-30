import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("enviro_data_with_distance.csv")

# ✅ ADD THIS
print(df["City"].unique())
print("Total cities:", df["City"].nunique())

# Drop missing values
df = df.fillna(0)

# -------------------------------
# ✅ SOURCE LABELING (IMPORTANT)
# -------------------------------
def label_source(row):
    if row["Dist_Industrial"] and row["Dist_Industrial"] < 1000 and row["SO2"] > 5:
        return "Industrial"
    elif row["Dist_Farmland"] and row["Dist_Farmland"] < 1000 and row["PM2.5"] > 20:
        return "Agricultural"
    elif row["Dist_Road"] and row["Dist_Road"] < 500 and row["NO2"] > 10:
        return "Vehicular"
    else:
        return "Natural"

df["Source"] = df.apply(label_source, axis=1)
# Save updated dataset with Source column
df.to_csv("enviro_data_with_distance.csv", index=False)

print("\nSample labeled data:")
print(df[["City", "Source"]])

# -------------------------------
# ✅ FEATURES (INPUT)
# -------------------------------
X = df[[
    "Dist_Road",
    "Dist_Industrial",
    "Dist_Farmland",
    "Temperature (°C)",
    "Humidity (%)",
    "Wind Speed (m/s)",
    "PM2.5",
    "PM10",
    "NO2",
    "SO2"
]]

# -------------------------------
# ✅ TARGET (OUTPUT)
# -------------------------------
y = df["Source"]

# Encode labels (text → numbers)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -------------------------------
# ✅ MODEL (Decision Tree)
# -------------------------------
model = DecisionTreeClassifier()
model.fit(X, y_encoded)

# Accuracy (on same data)
accuracy = model.score(X, y_encoded)
print("\nModel Accuracy:", accuracy)

# -------------------------------
# ✅ TEST PREDICTION
# -------------------------------
sample = X.iloc[0:1]
prediction = model.predict(sample)

print("\nPredicted Source:", le.inverse_transform(prediction)[0])