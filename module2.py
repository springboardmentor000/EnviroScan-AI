# ==========================================
# MODULE 2: DATA CLEANING & FEATURE ENGINEERING (FINAL)
# ==========================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv("dataset.csv")

print("Original Shape:", df.shape)
print("Columns:", df.columns)

# ==============================
# REMOVE DUPLICATES
# ==============================

df = df.drop_duplicates()

# ==============================
# HANDLE MISSING VALUES
# ==============================

df = df.interpolate()
df = df.fillna(df.median(numeric_only=True))

# ==============================
# CONVERT TIMESTAMP
# ==============================

df["timestamp"] = pd.to_datetime(df["timestamp"])

# ==============================
# TEMPORAL FEATURES
# ==============================

df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["day_of_week"] = df["timestamp"].dt.dayofweek

# ==============================
# REALISTIC SEASON LOGIC (IMPORTANT)
# ==============================

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8]:
        return "Monsoon"
    else:
        return "Post-Monsoon"

df["season"] = df["month"].apply(get_season)

# ==============================
# DATA VALIDATION (IMPORTANT)
# ==============================

# Remove negative pollution values (if any)
pollution_cols = ["pm25", "pm10", "no2", "co", "so2", "o3"]

for col in pollution_cols:
    df = df[df[col] >= 0]

# ==============================
# FEATURE SCALING
# ==============================

scale_cols = [
    "pm25","pm10","no2","co","so2","o3",
    "temperature","humidity","pressure","wind_speed",
    "dist_to_road","dist_to_industry","dist_to_dump"
]

scaler = MinMaxScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# ==============================
# FINAL CHECK
# ==============================

print("Cleaned Shape:", df.shape)
print(df.head())

# ==============================
# SAVE CLEAN DATASET
# ==============================

df.to_csv("clean_environment_dataset.csv", index=False)

print("✅ Clean dataset saved successfully!")