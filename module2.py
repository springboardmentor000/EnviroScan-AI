# ==========================================
# MODULE 2: DATA CLEANING & FEATURE ENGINEERING
# ==========================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv("dataset.csv")

print("Original Dataset Shape:", df.shape)


# ==============================
# REMOVE UNWANTED COLUMNS
# ==============================

# REMOVE UNWANTED COLUMNS
df = df.drop(columns=["visibility","aqi"], errors="ignore")



# ==============================
# REMOVE DUPLICATE RECORDS
# ==============================

df = df.drop_duplicates()


# ==============================
# HANDLE MISSING VALUES
# ==============================

df = df.interpolate()

df = df.fillna(df.median(numeric_only=True))


# ==============================
# STANDARDIZE TIMESTAMP
# ==============================

df["timestamp"] = pd.to_datetime(df["timestamp"])


# ==============================
# TEMPORAL FEATURES
# ==============================

df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["season"] = df["month"] % 12 // 3 + 1


# ==============================
# NORMALIZE POLLUTION + WEATHER
# ==============================

cols = [
    "pm25","pm10","no2","co","so2","o3",
    "temperature","humidity","pressure","wind_speed"
]

scaler = MinMaxScaler()

df[cols] = scaler.fit_transform(df[cols])


# ==============================
# SPATIAL PROXIMITY FEATURES
# ==============================

# Example reference locations
road_lat, road_lon = 17.70, 83.21
industry_lat, industry_lon = 17.68, 83.20
dump_lat, dump_lon = 17.69, 83.22


def distance(lat1, lon1, lat2, lon2):
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)


df["dist_to_road"] = distance(df["latitude"], df["longitude"], road_lat, road_lon)

df["dist_to_industry"] = distance(df["latitude"], df["longitude"], industry_lat, industry_lon)

df["dist_to_dump"] = distance(df["latitude"], df["longitude"], dump_lat, dump_lon)


# ==============================
# FINAL DATASET
# ==============================

print("Clean Dataset Shape:", df.shape)

print(df.head())


# ==============================
# SAVE CLEAN DATASET
# ==============================

df.to_csv("clean_environment_dataset.csv", index=False)

print("Clean dataset saved successfully!")