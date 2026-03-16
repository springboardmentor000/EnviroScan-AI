# module_2_updated.py
import pandas as pd
from geopy.distance import geodesic
from datetime import datetime

print("Starting Module 2: Data Cleaning & Feature Engineering...")

# ==============================
# 1️⃣ Load Module 1 Dataset
# ==============================
df = pd.read_csv("final_dataset.csv")
print("✅ Loaded Module 1 dataset with", len(df), "rows")

# ==============================
# 2️⃣ Handle Missing Values (fix Copy-on-Write warning)
# ==============================
numeric_cols = ['PM2_5', 'PM10', 'NO2', 'CO', 'SO2', 'O3',
                'temperature_C', 'humidity_%', 'wind_speed_mps']

for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Fill missing categorical columns
if 'city' in df.columns:
    df['city'] = df['city'].fillna(df['city'].mode()[0])
if 'source' in df.columns:
    df['source'] = df['source'].fillna('Unknown')

# ==============================
# 3️⃣ Standardize Column Names
# ==============================
df.rename(columns={
    'latitude_air': 'latitude',
    'longitude_air': 'longitude',
    'city_air': 'city',
    'timestamp_air': 'timestamp'
}, inplace=True)

# ==============================
# 4️⃣ Extract Temporal Features
# ==============================
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek  # Monday=0
df['month'] = df['timestamp'].dt.month

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

df['season'] = df['month'].apply(get_season)

# ==============================
# 5️⃣ Compute Distances to Features
# ==============================
# Read physical features safely, skip bad lines
features = pd.read_csv("physical_features.csv", on_bad_lines='skip')

# Separate types
roads_df = features[features['highway'] == True] if 'highway' in features.columns else pd.DataFrame()
industry_df = features[features['landuse'] == 'industrial'] if 'landuse' in features.columns else pd.DataFrame()
farmland_df = features[features['landuse'] == 'farmland'] if 'landuse' in features.columns else pd.DataFrame()

def nearest_distance(point, features):
    if features.empty or 'latitude' not in features.columns or 'longitude' not in features.columns:
        return None
    distances = [geodesic(point, (row['latitude'], row['longitude'])).meters for idx, row in features.iterrows()]
    return min(distances)

df['dist_to_road'] = df.apply(lambda row: nearest_distance((row['latitude'], row['longitude']), roads_df), axis=1)
df['dist_to_industry'] = df.apply(lambda row: nearest_distance((row['latitude'], row['longitude']), industry_df), axis=1)
df['dist_to_farmland'] = df.apply(lambda row: nearest_distance((row['latitude'], row['longitude']), farmland_df), axis=1)

# ==============================
# 6️⃣ Save Cleaned & Feature-Rich Dataset
# ==============================
df.to_csv("module2_cleaned_features.csv", index=False)
print("✅ Module 2 dataset saved → module2_cleaned_features.csv")
print("Columns now include:")
print(df.columns.tolist())
print("Module 2 Completed Successfully!")