import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from math import radians, sin, cos, sqrt, atan2

# ==========================================
# STEP 1: LOAD DATA
# ==========================================
df = pd.read_csv("vijayawada_final_dataset.csv")

print("Initial Shape:", df.shape)          
print("Columns      :", df.columns.tolist())
print(df.head())

# ==========================================
# STEP 2: REMOVE DUPLICATES & INVALID RECORDS
# ==========================================
before = len(df)
df = df.drop_duplicates()
df.info()

# Remove negative pollutant values
pollutants = ["pm2_5", "pm10", "no2", "so2", "co", "o3"]
for col in pollutants:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan          # mark negatives as NaN

print("After cleaning:", df.shape)

# ==========================================
# STEP 3: STANDARDIZE TIMESTAMPS & GPS
# ==========================================
df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", dayfirst=True)
df["timestamp"] = df["timestamp"].dt.round("15min")  # standardize to 15-min intervals

df["lat"] = df["lat"].round(5)
df["lon"] = df["lon"].round(5)

df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)
print("\nTimestamps & GPS standardized.")

# ==========================================
# STEP 4: HANDLE MISSING VALUES
# ==========================================
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

# Linear interpolation per station
df[numeric_cols] = (
    df.groupby("location_id")[numeric_cols]
    .transform(lambda g: g.interpolate(method="linear", limit=4))
)

# Remaining NaN → mean per station
df[numeric_cols] = (
    df.groupby("location_id")[numeric_cols]
    .transform(lambda g: g.fillna(g.mean()))
)

# Final fallback → global mean
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

print("Missing values handled.")
print("NaN remaining:\n", df[numeric_cols].isna().sum()[df[numeric_cols].isna().sum() > 0])

# ==========================================
# STEP 5: NORMALIZE NUMERICAL FEATURES
# ==========================================
scaler = MinMaxScaler()

columns_to_scale = [
    "pm2_5", "pm10", "no2", "so2", "co", "o3",
    "temperature_c", "humidity", "pressure_hpa",
    "wind_speed_ms", "wind_direction"
]
existing_cols = [col for col in columns_to_scale if col in df.columns]

df[existing_cols] = scaler.fit_transform(df[existing_cols])
print(f"\nNormalized {len(existing_cols)} columns.")

# ==========================================
# STEP 6: SPATIAL FEATURES
# ==========================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

CITY_CENTER   = (16.5062, 80.6480)
INDUSTRY_REF  = (16.5000, 80.6500)
ROAD_REF      = (16.5065, 80.6450)
DUMP_REF      = (16.4950, 80.6600)

df["dist_to_center_km"]   = df.apply(lambda r: haversine(r["lat"], r["lon"], *CITY_CENTER),  axis=1)
df["dist_to_industry_km"] = df.apply(lambda r: haversine(r["lat"], r["lon"], *INDUSTRY_REF), axis=1)
df["dist_to_road_km"]     = df.apply(lambda r: haversine(r["lat"], r["lon"], *ROAD_REF),     axis=1)
df["dist_to_dump_km"]     = df.apply(lambda r: haversine(r["lat"], r["lon"], *DUMP_REF),     axis=1)

print("Spatial features added.")

# ==========================================
# STEP 7: TEMPORAL FEATURES
# ==========================================
df["hour"]         = df["timestamp"].dt.hour
df["day_of_week"]  = df["timestamp"].dt.dayofweek   # 0=Mon, 6=Sun
df["day_of_year"]  = df["timestamp"].dt.dayofyear
df["month"]        = df["timestamp"].dt.month
df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
df["is_rush_hour"] = df["hour"].apply(lambda h: 1 if (7<=h<=10 or 17<=h<=20) else 0)
df["is_dry_season"]= df["month"].isin([11, 12, 1, 2]).astype(int)

def get_season(month):
    if month in [12, 1, 2]:     return "Winter"
    elif month in [3, 4, 5]:    return "Summer"
    elif month in [6, 7, 8, 9]: return "Monsoon"
    else:                       return "PostMonsoon"

df["season"]    = df["month"].apply(get_season)
df["season"] = df["season"].astype("category").cat.codes

# ================================
# FEATURE ENGINEERING
# ================================

df["traffic_indicator"] = df["hour"] * df["dist_to_road_km"]

df["industry_weather"] = df["dist_to_industry_km"] * df["wind_speed_ms"]

df["dispersion_index"] = df["wind_speed_ms"] / (df["humidity"] + 1)

# ==========================================
# STEP 8: FINAL DATASET CHECK
# ==========================================
print("\n── Final Dataset Info ──")
print(df.info())

print("\n── Statistical Summary ──")
print(df.describe())

print("\n── Label Distribution ──")
if "pollution_source" in df.columns:
    print(df["pollution_source"].value_counts().to_string())

print("\n── Preview ──")
print(df.head())
print(df.tail())

# ==========================================
# STEP 9: SAVE
# ==========================================
df.to_csv("vijayawada_feature_engineering.csv", index=False)
print("Months:", sorted(df["timestamp"].dt.month.unique()))
print("Seasons:", df["season"].unique())
print(f"\n✅ Saved → vijayawada_feature_engineering.csv")
print(f"   Shape   : {df.shape}")
print(f"   Columns : {df.columns.tolist()}")
