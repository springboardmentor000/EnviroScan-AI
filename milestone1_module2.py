import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 55)
print("  EnviroScan — Module 2: Data Cleaning & Feature Engineering")
print("=" * 55)

# -------------------------------------------------
# STEP 1: Load raw data
# -------------------------------------------------
print("\n STEP 1: Loading raw data...")

df = pd.read_csv("Hyderabad_pollution_latest.csv")
print(f"   Rows: {len(df)}  |  Columns: {len(df.columns)}")
print(f"   Columns: {list(df.columns)}")


# -------------------------------------------------
# STEP 2: Remove duplicates
# -------------------------------------------------
print("\n STEP 2: Removing duplicates...")

before = len(df)
df.drop_duplicates(subset=["station_id"], keep="first", inplace=True)
after = len(df)
print(f"   Removed {before - after} duplicate rows. Remaining: {after}")


# -------------------------------------------------
# STEP 3: Drop bc column (no station reports it)
# -------------------------------------------------
print("\n  STEP 3: Dropping 'bc' column (no data available)...")

if "bc" in df.columns:
    df.drop(columns=["bc"], inplace=True)
    print("   'bc' column dropped.")


# -------------------------------------------------
# STEP 4: Fix timestamp format
# -------------------------------------------------
print("\n STEP 4: Fixing timestamp...")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
print(f"   Timestamp sample: {df['timestamp'].iloc[0]}")


# -------------------------------------------------
# STEP 5: Handle missing pollutant values
# -------------------------------------------------
print("\n STEP 5: Handling missing pollutant values...")

pollutants = ["pm25", "pm10", "no2", "so2", "o3", "co"]

print("\n   Missing values BEFORE cleaning:")
for col in pollutants:
    missing = df[col].isna().sum()
    print(f"   {col:>6}: {missing} missing")

# Fill missing with median of available values
for col in pollutants:
    median_val = df[col].median()
    missing_count = df[col].isna().sum()
    df[col].fillna(round(median_val, 2), inplace=True)
    if missing_count > 0:
        print(f"    Filled {missing_count} missing '{col}' values with median: {round(median_val, 2)}")

print("\n   Missing values AFTER cleaning:")
for col in pollutants:
    print(f"   {col:>6}: {df[col].isna().sum()} missing")


# -------------------------------------------------
# STEP 6: Normalize pollutant + weather values (0 to 1)
# -------------------------------------------------
print("\n STEP 6: Normalizing pollutant and weather values...")

cols_to_normalize = ["pm25", "pm10", "no2", "so2", "o3", "co",
                     "temp", "humidity", "pressure", "wind_speed"]

for col in cols_to_normalize:
    min_val = df[col].min()
    max_val = df[col].max()
    if max_val - min_val > 0:
        df[f"{col}_norm"] = round((df[col] - min_val) / (max_val - min_val), 4)
    else:
        df[f"{col}_norm"] = 0.0

print(f"    Created {len(cols_to_normalize)} normalized columns (suffix: _norm)")


# -------------------------------------------------
# STEP 7: Derive temporal features from timestamp
# -------------------------------------------------
print("\n STEP 7: Deriving temporal features...")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"]       = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek    # 0=Monday, 6=Sunday
df["month"]      = df["timestamp"].dt.month

# Season based on month (India)
def get_season(month):
    if month in [12, 1, 2]:   return "Winter"
    elif month in [3, 4, 5]:  return "Summer"
    elif month in [6, 7, 8, 9]: return "Monsoon"
    else:                      return "Post-Monsoon"

df["season"] = df["month"].apply(get_season)
print(f"    Added: hour, day_of_week, month, season")
print(f"   Current season: {df['season'].iloc[0]}")


# -------------------------------------------------
# STEP 8: Calculate AQI category per station
# -------------------------------------------------
print("\n  STEP 8: Calculating AQI category from pm25...")

def aqi_category(pm25):
    if pm25 <= 30:    return "Good"
    elif pm25 <= 60:  return "Satisfactory"
    elif pm25 <= 90:  return "Moderate"
    elif pm25 <= 120: return "Poor"
    elif pm25 <= 250: return "Very Poor"
    else:             return "Severe"

df["aqi_category"] = df["pm25"].apply(aqi_category)
print("   AQI distribution:")
print(df["aqi_category"].value_counts().to_string())


# -------------------------------------------------
# STEP 9: Standardize coordinates
# -------------------------------------------------
print("\n STEP 9: Standardizing coordinates...")

df["latitude"]  = df["latitude"].round(6)
df["longitude"] = df["longitude"].round(6)
print("    Coordinates rounded to 6 decimal places")


# -------------------------------------------------
# STEP 10: Final summary
# -------------------------------------------------
print("\n STEP 10: Final dataset summary...")
print(f"   Total rows    : {len(df)}")
print(f"   Total columns : {len(df.columns)}")
print(f"\n   All columns:")
for col in df.columns:
    print(f"   → {col}")


# -------------------------------------------------
# SAVE cleaned dataset
# -------------------------------------------------
output_file = "Hyderabad_pollution_cleaned.csv"
df.to_csv(output_file, index=False)

print(f"\n{'=' * 55}")
print(f" DONE! Cleaned file saved as: {output_file}")
print(f"{'=' * 55}")