import pandas as pd

INPUT = "../../../data/processed/merged_dataset.csv"
OUTPUT = "../../../data/processed/feature_dataset.csv"

df = pd.read_csv(INPUT)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp_x"], dayfirst=True)

# Time features
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["weekday"] = df["timestamp"].dt.dayofweek

# Pollution features
df["total_pollution"] = (
    df["pm2.5 value"]
    + df["pm10 value"]
    + df["no2 value"]
    + df["so2 value"]
    + df["co value"]
    + df["o3 value"]
)

df["pm_ratio"] = df["pm2.5 value"] / (df["pm10 value"] + 1)

# Weather interaction
df["wind_dispersion"] = df["total_pollution"] / (df["wind_speed"] + 1)

df["humidity_pm"] = df["humidity"] * df["pm2.5 value"]

df.to_csv(OUTPUT, index=False)

print("Feature dataset created:", df.shape)