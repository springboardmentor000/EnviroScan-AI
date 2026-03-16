import pandas as pd

print("Enhancing Dataset with Advanced Features...\n")

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("clean_pollution_dataset.csv")

print("Dataset loaded")
print("Original Shape:", df.shape)

# -----------------------------
# 1 Remove duplicate rows
# -----------------------------
df = df.drop_duplicates()

print("After removing duplicates:", df.shape)

# -----------------------------
# 2 Convert timestamp
# -----------------------------
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# -----------------------------
# 3 Add day_of_week
# -----------------------------
if "timestamp" in df.columns:
    df["day_of_week"] = df["timestamp"].dt.day_name()

# -----------------------------
# 4 Pollution Index
# -----------------------------
required_cols = ["pm2_5","pm10","no2","so2"]

if all(col in df.columns for col in required_cols):
    df["pollution_index"] = (
        df["pm2_5"] +
        df["pm10"] +
        df["no2"] +
        df["so2"]
    ) / 4

# -----------------------------
# 5 AQI Category
# -----------------------------
if "pm2_5" in df.columns:
    df["aqi_category"] = pd.cut(
        df["pm2_5"],
        bins=[0,30,60,90,120,250],
        labels=[
            "Good",
            "Satisfactory",
            "Moderate",
            "Poor",
            "Very Poor"
        ]
    )

# -----------------------------
# 6 Rolling PM average
# -----------------------------
if "pm2_5" in df.columns:
    df["pm25_rolling_avg"] = df["pm2_5"].rolling(window=3, min_periods=1).mean()

# -----------------------------
# 7 Pollution change rate
# -----------------------------
if "pm2_5" in df.columns:
    df["pm25_change"] = df["pm2_5"].diff()

# -----------------------------
# Save enhanced dataset
# -----------------------------
df.to_csv("enhanced_pollution_dataset.csv", index=False)

print("\nEnhanced dataset saved as:")
print("enhanced_pollution_dataset.csv")

print("\nFinal Shape:", df.shape)