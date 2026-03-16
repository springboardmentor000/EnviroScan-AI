import pandas as pd

print("Cleaning Final Dataset...\n")

# -----------------------------
# 1 Load dataset
# -----------------------------
df = pd.read_csv("module3_labeled_dataset.csv")

print("Original Shape:", df.shape)

# -----------------------------
# 2 Remove empty columns
# -----------------------------
df = df.dropna(axis=1, how="all")

# -----------------------------
# 3 Remove duplicate rows
# -----------------------------
df = df.drop_duplicates()

# -----------------------------
# 4 Standardize column names
# -----------------------------
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# -----------------------------
# 5 Fix timestamp
# -----------------------------
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Extract temporal features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month

# -----------------------------
# 6 Fix city names
# -----------------------------
if "city" in df.columns:
    df["city"] = df["city"].astype(str)
    df["city"] = df["city"].str.split().str[0]

# -----------------------------
# 7 Convert numeric columns
# -----------------------------
numeric_cols = [
    "latitude",
    "longitude",
    "temperature_c",
    "humidity_%",
    "wind_speed_mps",
    "pm2_5",
    "pm10",
    "no2",
    "so2",
    "co",
    "dist_to_road",
    "dist_to_industry",
    "dist_to_farmland"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# 8 Handle missing values
# -----------------------------
df = df.fillna(df.median(numeric_only=True))

# -----------------------------
# 9 Remove corrupted rows
# -----------------------------
if "pm2_5" in df.columns:
    df = df.dropna(subset=["pm2_5"])

# -----------------------------
# 10 Reorder columns
# -----------------------------
preferred_order = [
    "timestamp",
    "city",
    "latitude",
    "longitude",
    "temperature_c",
    "humidity_%",
    "wind_speed_mps",
    "pm2_5",
    "pm10",
    "no2",
    "so2",
    "co",
    "season",
    "hour",
    "day",
    "month",
    "dist_to_road",
    "dist_to_industry",
    "dist_to_farmland",
    "pollution_source"
]

df = df[[c for c in preferred_order if c in df.columns]]

# -----------------------------
# 11 Save cleaned dataset
# -----------------------------
df.to_csv("clean_pollution_dataset.csv", index=False)

print("\nClean Dataset Shape:", df.shape)
print("Clean dataset saved → clean_pollution_dataset.csv")