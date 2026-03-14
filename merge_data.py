import pandas as pd

# ──────────────────────────────────────────
# STEP 1: Load all 3 files
# ──────────────────────────────────────────
aq = pd.read_csv("vijayawada_airquality.csv")
wt = pd.read_csv("vijayawada_weather.csv")
pf = pd.read_csv("vijayawada_physical_features.csv")

print(f"Air Quality  : {aq.shape}")
print(f"Weather      : {wt.shape}")
print(f"Physical     : {pf.shape}")
print(aq[['timestamp','location_id']].head())
print(wt[['timestamp','location_id']].head())

# ──────────────────────────────────────────
# STEP 2: Normalize timestamps
# airquality  → "01-02-2026 07:00"  (dayfirst)
# weather     → "2026-02-01 00:00:00"
# ──────────────────────────────────────────
# ──────────────────────────────────────────
# STEP 2: Normalize timestamps
# airquality  → "01-02-2026 07:00"  (dayfirst)
# weather     → "2026-02-01 00:00:00"
# ──────────────────────────────────────────
aq["timestamp"] = pd.to_datetime(aq["timestamp"], format='mixed', dayfirst=True)
wt["timestamp"] = pd.to_datetime(wt["timestamp"], format='mixed')

# Round both to nearest hour to ensure clean join
aq["timestamp"] = aq["timestamp"].dt.floor("h")
wt["timestamp"] = wt["timestamp"].dt.floor("h")

print(f"\nAQ  date range : {aq['timestamp'].min()} → {aq['timestamp'].max()}")
print(f"WT  date range : {wt['timestamp'].min()} → {wt['timestamp'].max()}")


# ──────────────────────────────────────────
# STEP 3: Merge AQ + Weather on timestamp + location_id
# ──────────────────────────────────────────
wt_cols = [
    "timestamp",
    "location_id",
    "temperature_c",
    "humidity",
    "pressure_hpa",
    "wind_speed_ms",
    "wind_direction"
]

df = pd.merge(
    aq,
    wt[wt_cols],
    on=["timestamp", "location_id"],
    how="left"
)


# ──────────────────────────────────────────
# STEP 4: Merge with Physical Features on location_id
# (static table — no timestamp, just per station)
# ──────────────────────────────────────────
pf_cols = ["location_id", "road_count", "industrial_count", "waste_count", "farmland_count"]

df = pd.merge(
    df,
    pf[pf_cols],
    on="location_id",
    how="left"
)

print(f"After Physical merge     : {df.shape}")
print(f"Physical matched rows    : {df['road_count'].notna().sum()} / {len(df)}")

# ──────────────────────────────────────────
# STEP 5: Final column order & save
# ──────────────────────────────────────────
final_cols = [
    "city", "timestamp", "location_id", "location", "lat", "lon",
    "pm2_5", "pm10", "no2", "o3", "so2", "co",
    "temperature_c", "humidity", "pressure_hpa", "wind_speed_ms", "wind_direction",
    "road_count", "industrial_count", "waste_count", "farmland_count"
]
final_cols = [c for c in final_cols if c in df.columns]
df = df[final_cols].sort_values(["location", "timestamp"]).reset_index(drop=True)

# Remove duplicate rows
before = len(df)
df = df.drop_duplicates(subset=["timestamp", "location_id"])
print(f"\nDuplicates removed : {before - len(df)}")

print(f"\n✅ Final shape   : {df.shape}")
print(f"   Columns      : {df.columns.tolist()}")
print(f"   Stations     : {df['location'].unique()}")
print(f"   Date range   : {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"   Missing vals :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(df.head(5).to_string())

df.to_csv("vijayawada_final_dataset.csv", index=False)
print("\n💾 Saved → vijayawada_final_dataset.csv")
print(aq['timestamp'].dtype)
print(wt['timestamp'].dtype)