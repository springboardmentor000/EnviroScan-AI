import pandas as pd

df = pd.read_csv("vijayawada_feature_engineering.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# STEP 4: columns for labeling 
df["_month"]  = df["timestamp"].dt.month  # ✅ Fixed column name
df["_is_dry"] = df["_month"].isin([11, 12, 1, 2]).astype(int)

# STEP 5: Dynamic thresholds
def pct(col, p):
    return pd.to_numeric(df[col], errors="coerce").quantile(p)

HIGH_NO2        = pct("no2",              0.75)
HIGH_SO2        = pct("so2",              0.75)
HIGH_PM25       = pct("pm2_5",            0.75)
HIGH_PM10       = pct("pm10",             0.75)
HIGH_CO         = pct("co",               0.75)
HIGH_O3         = pct("o3",               0.75)
HIGH_ROADS      = pct("road_count",       0.75)
HIGH_INDUSTRIAL = pct("industrial_count", 0.50)
HIGH_FARMLAND   = pct("farmland_count",   0.70)
HIGH_WASTE      = pct("waste_count",      0.70)

print(f"\n📊 Thresholds:")
print(f"   NO2>{HIGH_NO2:.1f} | SO2>{HIGH_SO2:.1f} | PM2.5>{HIGH_PM25:.1f}")
print(f"   Roads>{HIGH_ROADS:.0f} | Industrial>{HIGH_INDUSTRIAL:.0f} | Farmland>{HIGH_FARMLAND:.0f}| Waste>{HIGH_WASTE:.0f}")

# STEP 6: Rule-based labeling
def assign_label(row):

    near_road       = row["road_count"]       >= HIGH_ROADS
    near_industrial = row["industrial_count"] >= HIGH_INDUSTRIAL
    near_farmland   = row["farmland_count"]   >= HIGH_FARMLAND
    near_waste      = row["waste_count"]      >= HIGH_WASTE

    high_no2  = row["no2"]   >= HIGH_NO2
    high_so2  = row["so2"]   >= HIGH_SO2
    high_pm25 = row["pm2_5"] >= HIGH_PM25
    high_pm10 = row["pm10"]  >= HIGH_PM10
    high_co   = row["co"]    >= HIGH_CO
    high_o3   = row["o3"]    >= HIGH_O3

    dry = row["_is_dry"] == 1
    rush_hour = (9<= row["hour"] <= 12) or (17 <= row["hour"] <= 20)

    # 1️⃣ Vehicular pollution
    if near_road and rush_hour and (high_no2 or high_co) :
        return "Vehicular"

    # 2️⃣ Industrial pollution
    if near_industrial and (high_so2 or high_pm25):
        return "Industrial"

    # 3️⃣ Agricultural pollution
    if near_farmland and dry and (high_pm10 or high_pm25 and dry )  :
        return "Agricultural"

    # 4️⃣ Burning pollution (waste burning)
    if near_waste and ( high_co or high_no2 or high_pm25) :
        return "Burning"

    # 5️⃣ Natural pollution
    if (not high_no2 and not high_so2 and not high_pm25 and not high_pm10 and not high_co and not high_o3):
        return "Natural"

    # 6️⃣ Fallback logic (dominant pollutant)
    scores = {
        "Vehicular": row["no2"] + row["co"],
        "Industrial": row["so2"] + row["pm2_5"],
        "Agricultural": row["pm10"],
        "Burning": row["pm2_5"] + row["co"],
        "Natural": row["o3"]
    }

    return max(scores, key=scores.get)
    
print("\n⚙️ Applying pollution source labels...")
df["pollution_source"] = df.apply(assign_label, axis=1)



# STEP 7: Distribution report
print("\n📊 Label Distribution:")
print(df["pollution_source"].value_counts().to_string())
print("\n   Per station:")
print(df.groupby(["location", "pollution_source"]).size().unstack(fill_value=0).to_string())

# STEP 8: Final columns & save ✅ Define weather_val_cols
weather_val_cols = ["temperature_c", "humidity", "pressure_hpa", "wind_speed_ms", "wind_direction"]

final_cols = (
    ["city", "timestamp", "location_id", "location", "lat", "lon"] +
    ["pm2_5", "pm10", "no2", "o3", "so2", "co"] +
    [c for c in weather_val_cols if c in df.columns] +
    ["road_count", "industrial_count", "waste_count", "farmland_count"] +
    ["pollution_source",",dist_to_center_km","dist_to_industry_km","dist_to_road_km","dist_to_dump_km","hour","day_of_week","day_of_year","month","is_weekend","is_rush_hour","is_dry_season","season"]
)
final_cols = [c for c in final_cols if c in df.columns]

df_final = df[final_cols].sort_values(["location", "timestamp"]).reset_index(drop=True)

print(f"\n✅ Final shape   : {df_final.shape}")
print(f"   Columns      : {df_final.columns.tolist()}")
print(df_final.head(5).to_string())

df_final.to_csv("vijayawada_labelled_dataset.csv", index=False)
print("\n💾 Saved → vijayawada_labelled_dataset.csv")
print(df["farmland_count"].describe())