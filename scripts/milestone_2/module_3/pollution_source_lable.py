import pandas as pd

INPUT = "../../../data/processed/cleaned_dataset.csv"
OUTPUT = "../../../data/processed/labeled_dataset.csv"

df = pd.read_csv(INPUT)

def label_source(row):
    # Dominant pollutant
    pollutants = ["pm2.5 value","pm10 value","no2 value","so2 value","co value","o3 value"]
    dominant = max(pollutants, key=lambda x: row[x])

    # Weekend effect: more residential/agricultural burning
    is_weekend = row["weekday"] in [5, 6]

    # Vehicular: high NO2/CO/PM2.5, many roads, weekday rush, low wind dispersion
    if dominant in ["no2 value","co value","pm2.5 value"]:
        if row["road_count_2km"] > 25 and row["wind_dispersion"] > 50 and not is_weekend:
            return "Vehicular"

    # Industrial: high SO2/PM10/NO2, industry presence, weekday activity
    if dominant in ["so2 value","pm10 value","no2 value"]:
        if row["industry_count_5km"] > 5 and not is_weekend:
            return "Industrial"

    # Agricultural burning: high PM10, farmland nearby, dry air, often weekend
    if dominant == "pm10 value":
        if row["agriculture_count_5km"] > 3 and row["humidity"] < 40 and is_weekend:
            return "Agricultural_Burning"

    # Dump burning: high PM2.5/CO, dump presence, low humidity
    if dominant in ["pm2.5 value","co value"]:
        if row["dump_count_5km"] > 2 and row["humidity"] < 50:
            return "Dump_Burning"

    # Natural ozone formation: dominant O3, sunny, warm, moderate wind
    if dominant == "o3 value":
        if row["temperature"] > 25 and row["humidity"] < 60 and row["wind_speed"] > 3:
            return "Natural"

    # Wind effect: if wind speed is very high, PM2.5 can rise due to dust resuspension
    if row["wind_speed"] > 10 and row["pm2.5 value"] > 40:
        return "Natural"

    return "Mixed/Unknown"

df["pollution_source"] = df.apply(label_source, axis=1)
df.to_csv(OUTPUT, index=False)

print("Labeling completed")
print(df["pollution_source"].value_counts())