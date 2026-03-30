import pandas as pd

INPUT = "../../../data/processed/cleaned_dataset.csv"
OUTPUT = "../../../data/processed/labeled_dataset.csv"

df = pd.read_csv(INPUT)

# pollutant to possible sources map
pollutant_source_map = {
    "no2 value": ["Vehicular", "Industrial"],
    "so2 value": ["Industrial"],
    "pm2.5 value": ["Vehicular", "Dump_Burning"],
    "pm10 value": ["Agricultural_Burning", "Industrial"],
    "co value": ["Vehicular", "Dump_Burning"],
    "o3 value": ["Natural"]
}

def label_source(row):

    pollutants = [
        "pm2.5 value",
        "pm10 value",
        "no2 value",
        "so2 value",
        "co value",
        "o3 value"
    ]

    # find dominant pollutant
    dominant = max(pollutants, key=lambda x: row[x])

    possible_sources = pollutant_source_map[dominant]

    # decide using nearby features
    if "Vehicular" in possible_sources and row["road_count_2km"] > 40:
        return "Vehicular"

    if "Industrial" in possible_sources and row["industry_count_5km"] > 5:
        return "Industrial"

    if "Agricultural_Burning" in possible_sources and row["agriculture_count_5km"] > 3:
        return "Agricultural_Burning"

    if "Dump_Burning" in possible_sources and row["dump_count_5km"] > 2:
        return "Dump_Burning"

    return "Natural"


df["pollution_source"] = df.apply(label_source, axis=1)

df.to_csv(OUTPUT, index=False)

print("Labeling completed")
print(df["pollution_source"].value_counts())