import pandas as pd
import os

DATA_PATH = "../../../data/raw"
PROCESSED_PATH = "../../../data/processed"

print("Merging air quality, weather, and OSM datasets...")

air_df = pd.read_csv(f"{DATA_PATH}/air_quality_data.csv")
weather_df = pd.read_csv(f"{DATA_PATH}/weather_data.csv")
osm_df = pd.read_csv(f"{DATA_PATH}/osm_features.csv")


merged_df = pd.merge(
    air_df,
    weather_df,
    on=["city", "latitude", "longitude", "timestamp"],
    how="inner"
)

print("After merging Air + Weather:", merged_df.shape)


final_df = pd.merge(
    merged_df,
    osm_df,
    on=["city", "latitude", "longitude"],
    how="left"
)

print("After merging OSM:", final_df.shape)


os.makedirs(PROCESSED_PATH, exist_ok=True)

final_df.to_csv(f"{PROCESSED_PATH}/merged_dataset.csv", index=False)

print("Master dataset created successfully!")