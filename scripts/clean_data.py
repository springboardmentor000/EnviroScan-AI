import pandas as pd

def clean_weather_data():
    df = pd.read_csv("data/raw/weather_data.csv")
    df.to_csv("data/processed/weather_cleaned.csv", index=False)

def clean_air_data():
    df = pd.read_csv("data/raw/air_quality_data.csv")
    df.to_csv("data/processed/air_quality_cleaned.csv", index=False)

def clean_distance_data():
    df = pd.read_csv("data/processed/distance_cleaned.csv")
    df.to_csv("data/processed/distance_cleaned.csv", index=False)