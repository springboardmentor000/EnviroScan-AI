# scripts/collect_air_quality.py

import requests
import pandas as pd
from config import *

def collect_air_quality():

    weather = pd.read_csv("data/raw/weather_data.csv")

    rows = []

    for _, row in weather.iterrows():

        lat = row["latitude"]
        lon = row["longitude"]

        url = "http://api.openweathermap.org/data/2.5/air_pollution"

        params = {
            "lat": lat,
            "lon": lon,
            "appid": API_KEY
        }

        res = requests.get(url, params=params).json()
        comp = res["list"][0]["components"]

        rows.append({
            "latitude": lat,
            "longitude": lon,
            "PM2.5": comp["pm2_5"],
            "PM10": comp["pm10"],
            "NO2": comp["no2"],
            "CO": comp["co"],
            "SO2": comp["so2"],
            "O3": comp["o3"]
        })

    df = pd.DataFrame(rows)
    df.to_csv("data/raw/air_quality_data.csv", index=False)

    print("✔ Air Quality collected (20 unique locations)")