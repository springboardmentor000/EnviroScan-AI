import requests
import pandas as pd
from datetime import datetime
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import config

API_KEY = config.OWM_API_KEY

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
file_path = os.path.join(BASE_DIR, "data", "static_features.csv")

static_df = pd.read_csv(file_path)

data = []

for _, row in static_df.iterrows():

    lat = row["latitude"]
    lon = row["longitude"]

    try:
        # Pollution
        air_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        air = requests.get(air_url).json()
        comp = air["list"][0]["components"]

        # Weather
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        weather = requests.get(weather_url).json()

        temp = weather["main"]["temp"]
        humidity = weather["main"]["humidity"]
        wind = weather["wind"]["speed"]

        data.append({
            "city": row["city"],
            "latitude": lat,
            "longitude": lon,

            "pm2.5 value": comp["pm2_5"],
            "pm10 value": comp["pm10"],
            "no2 value": comp["no2"],
            "so2 value": comp["so2"],
            "co value": comp["co"],
            "o3 value": comp["o3"],

            "temperature": temp,
            "humidity": humidity,
            "wind_speed": wind,

            "road_count_2km": row["road_count_2km"],
            "industry_count_5km": row["industry_count_5km"],
            "agriculture_count_5km": row["agriculture_count_5km"],
            "dump_count_5km": row["dump_count_5km"],

            "timestamp": datetime.now()
        })

    except Exception as e:
        print(f"{row['city']} error:", e)

df = pd.DataFrame(data)

file_path = "../../data/live_data.csv"

if os.path.exists(file_path):
    old_df = pd.read_csv(file_path)
    df = pd.concat([old_df, df], ignore_index=True)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df[df["timestamp"] > (datetime.now() - pd.Timedelta(hours=1))]

output_path = os.path.join(BASE_DIR, "data", "live_data.csv")
df.to_csv(output_path, index=False)

print("Live data updated (static + dynamic combined)")