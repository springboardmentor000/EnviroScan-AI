import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
sys.path.append("../../..")

import config

DATA_PATH = "../../../data/raw"

print("Fetching selective AI-ready weather dataset...")

records = []

end_time = datetime(2026, 3, 30, 12, 0, 0)
start_time = end_time - timedelta(days=config.DAYS)

def is_first_or_last_week(dt):
    first_day = dt.replace(day=1)
    last_day = (dt.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    return dt.day <= 7 or dt.day > (last_day.day - 7)

for city, locations in config.CITIES.items():
    for lat, lon in locations:
        current_time = start_time

        while current_time <= end_time:
            if current_time.weekday() not in [0, 6]:
                current_time += timedelta(hours=config.HOURS_INTERVAL)
                continue

            if not is_first_or_last_week(current_time):
                current_time += timedelta(hours=config.HOURS_INTERVAL)
                continue

            try:

                owm_url = (
                    f"http://api.openweathermap.org/data/2.5/weather?"
                    f"lat={lat}&lon={lon}&appid={config.OWM_API_KEY}&units=metric"
                )
                owm_resp = requests.get(owm_url, timeout=10).json()
                temp = owm_resp["main"]["temp"]
                humidity = owm_resp["main"]["humidity"]
                wind_speed = owm_resp["wind"]["speed"]
                wind_dir = owm_resp["wind"].get("deg", None)

                openaq_url = (
                    f"https://api.openaq.org/v2/measurements?"
                    f"coordinates={lat},{lon}&limit=10&key={config.OPENAQ_API_KEY}"
                )
                openaq_resp = requests.get(openaq_url, timeout=10).json()
                openaq_values = {}
                if "results" in openaq_resp and len(openaq_resp["results"]) > 0:
                    for m in openaq_resp["results"]:
                        if m["parameter"] in ["temperature", "humidity", "wind_speed", "wind_direction"]:
                            openaq_values[m["parameter"]] = m["value"]

                temperature = openaq_values.get("temperature", temp)
                humidity_val = openaq_values.get("humidity", humidity)
                wind_speed_val = openaq_values.get("wind_speed", wind_speed)
                wind_dir_val = openaq_values.get("wind_direction", wind_dir)

                records.append({
                    "city": city,
                    "latitude": lat,
                    "longitude": lon,
                    "temperature": temperature,
                    "humidity": humidity_val,
                    "wind_speed": wind_speed_val,
                    "wind_direction": wind_dir_val,
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S")
                })

            except Exception as e:
                print(f"Error fetching weather for {lat},{lon} at {current_time}: {e}")

            current_time += timedelta(hours=config.HOURS_INTERVAL)

df = pd.DataFrame(records)

os.makedirs(DATA_PATH, exist_ok=True)
df.to_csv(f"{DATA_PATH}/weather_data.csv", index=False)

print("Weather dataset created:", len(df), "rows")