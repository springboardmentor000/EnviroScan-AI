import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
sys.path.append("../../..")

import config

DATA_PATH = "../../../data/raw"

print("Fetching LARGE AI-ready air quality dataset (OWM + OpenAQ combined)...")

records = []

start_time = datetime.now() - timedelta(days=config.DAYS)
end_time = datetime.now()

for city, locations in config.CITIES.items():
    for lat, lon in locations:

        current_time = start_time

        while current_time <= end_time:
            # 🔹 Only collect data every 4th day
            day_index = (current_time - start_time).days
            if day_index % 4 != 0:
                current_time += timedelta(hours=config.HOURS_INTERVAL)
                continue

            try:
                # 🔹 OWM Air Pollution API
                owm_url = (
                    f"http://api.openweathermap.org/data/2.5/air_pollution?"
                    f"lat={lat}&lon={lon}&appid={config.OWM_API_KEY}"
                )
                owm_resp = requests.get(owm_url).json()
                owm_comp = owm_resp["list"][0]["components"]

                # 🔹 OpenAQ API (latest measurement near coordinates)
                openaq_url = (
                    f"https://api.openaq.org/v2/measurements?"
                    f"coordinates={lat},{lon}&limit=10&key={config.OPENAQ_API_KEY}"
                )
                openaq_resp = requests.get(openaq_url).json()
                openaq_values = {}
                if "results" in openaq_resp and len(openaq_resp["results"]) > 0:
                    for m in openaq_resp["results"]:
                        openaq_values[m["parameter"]] = m["value"]

                # 🔹 Unified pollutant values (prefer OpenAQ, fallback to OWM)
                pm25 = openaq_values.get("pm25", owm_comp.get("pm2_5"))
                pm10 = openaq_values.get("pm10", owm_comp.get("pm10"))
                no2  = openaq_values.get("no2",  owm_comp.get("no2"))
                so2  = openaq_values.get("so2",  owm_comp.get("so2"))
                co   = openaq_values.get("co",   owm_comp.get("co"))
                o3   = openaq_values.get("o3",   owm_comp.get("o3"))

                records.append({
                    "city": city,
                    "latitude": lat,
                    "longitude": lon,
                    "pm2.5 value": pm25,
                    "pm10 value": pm10,
                    "no2 value": no2,
                    "so2 value": so2,
                    "co value": co,
                    "o3 value": o3,
                    "unit": "µg/m3",
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S")
                })

            except Exception as e:
                print(f"Error fetching data for {lat},{lon} at {current_time}: {e}")

            current_time += timedelta(hours=config.HOURS_INTERVAL)

df = pd.DataFrame(records)

os.makedirs(DATA_PATH, exist_ok=True)

df.to_csv(f"{DATA_PATH}/air_quality_data.csv", index=False)

print("Air quality dataset created:", len(df), "rows")