# scripts/collect_weather.py

import requests
import pandas as pd
import random
from config import *

def collect_weather():

    rows = []

    for i in range(TOTAL_ROWS):

        # generate nearby coordinate
        lat_offset = random.uniform(-0.05, 0.05)
        lon_offset = random.uniform(-0.05, 0.05)

        new_lat = LATITUDE + lat_offset
        new_lon = LONGITUDE + lon_offset

        url = "https://api.openweathermap.org/data/2.5/weather"

        params = {
            "lat": new_lat,
            "lon": new_lon,
            "appid": API_KEY,
            "units": "metric"
        }

        res = requests.get(url, params=params).json()

        rows.append({
            "latitude": new_lat,
            "longitude": new_lon,
            "temperature": res["main"]["temp"],
            "humidity": res["main"]["humidity"],
            "wind_speed": res["wind"]["speed"],
            "wind_direction": res["wind"]["deg"],
        })

    df = pd.DataFrame(rows)
    df.to_csv("data/raw/weather_data.csv", index=False)

    print("✔ Weather data collected (20 unique locations)")