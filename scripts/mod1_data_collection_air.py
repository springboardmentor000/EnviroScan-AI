import pandas as pd
import random
from datetime import datetime, timedelta
from config import LOCATIONS


def collect_air_quality():
    print("🌫 Generating air quality data (multi-city)...")

    data = []

    for loc in LOCATIONS:
        for i in range(70):  

            data.append({
                "location": loc["name"],
                "latitude": loc["lat"],
                "longitude": loc["lon"],

                "pm2_5": random.randint(20, 120),
                "pm10": random.randint(30, 150),
                "no2": random.randint(5, 50),
                "co": random.randint(100, 400),
                "so2": random.randint(2, 30),
                "o3": random.randint(10, 80),

                "timestamp": datetime.now() - timedelta(hours=i)
            })

    df = pd.DataFrame(data)

    df.to_csv("data/raw/air_quality_data.csv", index=False)

    print(f"✔ Air quality data generated (~{len(df)} rows)")