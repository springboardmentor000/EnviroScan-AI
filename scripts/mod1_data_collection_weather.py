import pandas as pd
import random
from datetime import datetime, timedelta


def collect_weather():
    print("🌦 Generating weather data...")

    data = []

    for i in range(40):

        data.append({
            "temperature": random.randint(20, 40),
            "humidity": random.randint(40, 90),
            "wind_speed": random.uniform(1, 10),
            "wind_direction": random.randint(0, 360),
            "timestamp": datetime.now() - timedelta(hours=i)
        })

    df = pd.DataFrame(data)

    df.to_csv("data/raw/weather_data.csv", index=False)

    print("✔ Weather Data Created (40 rows)")