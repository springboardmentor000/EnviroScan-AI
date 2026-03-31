import pandas as pd
import random

def collect_osm_features():

    data = []

    for i in range(40):
        data.append({
            "roads_count": random.randint(50, 500),
            "industrial_count": random.randint(0, 5),
            "farmland_count": random.randint(0, 5),
            "dumpsite_count": random.randint(0, 3)
        })

    df = pd.DataFrame(data)
    df.to_csv("data/raw/osm_features.csv", index=False)

    print("✔ OSM Features Created")