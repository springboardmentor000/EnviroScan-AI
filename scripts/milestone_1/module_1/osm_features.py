import osmnx as ox
import pandas as pd
from datetime import datetime
import os
import sys
sys.path.append("../../..")

import config

DATA_PATH = "../../../data/raw"

print("Collecting OSM geospatial features...")

ox.settings.timeout = 180        
ox.settings.use_cache = True     
ox.settings.log_console = True   

records = []

for city, locations in config.CITIES.items():
    for lat, lon in locations:
        point = (lat, lon)

        try:
            road_tags = {"highway": ["motorway", "trunk", "primary", "secondary"]}
            roads = ox.features_from_point(point, tags=road_tags, dist=2000)
            road_count = len(roads)
        except Exception as e:
            print(f"Error fetching roads for {point}: {e}")
            road_count = 0

        try:
 
            industry_tags = {
                "landuse": "industrial",
                "building": "industrial",
                "man_made": "works"
            }
            industries = ox.features_from_point(point, tags=industry_tags, dist=5000)
            industry_count = len(industries)
        except Exception as e:
            print(f"Error fetching industries for {point}: {e}")
            industry_count = 0

        try:
            dump_tags = {"landuse": "landfill"}
            dumps = ox.features_from_point(point, tags=dump_tags, dist=5000)
            dump_count = len(dumps)
        except Exception as e:
            print(f"Error fetching dumps for {point}: {e}")
            dump_count = 0

        try:
            agri_tags = {"landuse": "farmland"}
            farms = ox.features_from_point(point, tags=agri_tags, dist=5000)
            agri_count = len(farms)
        except Exception as e:
            print(f"Error fetching farmland for {point}: {e}")
            agri_count = 0

        records.append({
            "city": city,
            "latitude": lat,
            "longitude": lon,
            "road_count_2km": road_count,
            "industry_count_5km": industry_count,
            "dump_count_5km": dump_count,
            "agriculture_count_5km": agri_count,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

df = pd.DataFrame(records)

os.makedirs(DATA_PATH, exist_ok=True)

df.to_csv(f"{DATA_PATH}/osm_features.csv", index=False)

print("OSM features dataset created:", len(df), "rows")