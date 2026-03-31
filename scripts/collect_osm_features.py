# scripts/collect_osm_features.py

import osmnx as ox
from config import *

def collect_osm_features():

    tags = {
        "highway": True,
        "landuse": ["industrial", "farmland"],
        "amenity": ["waste_disposal"]
    }

    features = ox.features_from_place(CITY_NAME, tags)
    features.to_file("data/raw/physical_features.geojson", driver="GeoJSON")

    print("✔ OSM Features collected")