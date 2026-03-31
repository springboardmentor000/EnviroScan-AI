from scripts.mod1_data_collection_air import collect_air_quality
from scripts.mod1_data_collection_weather import collect_weather
from scripts.mod1_data_collection_osm import collect_osm_features

from scripts.mod2_data_cleaning import (
    clean_weather_data,
    clean_air_quality_data,
    clean_distance_data
)

from scripts.mod3_feature_engineering import feature_engineering
from scripts.mod4_source_prediction import label_sources
from scripts.mod5_geospatial_map import create_map
from scripts.mod6_export_excel import export_to_excel

from scripts.calculate_distance import calculate_distances


print("🚀 Starting Project...\n")

# MODULE 1: DATA COLLECTION

print("📦 MODULE 1: DATA COLLECTION")
collect_air_quality()
collect_weather()
collect_osm_features()

# DISTANCE CALCULATION

calculate_distances()

# MODULE 2: DATA CLEANING

print("🧹 MODULE 2: DATA CLEANING")
clean_weather_data()
clean_air_quality_data()
clean_distance_data()

# MODULE 3: FEATURE ENGINEERING

print("🔧 MODULE 3: FEATURE ENGINEERING")
feature_engineering()

# MODULE 4: SOURCE PREDICTION

print("🧠 MODULE 4: SOURCE PREDICTION")
label_sources()

# MODULE 5: GEOSPATIAL MAP

print("🗺 MODULE 5: GEOSPATIAL VISUALIZATION")
create_map()

# MODULE 6: EXPORT

print("📄 MODULE 6: EXPORT")
export_to_excel()

print("\n🎉 PROJECT COMPLETED SUCCESSFULLY!")