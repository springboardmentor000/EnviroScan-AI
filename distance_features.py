import pandas as pd
import osmnx as ox
from shapely.geometry import Point
import geopandas as gpd

# Load dataset
df = pd.read_csv("enviro_data_india.csv")

def get_distances(lat, lon):
    try:
        point = (lat, lon)

        tags = {
            "highway": True,
            "landuse": True
        }

        # Get nearby features
        gdf = ox.features_from_point(point, tags=tags, dist=2000)

        if gdf.empty:
            return (None, None, None)

        # Convert to projected CRS (meters)
        gdf = gdf.to_crs(epsg=3857)

        # ✅ FIX: Create proper GeoDataFrame for point
        point_geom = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]

        # Safe filtering
        roads = gdf[gdf["highway"].notnull()] if "highway" in gdf.columns else gdf.iloc[0:0]
        industrial = gdf[gdf["landuse"] == "industrial"] if "landuse" in gdf.columns else gdf.iloc[0:0]
        farmland = gdf[gdf["landuse"] == "farmland"] if "landuse" in gdf.columns else gdf.iloc[0:0]

        def min_distance(geo_df):
            if geo_df.empty:
                return None
            return geo_df.distance(point_geom).min()

        return (
            min_distance(roads),
            min_distance(industrial),
            min_distance(farmland)
        )

    except Exception as e:
        print("Error:", e)
        return (None, None, None)


# -------- MAIN EXECUTION --------
print("Processing rows...")

distances = []

for index, row in df.iterrows():
    print(f"Processing row {index+1}/{len(df)}...")
    d = get_distances(row["Latitude"], row["Longitude"])
    distances.append(d)

# Add new columns
df[["Dist_Road", "Dist_Industrial", "Dist_Farmland"]] = pd.DataFrame(distances)

# Save file
df.to_csv("enviro_data_with_distance.csv", index=False)

print("\n✅ Distance features added successfully!")