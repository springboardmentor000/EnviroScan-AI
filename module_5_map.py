

import pandas as pd
import folium
from folium.plugins import HeatMap
import os

print("🚀 Starting Module 5: Advanced Geospatial Visualization\n")

try:
    file_path = "Infosys_dataset.csv"

    if not os.path.exists(file_path):
        raise Exception(" Dataset file not found!")

    df = pd.read_csv(file_path)
    df = df.drop_duplicates()

    print(f"📊 Loaded Data Shape: {df.shape}")

    required_cols = ["latitude", "longitude", "PM2_5", "source_domain"]

    for col in required_cols:
        if col not in df.columns:
            raise Exception(f" Missing column: {col}")

    m = folium.Map(
        location=[df["latitude"].mean(), df["longitude"].mean()],
        zoom_start=5,
        tiles="cartodbpositron"   
    )

   
    heat_layer = folium.FeatureGroup(name="🔥Pollution Heatmap")

    heat_data = df[["latitude", "longitude", "PM2_5"]].dropna().values.tolist()

    HeatMap(
        heat_data,
        radius=18,
        blur=12
    ).add_to(heat_layer)

    heat_layer.add_to(m)

    print("✔ Heatmap added")


    def get_color(source):
        return {
            "Vehicular": "blue",
            "Industrial": "red",
            "Agricultural": "green",
            "Urban Dust": "orange",
            "Natural": "purple",
            "Burning": "black"
        }.get(source, "gray")

    
    layers = {
        "Vehicular": folium.FeatureGroup(name="🚗 Source: Vehicular"),
        "Industrial": folium.FeatureGroup(name="🏭 Source: Industrial"),
        "Agricultural": folium.FeatureGroup(name="🌾 Source: Agricultural"),
        "Urban Dust": folium.FeatureGroup(name="🌪 Source: Urban Dust"),
        "Natural": folium.FeatureGroup(name="🌿 Source: Natural"),
        "Burning": folium.FeatureGroup(name="🔥 Source: Burning")
    }


    for _, row in df.iterrows():

        src = row["source_domain"]

        marker = folium.Marker(
            location=[row["latitude"], row["longitude"]],
            icon=folium.Icon(
                color=get_color(src),
                icon="info-sign"
            ),
            popup=f"""
            <b>City:</b> {row.get('city','')}<br>
            <b>Source:</b> {src}<br>
            <b>AQI:</b> {row.get('AQI_score','')}
            """
        )

        if src in layers:
            marker.add_to(layers[src])
        else:
            marker.add_to(m)

    
    for layer in layers.values():
        layer.add_to(m)

  
    folium.LayerControl(collapsed=False).add_to(m)

    print("✔ Layer control added")

   
    output_file = "pollution_heatmap.html"
    m.save(output_file)

    print(f"Map saved → {output_file}")

except Exception as e:
    print(" ERROR:", e)

print("\n Module 5 Completed Successfully!")