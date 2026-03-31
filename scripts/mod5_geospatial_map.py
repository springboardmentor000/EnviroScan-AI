import pandas as pd
import folium
from folium.plugins import HeatMap


def create_map():
    df = pd.read_csv("data/processed/labeled_dataset.csv")

    m = folium.Map(location=[22.5, 78.9], zoom_start=5)

    heat_data = [
        [row["latitude"], row["longitude"], row["pm2_5"]]
        for _, row in df.iterrows()
    ]

    HeatMap(heat_data, radius=20).add_to(m)

    for _, row in df.iterrows():

        if row["pollution_source"] == "Vehicular":
            color = "blue"
        elif row["pollution_source"] == "Industrial":
            color = "red"
        elif row["pollution_source"] == "Agricultural":
            color = "green"
        elif row["pollution_source"] == "Natural":
            color = "gray"
        else:
            color = "purple"

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            popup=f"{row['location']} | {row['pollution_source']} | PM2.5: {row['pm2_5']}"
        ).add_to(m)

    m.save("data/processed/pollution_map.html")


if __name__ == "__main__":
    create_map()