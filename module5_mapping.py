# =========================================
#  IMPORTS
# =========================================
import pandas as pd
import folium
import os
from folium.plugins import HeatMap

# =========================================
#  PATH SETUP
# =========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "outputs", "predictions.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "pollution_map.html")

print("\n Module 5 Started...")

# =========================================
#  LOAD DATA
# =========================================
df = pd.read_csv(DATA_PATH)

# Fix column names
df.rename(columns={"lat": "latitude", "lon": "longitude"}, inplace=True)

# =========================================
#  CLEAN DATA
# =========================================
df = df.dropna(subset=["latitude", "longitude", "pm2_5", "predicted_source"])

# =========================================
#  CORRECT LOGIC (WEIGHTED SOURCE)
# =========================================
def get_main_source(group):
    total = group['pm2_5'].sum()
    weights = group.groupby('predicted_source')['pm2_5'].sum() / total
    return weights.idxmax()

# =========================================
#  AGGREGATE DATA (CITY LEVEL)
# =========================================
df_grouped = df.groupby('city').apply(
    lambda x: pd.Series({
        'latitude': x['latitude'].iloc[0],
        'longitude': x['longitude'].iloc[0],
        'pm2_5': x['pm2_5'].mean(),
        'main_source': get_main_source(x)
    })
).reset_index()

print("\n Final Aggregated Data:")
print(df_grouped)

# =========================================
#  CREATE MAP
# =========================================
m = folium.Map(
    location=[20.5937, 78.9629],
    zoom_start=5,
    tiles="CartoDB positron"
)

# =========================================
#  HEATMAP (STRONG + BALANCED)
# =========================================
max_pm = df['pm2_5'].max()

heat_data = []
for _, row in df.iterrows():
    normalized = row['pm2_5'] / max_pm
    weight = (normalized ** 0.5) * 20   # boost for better red visibility
    heat_data.append([row['latitude'], row['longitude'], weight])

HeatMap(
    heat_data,
    radius=60,
    blur=70,
    min_opacity=0.7,
    gradient={
        0.0: "blue",
        0.3: "lime",
        0.5: "yellow",
        0.7: "red",
        1.0: "darkred"
    }
).add_to(m)

# =========================================
#  ICON MAPPING
# =========================================
icon_map = {
    "Vehicular": ("car", "blue"),
    "Industrial": ("industry", "red"),
    "Agricultural": ("leaf", "green"),
    "Burning": ("fire", "orange"),
    "Natural": ("cloud", "purple")
}

# =========================================
#  ADD MARKERS (ONE PER CITY)
# =========================================
for _, row in df_grouped.iterrows():

    icon_name, color = icon_map.get(row['main_source'], ("info-sign", "gray"))

    popup_html = f"""
    <b>City:</b> {row['city']}<br>
    <b>Main Source:</b> {row['main_source']}<br>
    <b>Avg PM2.5:</b> {row['pm2_5']:.2f}
    """

    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=popup_html,
        icon=folium.Icon(icon=icon_name, prefix='fa', color=color)
    ).add_to(m)

# =========================================
#  LEGEND
# =========================================
legend_html = """
<div style="
position: fixed; 
bottom: 40px; left: 40px; width: 230px;
background-color: white; 
border:2px solid grey; 
z-index:9999; 
padding: 12px;
border-radius:10px;
box-shadow:2px 2px 10px rgba(0,0,0,0.3);
font-size:14px;
">

<b>🌍 Pollution Sources</b><br><br>

🚗 Vehicular<br>
🏭 Industrial<br>
🌾 Agricultural<br>
🔥 Burning<br>
☁ Natural<br>

</div>
"""

m.get_root().html.add_child(folium.Element(legend_html))

# =========================================
#  SAVE MAP
# =========================================
m.save(OUTPUT_PATH)

print(f"\n Map saved at:\n{OUTPUT_PATH}")