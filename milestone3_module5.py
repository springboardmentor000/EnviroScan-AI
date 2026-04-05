import pandas as pd
import folium
from folium.plugins import HeatMap
import os

print("=" * 60)
print("  EnviroScan — Module 5: Geospatial Mapping & Heatmaps")
print("=" * 60)

os.makedirs("outputs", exist_ok=True)

# -------------------------------------------------
# STEP 1: Load labeled data (real stations only)
# -------------------------------------------------
print("\n STEP 1: Loading labeled data...")
df = pd.read_csv("Hyderabad_pollution_labeled.csv")

# Keep only real stations (not simulated)
real_df = df[~df["station_id"].astype(str).str.startswith("SIM")].copy()
print(f"   Total records : {len(df)}")
print(f"   Real stations : {len(real_df)}")

# Fill missing values with median
for col in ["pm25", "pm10", "no2", "so2", "o3", "co"]:
    real_df[col] = real_df[col].fillna(real_df[col].median())

# -------------------------------------------------
# SOURCE COLORS AND ICONS
# -------------------------------------------------
SOURCE_COLORS = {
    "Vehicular":    "#E74C3C",   # Red
    "Industrial":   "#3498DB",   # Blue
    "Agricultural": "#2ECC71",   # Green
    "Burning":      "#F39C12",   # Orange
    "Natural":      "#9B59B6",   # Purple
}

SOURCE_ICONS = {
    "Vehicular":    "car",
    "Industrial":   "industry",
    "Agricultural": "leaf",
    "Burning":      "fire",
    "Natural":      "tree",
}

SOURCE_EMOJIS = {
    "Vehicular":    "",
    "Industrial":   "",
    "Agricultural": "",
    "Burning":      "",
    "Natural":      "",
}

def aqi_color(pm25):
    if pm25 <= 30:    return "#2ECC71"   # Green — Good
    elif pm25 <= 60:  return "#F1C40F"   # Yellow — Satisfactory
    elif pm25 <= 90:  return "#F39C12"   # Orange — Moderate
    elif pm25 <= 120: return "#E74C3C"   # Red — Poor
    elif pm25 <= 250: return "#8E44AD"   # Purple — Very Poor
    else:             return "#2C3E50"   # Dark — Severe

def aqi_label(pm25):
    if pm25 <= 30:    return "Good"
    elif pm25 <= 60:  return "Satisfactory"
    elif pm25 <= 90:  return "Moderate"
    elif pm25 <= 120: return "Poor"
    elif pm25 <= 250: return "Very Poor"
    else:             return "Severe"

# -------------------------------------------------
# MAP 1: PM2.5 Heatmap
# -------------------------------------------------
print("\n  STEP 2: Creating PM2.5 Heatmap...")

m1 = folium.Map(
    location=[17.3850, 78.4867],
    zoom_start=11,
    tiles="CartoDB positron"
)

# Add heatmap layer
heat_data = [[row["latitude"], row["longitude"], row["pm25"]]
             for _, row in real_df.iterrows()]
HeatMap(
    heat_data,
    radius=40,
    blur=25,
    max_zoom=13,
    gradient={"0.2": "blue", "0.4": "lime", "0.6": "yellow", "0.8": "orange", "1.0": "red"}
).add_to(m1)

# Add station markers
for _, row in real_df.iterrows():
    color  = aqi_color(row["pm25"])
    label  = aqi_label(row["pm25"])
    popup_html = f"""
    <div style="font-family:Arial; width:220px;">
        <h4 style="color:#1F5C99; margin:0 0 8px 0;">{row['station_name']}</h4>
        <hr style="margin:4px 0;">
        <b>PM2.5:</b> {row['pm25']} µg/m³ &nbsp;
        <span style="background:{color};color:white;padding:2px 6px;border-radius:4px;font-size:11px;">{label}</span><br>
        <b>PM10:</b> {round(row['pm10'],1)} µg/m³<br>
        <b>NO2:</b> {round(row['no2'],1)} µg/m³<br>
        <b>SO2:</b> {round(row['so2'],1)} µg/m³<br>
        <b>CO:</b> {round(row['co'],2)} mg/m³<br>
        <b>Temp:</b> {row['temp']}°C &nbsp; <b>Humidity:</b> {row['humidity']}%<br>
        <hr style="margin:4px 0;">
        <b>Source:</b> {row['pollution_source']}<br>
        <b>Confidence:</b> {int(row['source_confidence']*100)}%
    </div>
    """
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=10,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=f"{row['station_name']} — PM2.5: {row['pm25']} µg/m³"
    ).add_to(m1)

# Legend
legend_html = """
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
            padding:12px;border-radius:8px;border:2px solid #ccc;font-family:Arial;font-size:12px;">
    <b>PM2.5 AQI Levels</b><br>
    <span style="background:#2ECC71;color:white;padding:1px 8px;border-radius:3px;">Good</span> 0-30 µg/m³<br>
    <span style="background:#F1C40F;color:white;padding:1px 8px;border-radius:3px;">Satisfactory</span> 31-60<br>
    <span style="background:#F39C12;color:white;padding:1px 8px;border-radius:3px;">Moderate</span> 61-90<br>
    <span style="background:#E74C3C;color:white;padding:1px 8px;border-radius:3px;">Poor</span> 91-120<br>
    <span style="background:#8E44AD;color:white;padding:1px 8px;border-radius:3px;">Very Poor</span> 121-250<br>
</div>
"""
m1.get_root().html.add_child(folium.Element(legend_html))
m1.save("outputs/map1_pm25_heatmap.html")
print("    Saved → outputs/map1_pm25_heatmap.html")

# -------------------------------------------------
# MAP 2: Pollution Source Map
# -------------------------------------------------
print("\n  STEP 3: Creating Pollution Source Map...")

m2 = folium.Map(
    location=[17.3850, 78.4867],
    zoom_start=11,
    tiles="CartoDB positron"
)

for _, row in real_df.iterrows():
    source = row["pollution_source"]
    color  = SOURCE_COLORS.get(source, "#999999")
    emoji  = SOURCE_EMOJIS.get(source, "")
    conf   = int(row["source_confidence"] * 100)

    popup_html = f"""
    <div style="font-family:Arial; width:230px;">
        <h4 style="color:#1F5C99; margin:0 0 8px 0;">{row['station_name']}</h4>
        <hr style="margin:4px 0;">
        <b>Predicted Source:</b><br>
        <span style="background:{color};color:white;padding:3px 10px;
                     border-radius:5px;font-size:13px;">{source}</span><br><br>
        <b>Confidence:</b> {conf}%<br>
        <b>PM2.5:</b> {row['pm25']} µg/m³
        <span style="background:{aqi_color(row['pm25'])};color:white;
                     padding:1px 6px;border-radius:3px;font-size:11px;">
            {aqi_label(row['pm25'])}</span><br>
        <b>NO2:</b> {round(row['no2'],1)} µg/m³ &nbsp;
        <b>SO2:</b> {round(row['so2'],1)} µg/m³<br>
        <b>Road count:</b> {int(row['road_count'])} &nbsp;
        <b>Industrial zones:</b> {int(row['industrial_zone_count'])}
    </div>
    """

    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        popup=folium.Popup(popup_html, max_width=260),
        tooltip=f"{source} — {row['station_name']}",
        icon=folium.Icon(color="white", icon_color=color, icon="circle", prefix="fa")
    ).add_to(m2)

    # Colored circle behind marker
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=14,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.5,
    ).add_to(m2)

# Source legend
source_legend = """
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
            padding:12px;border-radius:8px;border:2px solid #ccc;font-family:Arial;font-size:12px;">
    <b>Pollution Sources</b><br>
    <span style="color:#E74C3C;">●</span> Vehicular<br>
    <span style="color:#3498DB;">●</span> Industrial<br>
    <span style="color:#2ECC71;">●</span> Agricultural<br>
    <span style="color:#F39C12;">●</span> Burning<br>
    <span style="color:#9B59B6;">●</span> Natural<br>
</div>
"""
m2.get_root().html.add_child(folium.Element(source_legend))
m2.save("outputs/map2_pollution_sources.html")
print("    Saved → outputs/map2_pollution_sources.html")

# -------------------------------------------------
# MAP 3: Combined Map (both layers)
# -------------------------------------------------
print("\n  STEP 4: Creating Combined Map...")

m3 = folium.Map(
    location=[17.3850, 78.4867],
    zoom_start=11,
    tiles="CartoDB positron"
)

# Heatmap layer
HeatMap(
    heat_data,
    radius=40,
    blur=25,
    max_zoom=13,
    gradient={"0.2": "blue", "0.4": "lime", "0.6": "yellow", "0.8": "orange", "1.0": "red"},
    name="PM2.5 Heatmap"
).add_to(m3)

# Source markers layer
source_group = folium.FeatureGroup(name="Pollution Sources")
for _, row in real_df.iterrows():
    source = row["pollution_source"]
    color  = SOURCE_COLORS.get(source, "#999999")
    emoji  = SOURCE_EMOJIS.get(source, "")

    popup_html = f"""
    <div style="font-family:Arial; width:230px;">
        <h4 style="color:#1F5C99; margin:0 0 8px 0;">{row['station_name']}</h4>
        <hr style="margin:4px 0;">
        <b>Source:</b>
        <span style="background:{color};color:white;padding:2px 8px;border-radius:4px;">
            {source}</span><br>
        <b>Confidence:</b> {int(row['source_confidence']*100)}%<br>
        <b>PM2.5:</b> {row['pm25']} µg/m³
        <span style="background:{aqi_color(row['pm25'])};color:white;
                     padding:1px 6px;border-radius:3px;font-size:11px;">
            {aqi_label(row['pm25'])}</span><br>
        <b>NO2:</b> {round(row['no2'],1)} &nbsp;
        <b>SO2:</b> {round(row['so2'],1)} &nbsp;
        <b>CO:</b> {round(row['co'],2)}<br>
        <b>Temp:</b> {row['temp']}°C &nbsp;
        <b>Wind:</b> {row['wind_speed']} m/s
    </div>
    """

    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=12,
        color="white",
        weight=2,
        fill=True,
        fill_color=color,
        fill_opacity=0.9,
        popup=folium.Popup(popup_html, max_width=260),
        tooltip=f"{source} | PM2.5: {row['pm25']} | {row['station_name']}"
    ).add_to(source_group)

source_group.add_to(m3)
folium.LayerControl().add_to(m3)

# Combined legend
combined_legend = """
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
            padding:12px;border-radius:8px;border:2px solid #ccc;font-family:Arial;font-size:12px;width:160px;">
    <b>Pollution Sources</b><br>
    <span style="color:#E74C3C;">●</span> Vehicular<br>
    <span style="color:#3498DB;">●</span> Industrial<br>
    <span style="color:#2ECC71;">●</span> Agricultural<br>
    <span style="color:#F39C12;">●</span> Burning<br>
    <span style="color:#9B59B6;">●</span> Natural<br>
    <hr style="margin:6px 0;">
    <b>PM2.5 Heatmap</b><br>
    <span style="color:blue;">■</span> Low &nbsp;
    <span style="color:lime;">■</span> Med &nbsp;
    <span style="color:red;">■</span> High
</div>
"""
m3.get_root().html.add_child(folium.Element(combined_legend))
m3.save("outputs/map3_combined.html")
print("    Saved → outputs/map3_combined.html")

# -------------------------------------------------
# SUMMARY
# -------------------------------------------------
print(f"\n{'='*60}")
print(" Module 5 Complete! 3 maps generated:\n")
print("    map1_pm25_heatmap.html   — PM2.5 heatmap across Hyderabad")
print("    map2_pollution_sources.html — Source type per station")
print("    map3_combined.html       — Both layers combined")
print("\n    Open any .html file in your browser to view the map!")
print(f"{'='*60}")