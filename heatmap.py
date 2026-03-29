import pandas as pd
import joblib
import folium
from folium.plugins import HeatMap
import shap
import numpy as np
import matplotlib.pyplot as plt
import os

# ================= LOAD MODEL =================
model = joblib.load("models/gradient_boost_pollution_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# ================= LOAD DATA =================
data = pd.read_csv("vijayawada_labelled_dataset.csv")

FEATURE_COLS = [
    "pm2_5","pm10","no2","o3","so2","co",
    "road_count","industrial_count","waste_count","farmland_count"
]

# ================= LOCATIONS =================
wide_points = [
    [16.4985, 80.6575], [16.5020, 80.6520], [16.5050, 80.6480], 
    [16.5150, 80.6450], [16.5200, 80.6350], [16.5350, 80.6200],
    [16.4700, 80.6800], [16.5050, 80.6300], [16.5500, 80.6100], 
    [16.5100, 80.6420], [16.5180, 80.6380], [16.4900, 80.6750]
]

location_names = [
    "Benz Circle", "Governorpet", "Vijayawada Central", "Patamata", 
    "Moghalrajpuram", "Bhavanipuram", "Vaddeswaram", "Autonagar",
    "Gannavaram", "One Town", "Krishna Lanka", "Enikepadu"
]

# ================= PREP DATA =================
selected = data.sample(len(wide_points), random_state=42).copy()
selected["location"] = location_names
selected["latitude"] = [p[0] for p in wide_points]
selected["longitude"] = [p[1] for p in wide_points]
# ================= ADD 2 DATASET LOCATIONS =================
# Take 2 real locations from dataset
dataset_locs = data.drop_duplicates(subset=["location"]).head(2).copy()

dataset_locs["latitude"] = dataset_locs["lat"]
dataset_locs["longitude"] = dataset_locs["lon"]

# Keep same structure as selected
dataset_locs["location"] = dataset_locs["location"]

# Combine with your existing selected data
selected = pd.concat([selected, dataset_locs], ignore_index=True)
# ================= PREDICTION =================
predictions = model.predict(selected[FEATURE_COLS])
probas = model.predict_proba(selected[FEATURE_COLS])

selected["predicted_source"] = label_encoder.inverse_transform(predictions)
selected["confidence"] = np.max(probas, axis=1) * 100

# ================= SEVERITY =================
weights = {'pm2_5':0.4,'pm10':0.25,'no2':0.15,'o3':0.1,'so2':0.05,'co':0.05}
selected["severity"] = selected.apply(
    lambda r: sum(r[p]*weights[p]*100 for p in weights),
    axis=1
)

# ================= MAP =================
# ================= MAP =================
m = folium.Map(location=[16.50, 80.65], zoom_start=12, tiles="OpenStreetMap")

# ================= HEATMAP LAYER =================
heat_layer = folium.FeatureGroup(name="🔥 Heatmap")

max_sev = selected["severity"].max()

HeatMap(
    [[row.latitude, row.longitude, row.severity/max_sev]
     for _, row in selected.iterrows()],
    radius=30,          # bigger glow
    blur=25,            # smooth glow
    min_opacity=0.5
).add_to(heat_layer)

heat_layer.add_to(m)

# ================= EMOJI ICONS =================
emoji_icons = {
    "Industrial": "🏭",
    "Vehicular": "🚗",
    "Agricultural": "🌾",
    "Burning": "🔥",
    "Natural": "🌿"
}

# ================= SOURCE LAYER =================
marker_layer = folium.FeatureGroup(name="📍 Pollution Sources")

# ================= SOURCE-WISE TOGGLE LAYERS =================
source_layers = {}

for src in selected["predicted_source"].unique():
    source_layers[src] = folium.FeatureGroup(name=f"{src} Sources")

for _, row in selected.iterrows():
    src = row["predicted_source"]
    emoji = emoji_icons.get(src, "📍")

    popup = f"""
    <div style="width:220px">
        <h4>{emoji} {row['location']}</h4>
        <b>Source:</b> {src}<br>
        <b>Severity:</b> <span style="color:red">{row['severity']:.2f}</span><br>
        <b>Confidence:</b> {row['confidence']:.1f}%<br>
    </div>
    """

    layer = source_layers[src]

    # Glow effect
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=18,
        color=None,
        fill=True,
        fill_color="red",
        fill_opacity=0.2
    ).add_to(layer)

    # Emoji marker
    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        popup=folium.Popup(popup, max_width=250),
        tooltip=f"{emoji} {row['location']}",
        icon=folium.DivIcon(
            html=f"""<div style="font-size:20px">{emoji}</div>"""
        )
    ).add_to(layer)

# Add all layers
for layer in source_layers.values():
    layer.add_to(m)

    # 🔥 GLOW EFFECT (like your image)
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=18,
        color=None,
        fill=True,
        fill_color="red",
        fill_opacity=0.2
    ).add_to(marker_layer)

    # 🎯 MAIN MARKER (emoji)
    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        popup=folium.Popup(popup, max_width=250),
        tooltip=f"{emoji} {row['location']}",
        icon=folium.DivIcon(
            html=f"""<div style="font-size:20px">{emoji}</div>"""
        )
    ).add_to(marker_layer)

marker_layer.add_to(m)

# ================= LEGEND =================
legend_html = """
<div style="
position: fixed; 
bottom: 50px; left: 50px; width: 200px; height: 150px; 
background-color: white; 
border-radius: 10px;
padding: 10px;
z-index:9999;
font-size:14px;
box-shadow: 2px 2px 6px gray;
">
<b>🌍 Pollution Sources</b><br>
🏭 Industrial<br>
🚗 Vehicular<br>
🌾 Agricultural<br>
🔥 Burning<br>
🌿 Natural
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# ================= CONTROLS =================
folium.LayerControl(collapsed=False).add_to(m)

# ================= SAVE =================
m.save("map.html")
print("✅ Map ready like your image!")

# ================= SHAP (UNCHANGED - YOUR WORKING VERSION) =================
print("\n🔍 SHAP Waterfalls...")

os.makedirs("plots", exist_ok=True)

background = shap.kmeans(data[FEATURE_COLS], 20)
explainer = shap.KernelExplainer(model.predict_proba, background.data)

unique_sources = selected["predicted_source"].unique()

for src in unique_sources:
    try:
        sample_row = selected[selected["predicted_source"] == src].iloc[0]

        sample_features = pd.DataFrame(
            [sample_row[FEATURE_COLS].values],
            columns=FEATURE_COLS
        )

        shap_values = explainer.shap_values(sample_features, nsamples=100)

        proba = model.predict_proba(sample_features)[0]
        class_idx = np.argmax(proba)

        if isinstance(shap_values, list):
            sv = shap_values[class_idx][0]
            base_val = explainer.expected_value[class_idx]
        else:
            sv = shap_values[0]
            base_val = explainer.expected_value

        sv = np.array(sv).flatten()[:len(FEATURE_COLS)]

        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(np.array(base_val).flatten()[0])
        else:
            base_val = float(base_val)

        exp = shap.Explanation(
            values=sv,
            base_values=base_val,
            data=sample_features.iloc[0],
            feature_names=FEATURE_COLS
        )

        plt.figure(figsize=(12,7))
        shap.plots.waterfall(exp, show=False)

        plt.title(f"{src} Source")
        plt.tight_layout()

        plt.savefig(f"plots/source_{src}.png")
        plt.close()

        print(f"✅ {src}")

    except Exception as e:
        print(f"⚠️ {src} failed: {e}")
# ================= BEESWARM =================
print("🐝 Source summary...")

try:
    sample_df = data[FEATURE_COLS].sample(100, random_state=42)
    shap_vals = explainer.shap_values(sample_df, nsamples=100)

    plt.figure(figsize=(14,8))
    shap.summary_plot(shap_vals, sample_df, show=False)

    plt.savefig("plots/source_beeswarm.png", bbox_inches='tight', dpi=150)
    plt.close()

    print("✅ Beeswarm saved")

except Exception as e:
    print(f"⚠️ Beeswarm skipped: {e}")