import pandas as pd
import joblib
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt 
import shap
import os
import numpy as np 
# ================= LOAD =================
model = joblib.load("models/gradient_boost_pollution_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

data = pd.read_csv("vij_hyd_labelled_dataset.csv")
data.columns = data.columns.str.strip().str.lower()

# ================= FEATURES =================
FEATURE_COLS = [
    "pm2_5", "pm10", "no2", "o3", "so2", "co",
    "road_count", "industrial_count", "waste_count", "farmland_count"
]

data = data.dropna(subset=["lat", "lon"]).copy()

# ================= PREDICTION =================
pred = model.predict(data[FEATURE_COLS])
proba = model.predict_proba(data[FEATURE_COLS])

data["predicted_source"] = label_encoder.inverse_transform(pred)
data["confidence"] = proba.max(axis=1) * 100

# ================= SEVERITY =================
weights = {'pm2_5': 0.4, 'pm10': 0.25, 'no2': 0.15, 'o3': 0.1, 'so2': 0.05, 'co': 0.05}

data["severity"] = data.apply(
    lambda r: sum(r[p] * weights[p] for p in weights),
    axis=1
)

max_sev = data["severity"].max()
data["severity_norm"] = data["severity"] / max_sev if max_sev != 0 else 0

# ================= MAP =================
m = folium.Map(
    location=[data["lat"].mean(), data["lon"].mean()],
    zoom_start=11,
    tiles="CartoDB positron"
)

# ================= LAYERS =================
heat_layer = folium.FeatureGroup(name="Severity Heatmap", show=True)
all_markers_layer = folium.FeatureGroup(name="All Source Markers", show=True)

source_colors = {
    "Industrial": "red",
    "Vehicular": "blue",
    "Agricultural": "green",
    "Burning": "orange",
    "Natural": "purple"
}

source_layers = {}
for source in source_colors.keys():
    source_layers[source] = folium.FeatureGroup(name=f"{source} Markers", show=False)

# ================= HEATMAP =================
HeatMap(
    data[["lat", "lon", "severity_norm"]].values.tolist(),
    radius=25,
    blur=30,
    name="Severity Heatmap"
).add_to(heat_layer)

heat_layer.add_to(m)

# ================= SEVERITY → SIZE =================
def get_radius(sev):
    if sev > 0.8:
        return 10
    elif sev > 0.6:
        return 8
    elif sev > 0.4:
        return 6
    elif sev > 0.2:
        return 5
    else:
        return 4

# ================= ADD MARKERS =================
for _, row in data.iterrows():
    popup = f"""
    <b>Location:</b> {row.get('location', 'N/A')}<br>
    <b>Predicted Source:</b> {row['predicted_source']}<br>
    <b>Confidence:</b> {row['confidence']:.2f}%<br>
    <b>Severity:</b> {row['severity']:.2f}
    """

    marker = folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=get_radius(row["severity_norm"]),
        color=source_colors.get(row["predicted_source"], "gray"),
        fill=True,
        fill_color=source_colors.get(row["predicted_source"], "gray"),
        fill_opacity=0.7,
        popup=folium.Popup(popup, max_width=300)
    )

    marker.add_to(all_markers_layer)

    if row["predicted_source"] in source_layers:
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=get_radius(row["severity_norm"]),
            color=source_colors.get(row["predicted_source"], "gray"),
            fill=True,
            fill_color=source_colors.get(row["predicted_source"], "gray"),
            fill_opacity=0.7,
            popup=folium.Popup(popup, max_width=300)
        ).add_to(source_layers[row["predicted_source"]])

all_markers_layer.add_to(m)

for layer in source_layers.values():
    layer.add_to(m)

# ================= LEGEND =================
legend_html = '''
<div style="
position: fixed;
bottom: 30px; left: 50px; width: 250px; height: 200px;
background-color: white; z-index:9999; font-size:14px;
border:2px solid grey; padding: 10px;">
<b>Pollution Source</b><br>
<span style="color:red;">●</span> Industrial<br>
<span style="color:blue;">●</span> Vehicular<br>
<span style="color:green;">●</span> Agricultural<br>
<span style="color:orange;">●</span> Burning<br>
<span style="color:purple;">●</span> Natural<br>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# ================= TOGGLE CONTROL =================
folium.LayerControl(collapsed=False).add_to(m)

# ================= SAVE =================
m.save("map.html")

print("Map Created Successfully with Toggle Layers!")


# ================= WATERFALL (PER SOURCE) =================

print("\n🔍 Generating SHAP Plots...")

# ================= CREATE FOLDER =================
os.makedirs("plots", exist_ok=True)

# ================= SAMPLE DATA =================
sample_data = data[FEATURE_COLS].sample(50, random_state=42)

# ================= BACKGROUND =================
background = shap.kmeans(sample_data, 10)

# ================= EXPLAINER =================
explainer = shap.KernelExplainer(model.predict_proba, background)

# ================= WATERFALL =================
unique_sources = data["predicted_source"].unique()

for src in unique_sources:
    try:
        idx = data[data["predicted_source"] == src].index[0]

        row_df = data.loc[[idx], FEATURE_COLS]

        shap_values = explainer.shap_values(row_df, nsamples=100)

        if isinstance(shap_values, list):
            class_idx = np.argmax(model.predict_proba(row_df)[0])

            # ✅ Take first sample and flatten
            sv = np.array(shap_values[class_idx])[0].flatten()

            # ✅ Convert base value to scalar safely
            base_val = float(np.array(explainer.expected_value).reshape(-1)[class_idx])

        else:
            sv = np.array(shap_values)[0].flatten()
            base_val = float(np.array(explainer.expected_value).reshape(-1)[0])

        # ✅ Ensure correct shape
        sv = sv[:len(FEATURE_COLS)]

        exp = shap.Explanation(
            values=sv,
            base_values=base_val,
            data=row_df.iloc[0].values,
            feature_names=FEATURE_COLS
        )

        plt.figure(figsize=(10,6))
        shap.plots.waterfall(exp, show=False)

        plt.title(f"{src} Source")
        plt.tight_layout()

        plt.savefig(f"plots/waterfall_{src}.png")
        plt.close()

        print(f"✅ {src}")

    except Exception as e:
        print(f"⚠️ {src} failed: {e}")

# ================= BEESWARM =================
print("\n🐝 Generating Beeswarm Plot...")

try:
    shap_vals = explainer.shap_values(sample_data, nsamples=100)

    plt.figure(figsize=(12,7))
    shap.summary_plot(shap_vals, sample_data, show=False)

    plt.savefig("plots/beeswarm.png", bbox_inches="tight")
    plt.close()

    print("✅ Beeswarm saved")

except Exception as e:
    print(f"⚠️ Beeswarm failed: {e}")