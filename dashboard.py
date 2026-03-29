import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import smtplib
import folium
from email.mime.text import MIMEText
from streamlit_folium import st_folium

from folium.plugins import HeatMap

st.set_page_config(page_title="EnviroScan Dashboard", layout="wide")
st.title("🌍 EnviroScan - Air Pollution Monitoring Dashboard")
df = pd.read_csv("vijayawada_labelled_dataset.csv")

model = joblib.load("models/gradient_boost_pollution_model.pkl")
le = joblib.load("models/label_encoder.pkl")

FEATURE_COLS = [
    "pm2_5","pm10","no2","o3","so2","co",
    "road_count","industrial_count","waste_count","farmland_count"
]

st.sidebar.header("🔎 Filters")

location = st.sidebar.selectbox(
    "Select Location",
    sorted(df["location"].unique())
)

pollutant = st.sidebar.selectbox(
    "Select Pollutant",
    ["pm2_5","pm10","no2","o3","so2","co"]
)

filtered_df = df[df["location"] == location]

if filtered_df.empty:
    st.warning("No data available")
    st.stop()


tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🗺️ Map", "📈 Reports"])


with tab1:

    st.subheader("📊 Pollution Indicators")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("PM2.5", round(filtered_df["pm2_5"].mean(),2))
    col2.metric("PM10", round(filtered_df["pm10"].mean(),2))
    col3.metric("NO2", round(filtered_df["no2"].mean(),2))
    col4.metric("O3", round(filtered_df["o3"].mean(),2))
    # SLIDERS
    st.subheader("🎛️ Adjust Pollution Parameters")

    col1, col2 = st.columns(2)

    with col1:
        pm2_5 = st.slider("PM2.5", float(df["pm2_5"].min()), float(df["pm2_5"].max()), float(df["pm2_5"].mean()))
        pm10 = st.slider("PM10", float(df["pm10"].min()), float(df["pm10"].max()), float(df["pm10"].mean()))
        no2 = st.slider("NO2", float(df["no2"].min()), float(df["no2"].max()), float(df["no2"].mean()))
        o3 = st.slider("O3", float(df["o3"].min()), float(df["o3"].max()), float(df["o3"].mean()))
        so2 = st.slider("SO2", float(df["so2"].min()), float(df["so2"].max()), float(df["so2"].mean()))
        co = st.slider("CO", float(df["co"].min()), float(df["co"].max()), float(df["co"].mean()))

    with col2:
        road = st.slider("Road Count", int(df["road_count"].min()), int(df["road_count"].max()), int(df["road_count"].mean()))
        industrial = st.slider("Industrial Count", int(df["industrial_count"].min()), int(df["industrial_count"].max()), int(df["industrial_count"].mean()))
        waste = st.slider("Waste Count", int(df["waste_count"].min()), int(df["waste_count"].max()), int(df["waste_count"].mean()))
        farm = st.slider("Farmland Count", int(df["farmland_count"].min()), int(df["farmland_count"].max()), int(df["farmland_count"].mean()))

    # PREDICTION
    st.subheader("🤖 Predict Pollution Source")

    if st.button("Predict Source"):
        input_data = pd.DataFrame([[pm2_5, pm10, no2, o3, so2, co,
                                    road, industrial, waste, farm]],
                                  columns=FEATURE_COLS)

        pred = model.predict(input_data)
        proba = model.predict_proba(input_data)

        source = le.inverse_transform(pred)[0]
        confidence = max(proba[0]) * 100

        st.success(f"Predicted Source: {source}")
        st.info(f"Confidence: {confidence:.2f}%")


    st.subheader("📋 Detailed Pollution Report")

    table_df = df[df["location"] == location].copy()

    # Sort by selected pollutant
    table_df = table_df.sort_values(by=pollutant, ascending=False)

    table_df.reset_index(drop=True, inplace=True)

    st.write(f"Showing {len(table_df)} records for {location}")

    st.dataframe(table_df, use_container_width=True, height=400)

    
    # 📥 DOWNLOAD REPORT

    st.subheader("📥 Download Report")

    csv = table_df.to_csv(index=False)

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{location}_pollution_report.csv",
        mime="text/csv"
    )

    # 📩 EMAIL ALERT
    st.subheader("📩 Send Email Alert")

    if st.button("Send Email Alert"):
        try:
            msg = MIMEText(f"Pollution Alert in {location}")
            msg["Subject"] = "Pollution Alert"
            msg["From"] = "chandusst987@gmail.com"
            msg["To"] = "chandusst987@gmail.com"

            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login("chandusst987@gmail.com", "jiof corb rrbn gytq")
            server.send_message(msg)
            server.quit()

            st.success("Email Sent Successfully!")
        except Exception as e:
            st.error(f"Error: {e}")
        
with tab2:
    st.subheader("🗺️ Vijayawada Wide Coverage Map")
    
    
    locations_df = df.groupby('location')[FEATURE_COLS].mean(numeric_only=True)
    
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
    
    map_data = []
    loc_indices = list(locations_df.index)
    
    for i, (lat, lon) in enumerate(wide_points):
        data_loc = loc_indices[i % len(loc_indices)]
        row_data = locations_df.loc[data_loc]
        map_data.append({
            'lat': lat, 'lon': lon, 'location': location_names[i], **row_data.to_dict()
        })
    
    map_df = pd.DataFrame(map_data)
    
    
    X_map = map_df[FEATURE_COLS]
    preds = model.predict(X_map)
    probas = model.predict_proba(X_map)
    map_df['predicted_source'] = le.inverse_transform(preds)
    map_df['confidence'] = [max(p)*100 for p in probas]
    
    # MAP CREATION
    m = folium.Map(location=[16.506, 80.648], zoom_start=11)
    
    # Heatmap
    heat_data = [[row["lat"], row["lon"], float(row[pollutant])] for _, row in map_df.iterrows()]
    HeatMap(heat_data, radius=30, blur=20).add_to(m)
    
    # Icons dictionary
    source_icons = {
        "Industrial": "🏭", "Vehicular": "🚗", "Agricultural": "🌾", 
        "Burning": "🔥", "Natural": "🌿"
    }
    
    
    for _, row in map_df.iterrows():
        source = row['predicted_source']
        confidence = row['confidence']
        icon_html = source_icons.get(source, "📍")
        
        popup_html = f"""
        <div style="font-family: Arial; width: 250px">
            <h4><b>📍 {row['location']}</b></h4>
            <b>🔥 Source:</b> {source}<br>
            <b>📊 Confidence:</b> <span style="color: #4CAF50; font-weight: bold">{confidence:.1f}%</span><br>
        </div>
        """
        
        
        folium.Marker(
            [row["lat"], row["lon"]],
            popup=folium.Popup(popup_html, max_width=280),
            icon=folium.DivIcon(html=f'<div style="font-size: 24px; padding: 8px">{icon_html}</div>')
        ).add_to(m)
    

    st_folium(m, width=1000, height=800, key="vijayawada_map_final")


with tab3:

    st.subheader("📊 Reports & Analysis")

    all_sources = ["Industrial","Vehicular","Agricultural","Burning","Natural"]

    source_counts = filtered_df["pollution_source"] \
        .value_counts() \
        .reindex(all_sources, fill_value=0)

    # TABLE
    st.write("### 📋 Source Table")
    table_df = source_counts.reset_index()
    table_df.columns = ["Source", "Count"]
    st.dataframe(table_df)

    # PIE
    # PIE (FIXED CLEAN VERSION)
    st.write("### 🥧 Pie Chart")

# ❌ Remove zero values
    clean_counts = source_counts[source_counts > 0]

    labels = clean_counts.index
    sizes = clean_counts.values

    colors = ["purple", "blue", "green", "red", "orange"]

# ✅ Hide small % text
    def autopct_format(pct):
    # Always show Vehicular OR show if > 2%
        if pct > 2:
          return f'{pct:.1f}%'
          

    fig, ax = plt.subplots(figsize=(5,5))

    wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    colors=colors[:len(labels)],
    autopct="%1.1f%%",   # ✅ show all %
    startangle=140,
    pctdistance=0.75,
    labeldistance=1.1
)

# Reduce font to avoid overlap
    for text in texts:
      text.set_fontsize(9)

    for autotext in autotexts:
      autotext.set_fontsize(8)

    ax.set_title("Pollution Source Distribution")
    plt.tight_layout()
    st.pyplot(fig)

    # LINE
    st.write("### 📈 Trend")
    trend = filtered_df.groupby("hour")[pollutant].mean()
    st.line_chart(trend)
 
