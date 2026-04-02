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
df = pd.read_csv("vij_hyd_labelled_dataset.csv")

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


tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard", 
    "🗺️ Map", 
    "📈 Reports",
    "🤖 Chatbot"
])


with tab1:

    st.subheader("📊 Pollution Indicators")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("PM2.5", round(filtered_df["pm2_5"].mean(),2))
    col2.metric("PM10", round(filtered_df["pm10"].mean(),2))
    col3.metric("NO2", round(filtered_df["no2"].mean(),2))
    col4.metric("O3", round(filtered_df["o3"].mean(),2))
    # ================= AQI INDICATOR =================
    st.subheader("🌡️ Air Quality Index (AQI)")

    def get_aqi_category(pm25):
        if pm25 <= 0.10:
           return "Good 🟢"
        elif pm25 <= 0.15:
           return "Moderate 🟡"
        elif pm25 <= 0.20:
           return "Unhealthy for Sensitive 🟠"
        elif pm25 <= 0.50:
           return "Poor 🔴"
        else:
           return "Hazardous ⚫"

    avg_pm25 = filtered_df["pm2_5"].mean()
    aqi_status = get_aqi_category(avg_pm25)

    st.metric("AQI Status", aqi_status)
    st.write(f"Average PM2.5: {avg_pm25:.2f}")
    # SLIDERS
    
    st.subheader("🩺 Health Recommendations")

    def health_advice(pm25):
        if pm25 <= 0.10:
            return "✅ Air quality is good. Enjoy outdoor activities 🌿"
    
        elif pm25 <= 0.15:
            return "Moderate air quality. Sensitive people should reduce prolonged outdoor exposure."
    
        elif pm25 <= 0.20:
           return "⚠️ Unhealthy for sensitive groups. Wear mask  and avoid long outdoor stays."
    
        elif pm25 <= 0.50:
           return "🚨 Poor air quality. Limit outdoor activities. Use air purifiers indoors."
    
        else:
           return "☠️ Hazardous! Stay indoors. Avoid travel. Use N95 masks."

    advice = health_advice(avg_pm25)

    st.warning(advice)
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
    st.subheader("🗺️ Map")

    # ================= LOAD =================
    data = pd.read_csv("vij_hyd_labelled_dataset.csv")
    data.columns = data.columns.str.strip().str.lower()

    data = data.dropna(subset=["lat", "lon"]).copy()

    # ================= PREDICTION =================
    pred = model.predict(data[FEATURE_COLS])
    proba = model.predict_proba(data[FEATURE_COLS])

    data["predicted_source"] = le.inverse_transform(pred)
    data["confidence"] = proba.max(axis=1) * 100

    # ================= SEVERITY =================
    weights = {'pm2_5':0.4,'pm10':0.25,'no2':0.15,'o3':0.1,'so2':0.05,'co':0.05}

    data["severity"] = data.apply(
        lambda r: sum(r[p]*weights[p] for p in weights),
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

    # ================= HEATMAP =================
    HeatMap(
        data[["lat","lon","severity_norm"]].values.tolist(),
        radius=25,
        blur=30
    ).add_to(m)

    # ================= SOURCE COLORS =================
    source_colors = {
        "Industrial": "red",
        "Vehicular": "blue",
        "Agricultural": "green",
        "Burning": "orange",
        "Natural": "purple"
    }

    # ================= MARKERS =================
    for _, row in data.iterrows():

        popup = f"""
        <b>Location:</b> {row.get('location','N/A')}<br>
        <b>Source:</b> {row['predicted_source']}<br>
        <b>Confidence:</b> {row['confidence']:.2f}%<br>
        <b>Severity:</b> {row['severity']:.2f}
        """

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            color=source_colors.get(row["predicted_source"], "gray"),
            fill=True,
            fill_opacity=0.7,
            popup=popup
        ).add_to(m)

    # ================= LEGEND =================
    legend_html = '''
    <div style="
    position: fixed;
    bottom: 30px; left: 50px; width: 250px;
    background-color: white; z-index:9999;
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

    # ================= DISPLAY =================
    st_folium(m, width=1000, height=900)


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

    # ================= TOP LOCATIONS =================
    st.subheader("Top Polluted Locations (PM2.5)")

    top_locations = (
       df.groupby("location")["pm2_5"]
       .mean()
       .sort_values(ascending=False)
       .head(5)
)

    st.bar_chart(top_locations)

# Optional table view
    top_df = top_locations.reset_index()
    top_df.columns = ["Location", "Avg PM2.5"]
    st.dataframe(top_df)
    
    
with tab4:
    import streamlit as st
    import pandas as pd
    import re

    st.subheader("🤖 EnviroScan Assistant (Chatbot)")

    # ================= PREPARE DATA FOR CHATBOT =================
    chatbot_df = df.copy()

    # Normalize required columns safely
    if "timestamp" in chatbot_df.columns:
        chatbot_df["timestamp"] = pd.to_datetime(chatbot_df["timestamp"], errors="coerce")
        chatbot_df["date_only"] = chatbot_df["timestamp"].dt.date
        chatbot_df["hour_only"] = chatbot_df["timestamp"].dt.hour
    elif "date" in chatbot_df.columns:
        chatbot_df["date_only"] = pd.to_datetime(chatbot_df["date"], errors="coerce").dt.date
        if "hour" in chatbot_df.columns:
            chatbot_df["hour_only"] = pd.to_numeric(chatbot_df["hour"], errors="coerce")
    else:
        chatbot_df["date_only"] = pd.NaT
        chatbot_df["hour_only"] = pd.NA

    if "pm2_5" in chatbot_df.columns:
        chatbot_df["pm2_5"] = pd.to_numeric(chatbot_df["pm2_5"], errors="coerce")

    if "hour" in chatbot_df.columns:
        chatbot_df["hour"] = pd.to_numeric(chatbot_df["hour"], errors="coerce")

    chatbot_df["location_lower"] = chatbot_df["location"].astype(str).str.lower().str.strip()

    # ================= CHAT MEMORY =================
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ================= HELPER FUNCTIONS =================
    def extract_date(query):
        match = re.search(r"(\d{4}-\d{2}-\d{2})", query)
        if match:
            try:
                return pd.to_datetime(match.group(1)).date()
            except:
                return None
        return None

    def extract_hour(query):
        patterns = [
            r"\b(?:at|hour)\s*(\d{1,2})\b",
            r"\b(\d{1,2}):00\b",
            r"\b(\d{1,2})\s*(?:am|pm)\b"
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                hour_val = int(match.group(1))

                ampm_match = re.search(r"\b(\d{1,2})\s*(am|pm)\b", query)
                if ampm_match:
                    h = int(ampm_match.group(1))
                    meridian = ampm_match.group(2)
                    if meridian == "pm" and h != 12:
                        hour_val = h + 12
                    elif meridian == "am" and h == 12:
                        hour_val = 0
                    else:
                        hour_val = h

                if 0 <= hour_val <= 23:
                    return hour_val
        return None

    def detect_location(query, data):
        locations = sorted(data["location"].dropna().astype(str).unique(), key=len, reverse=True)
        query_lower = query.lower().strip()

        for loc in locations:
            if loc.lower() in query_lower:
                return loc
        return None

    def classify_aqi(pm25):
        if pd.isna(pm25):
            return "Unknown"
        elif pm25 <= 50:
            return "🟢 Good AQI"
        elif pm25 <= 100:
            return "🟡 Moderate AQI"
        elif pm25 <= 150:
            return "🟠 Unhealthy"
        elif pm25 <= 200:
            return "🔴 Poor"
        else:
            return "⚫ Hazardous"

    # ================= CHATBOT FUNCTION =================
    def chatbot_response(query):
        q = query.lower().strip()

        try:
            # ================= MOST POLLUTED =================
            if "most polluted" in q:
                grp = chatbot_df.groupby("location", dropna=True)["pm2_5"].mean().dropna()
                if grp.empty:
                    return "No pollution data available."
                loc = grp.idxmax()
                val = grp.max()
                return f"🚨 Most polluted location is **{loc}** (Avg PM2.5 = {val:.2f})"

            # ================= SAFEST =================
            elif "safe" in q or "least polluted" in q or "safest" in q:
                grp = chatbot_df.groupby("location", dropna=True)["pm2_5"].mean().dropna()
                if grp.empty:
                    return "No pollution data available."
                loc = grp.idxmin()
                val = grp.min()
                return f"✅ Safest location is **{loc}** (Avg PM2.5 = {val:.2f})"

            # ================= TOP POLLUTED =================
            elif "top" in q and ("polluted" in q or "pollution" in q):
                top5 = (
                    chatbot_df.groupby("location", dropna=True)["pm2_5"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(5)
                )
                if top5.empty:
                    return "No pollution data available."

                lines = [f"{i+1}. {loc} → {val:.2f}" for i, (loc, val) in enumerate(top5.items())]
                return "🏆 Top 5 polluted locations:\n\n" + "\n".join(lines)

            # ================= SEASON MOST POLLUTED =================
            elif "season" in q and "polluted" in q:
                grp = chatbot_df.groupby("season", dropna=True)["pm2_5"].mean().dropna()
                if grp.empty:
                    return "No season data available."
                season = grp.idxmax()
                val = grp.max()
                return f"🌦️ Most polluted season is **{season}** (Avg PM2.5 = {val:.2f})"

            # ================= SEASON SUMMARY =================
            elif "season" in q:
                season_data = chatbot_df.groupby("season", dropna=True)["pm2_5"].mean().dropna()
                if season_data.empty:
                    return "No season data available."

                lines = [f"- {season}: {val:.2f}" for season, val in season_data.items()]
                return "📊 Season-wise average PM2.5:\n\n" + "\n".join(lines)

            # ================= DATE / TIME / SOURCE FIXED =================
            elif "date" in q or "time" in q or re.search(r"\d{4}-\d{2}-\d{2}", q):
                date_val = extract_date(q)
                hour_val = extract_hour(q)
                loc_val = detect_location(q, chatbot_df)

                if date_val is None:
                    return "⚠️ Please provide date in **YYYY-MM-DD** format."

                filtered = chatbot_df[chatbot_df["date_only"] == date_val]

                if loc_val:
                    filtered = filtered[filtered["location"] == loc_val]

                if filtered.empty:
                    if loc_val:
                        return f"No data available for **{loc_val}** on **{date_val}**."
                    return f"No data available for **{date_val}**."

                if hour_val is not None:
                    filtered = filtered[filtered["hour_only"] == hour_val]

                    if filtered.empty:
                        if loc_val:
                            return f"No data available for **{loc_val}** at **{hour_val:02d}:00** on **{date_val}**."
                        return f"No data available at **{hour_val:02d}:00** on **{date_val}**."

                    # If exact single row exists
                    if len(filtered) == 1:
                        row = filtered.iloc[0]
                        return (
                            f"📅 **{date_val} at {hour_val:02d}:00**\n\n"
                            f"📍 Location: **{row['location']}**\n"
                            f"🏭 Pollution source: **{row['pollution_source']}**\n"
                            f"🌫️ PM2.5: **{row['pm2_5']:.2f}**"
                        )

                    # Multiple rows at same date/hour
                    src_counts = filtered["pollution_source"].value_counts().to_dict()
                    avg_pm = filtered["pm2_5"].mean()

                    if loc_val and filtered["location"].nunique() == 1:
                        return (
                            f"📅 **{date_val} at {hour_val:02d}:00** for **{loc_val}**\n\n"
                            f"🏭 Source distribution: **{src_counts}**\n"
                            f"🌫️ Avg PM2.5: **{avg_pm:.2f}**\n"
                            f"📌 Matching records: **{len(filtered)}**"
                        )
                    else:
                        grouped = filtered[["location", "pollution_source", "pm2_5"]].copy()
                        grouped = grouped.sort_values(["location", "pm2_5"], ascending=[True, False])

                        lines = []
                        for _, row in grouped.iterrows():
                            lines.append(
                                f"- {row['location']} → Source: {row['pollution_source']}, PM2.5: {row['pm2_5']:.2f}"
                            )

                        return (
                            f"📅 **{date_val} at {hour_val:02d}:00**\n\n"
                            + "\n".join(lines[:10])
                        )

                # Date only, no hour
                src_counts = filtered["pollution_source"].value_counts().to_dict()
                avg_pm = filtered["pm2_5"].mean()

                if loc_val:
                    return (
                        f"📅 On **{date_val}** for **{loc_val}**\n\n"
                        f"🏭 Source distribution: **{src_counts}**\n"
                        f"🌫️ Avg PM2.5: **{avg_pm:.2f}**\n"
                        f"📌 Records found: **{len(filtered)}**"
                    )

                top_sources = filtered["pollution_source"].value_counts().head(5).to_dict()
                return (
                    f"📅 On **{date_val}**\n\n"
                    f"🏭 Top pollution sources: **{top_sources}**\n"
                    f"🌫️ Avg PM2.5: **{avg_pm:.2f}**\n"
                    f"📌 Records found: **{len(filtered)}**"
                )

            # ================= LOCATION =================
            elif "pollution in" in q or "source in" in q or "pm2.5 in" in q or "aqi in" in q:
                loc = detect_location(q, chatbot_df)
                if loc:
                    loc_df = chatbot_df[chatbot_df["location"] == loc]
                    if loc_df.empty:
                        return f"No data available for **{loc}**."

                    avg = loc_df["pm2_5"].mean()
                    src = loc_df["pollution_source"].value_counts().idxmax()
                    return f"📍 In **{loc}**: Avg PM2.5 = **{avg:.2f}**, Main source = **{src}**"

                return "⚠️ Please mention a valid location name."

            # ================= SOURCE =================
            elif "source" in q:
                src_counts = chatbot_df["pollution_source"].value_counts()
                if src_counts.empty:
                    return "No pollution source data available."
                src = src_counts.idxmax()
                return f"🏭 Most common pollution source is **{src}**"

            # ================= TREND =================
            elif "trend" in q:
                if "hour_only" not in chatbot_df.columns:
                    return "Hour data not available for trend analysis."

                trend = chatbot_df.groupby("hour_only")["pm2_5"].mean().dropna()
                if trend.empty:
                    return "Trend data not available."

                peak = int(trend.idxmax())
                peak_val = trend.max()
                return f"📈 Pollution peaks around **{peak:02d}:00** (Avg PM2.5 = **{peak_val:.2f}**)"

            # ================= AQI =================
            elif "aqi" in q:
                loc = detect_location(q, chatbot_df)

                if loc:
                    loc_df = chatbot_df[chatbot_df["location"] == loc]
                    if loc_df.empty:
                        return f"No AQI data available for **{loc}**."
                    avg = loc_df["pm2_5"].mean()
                    return f"📍 **{loc}** → Avg PM2.5 = **{avg:.2f}** → {classify_aqi(avg)}"

                avg = chatbot_df["pm2_5"].mean()
                return f"🌍 Overall Avg PM2.5 = **{avg:.2f}** → {classify_aqi(avg)}"

            # ================= DEFAULT =================
            else:
                return """🤖 You can ask things like:

- Most polluted location
- Safest area
- Top polluted places
- Which season is most polluted
- Pollution on 2026-02-12
- Pollution on 2026-02-12 at 20
- Pollution source on 2026-02-12 at 20 in Zoo Park, Hyderabad - TSPCB
- Pollution in Zoo Park, Hyderabad - TSPCB
- AQI status
- Trend analysis"""

        except Exception as e:
            return f"⚠️ Error: {e}"

    # ================= USER INPUT =================
    user_input = st.chat_input("Ask anything about pollution...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        response = chatbot_response(user_input)

        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)