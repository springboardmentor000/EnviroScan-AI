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
    from openai import OpenAI

    st.subheader("🤖 EnviroScan Hybrid AI Assistant")

    # ================= GROQ SETUP =================
    client = OpenAI(
        api_key="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",   # replace with your real Groq API key
        base_url="https://api.groq.com/openai/v1"
    )

    # ================= DATA PREP =================
    # Make sure df already exists in your app before tab4.
    # Example:
    # df = pd.read_csv("vij_hyd_labelled_dataset.csv")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["date"] = df["timestamp"].dt.date

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ================= HELPERS =================
    def find_location_from_query(query, df):
        query_l = query.lower()

        for loc in df["location"].dropna().unique():
            if loc.lower() in query_l:
                return loc

        simplified_map = {
            "central university": "Central University, Hyderabad - TSPCB",
            "zoo park": "Zoo Park, Hyderabad - TSPCB",
            "somajiguda": "Somajiguda, Hyderabad - TSPCB",
            "sanathnagar": "Sanathnagar, Hyderabad - TSPCB",
            "kompally": "Kompally Municipal Office, Hyderabad - TSPCB",
            "kanuru": "Kanuru, Vijayawada - APPCB",
            "hb colony": "HB Colony, Vijayawada - APPCB"
        }

        for key, full_loc in simplified_map.items():
            if key in query_l:
                return full_loc

        return None

    def aqi_label_from_pm25(pm):
        # Your dataset appears normalized around 0 to 1, not raw AQI values.
        # So this is a dataset-relative interpretation, not official AQI conversion.
        if pm <= 0.12:
            return "🟢 Good"
        elif pm <= 0.20:
            return "🟡 Moderate"
        elif pm <= 0.30:
            return "🟠 Unhealthy for sensitive groups"
        elif pm <= 0.45:
            return "🔴 Poor"
        else:
            return "⚫ Hazardous"

    # ================= RULE-BASED FUNCTION =================
    def rule_based_answer(query):
        query_l = query.lower().strip()

        # -------- DATE --------
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", query_l)
        hour_match = re.search(r"(?:\b|at\s)([0-9]|1[0-9]|2[0-3])(?::00)?\b", query_l)

        location_match = find_location_from_query(query_l, df)

        # -------- DATE + TIME + OPTIONAL LOCATION --------
        if date_match:
            date_val = pd.to_datetime(date_match.group()).date()
            filtered = df[df["date"] == date_val]

            if filtered.empty:
                return f"⚠️ No data for {date_val}."

            if location_match:
                filtered = filtered[filtered["location"] == location_match]
                if filtered.empty:
                    return f"⚠️ No data for {location_match} on {date_val}."

            if hour_match:
                hour_val = int(hour_match.group(1))
                filtered = filtered[filtered["hour"] == hour_val]

                if filtered.empty:
                    if location_match:
                        return f"⚠️ No data for {location_match} on {date_val} at {hour_val:02d}:00."
                    return f"⚠️ No data for {date_val} at {hour_val:02d}:00."

                if location_match:
                    row = filtered.iloc[0]
                    return (
                        f"📅 {location_match} on {date_val} at {hour_val:02d}:00\n\n"
                        f"- PM2.5: {row['pm2_5']:.3f}\n"
                        f"- Pollution source: {row['pollution_source']}\n"
                        f"- AQI status: {aqi_label_from_pm25(row['pm2_5'])}"
                    )

                avg_pm = filtered["pm2_5"].mean()
                top_source = filtered["pollution_source"].mode().iloc[0]
                return (
                    f"📅 {date_val} at {hour_val:02d}:00\n\n"
                    f"- Matching records: {len(filtered)}\n"
                    f"- Avg PM2.5: {avg_pm:.3f}\n"
                    f"- Main pollution source: {top_source}\n"
                    f"- AQI status: {aqi_label_from_pm25(avg_pm)}"
                )

            # date only
            if location_match:
                avg_pm = filtered["pm2_5"].mean()
                top_source = filtered["pollution_source"].mode().iloc[0]
                return (
                    f"📅 {location_match} on {date_val}\n\n"
                    f"- Avg PM2.5: {avg_pm:.3f}\n"
                    f"- Main pollution source: {top_source}\n"
                    f"- AQI status: {aqi_label_from_pm25(avg_pm)}"
                )

            avg_pm = filtered["pm2_5"].mean()
            top_source = filtered["pollution_source"].mode().iloc[0]
            return (
                f"📅 {date_val}\n\n"
                f"- Avg PM2.5: {avg_pm:.3f}\n"
                f"- Main pollution source: {top_source}\n"
                f"- AQI status: {aqi_label_from_pm25(avg_pm)}"
            )

        # -------- TOP POLLUTED --------
        if (
            "top polluted" in query_l
            or "top polluted areas" in query_l
            or "most polluted areas" in query_l
            or "most poluted areas" in query_l
            or "top poluted areas" in query_l
            or ("top" in query_l and "polluted" in query_l)
        ):
            top5 = (
                df.groupby("location")["pm2_5"]
                .mean()
                .sort_values(ascending=False)
                .head(5)
            )
            lines = ["🏆 Top 5 polluted locations:\n"]
            for i, (loc, val) in enumerate(top5.items(), start=1):
                lines.append(f"{i}. {loc} — PM2.5: {val:.3f}")
            return "\n".join(lines)

        # -------- MOST POLLUTED --------
        if "most polluted" in query_l or "worst location" in query_l:
            loc_avg = df.groupby("location")["pm2_5"].mean()
            loc = loc_avg.idxmax()
            val = loc_avg.max()
            return f"🚨 Most polluted location: {loc} (Avg PM2.5: {val:.3f})"

        # -------- SAFEST --------
        if "safe" in query_l or "safest" in query_l or "least polluted" in query_l or "cleanest" in query_l:
            loc_avg = df.groupby("location")["pm2_5"].mean()
            loc = loc_avg.idxmin()
            val = loc_avg.min()
            return f"✅ Safest location: {loc} (Avg PM2.5: {val:.3f})"

        # -------- LOCATION --------
        if location_match and ("pollution" in query_l or "pm2.5" in query_l or "aqi" in query_l):
            loc_df = df[df["location"] == location_match]

            if loc_df.empty:
                return "⚠️ Location not found."

            avg_pm = loc_df["pm2_5"].mean()
            top_source = loc_df["pollution_source"].mode().iloc[0]
            return (
                f"📍 {location_match}\n\n"
                f"- Avg PM2.5: {avg_pm:.3f}\n"
                f"- Main pollution source: {top_source}\n"
                f"- AQI status: {aqi_label_from_pm25(avg_pm)}"
            )

        # -------- AQI --------
        if "aqi" in query_l:
            if location_match:
                loc_df = df[df["location"] == location_match]
                avg_pm = loc_df["pm2_5"].mean()
                return f"🌫️ AQI status for {location_match}: {aqi_label_from_pm25(avg_pm)} (PM2.5: {avg_pm:.3f})"

            avg_pm = df["pm2_5"].mean()
            return f"🌫️ Overall AQI status: {aqi_label_from_pm25(avg_pm)} (Avg PM2.5: {avg_pm:.3f})"

        # -------- PEAK HOUR --------
        if "peak" in query_l or "trend" in query_l or "pollution hour" in query_l:
            trend = df.groupby("hour")["pm2_5"].mean()
            peak = trend.idxmax()
            val = trend.max()
            return f"⏰ Pollution peaks around hour {peak}:00 (Avg PM2.5: {val:.3f})"

        # -------- SEASON --------
        if "season" in query_l:
            season_avg = df.groupby("season")["pm2_5"].mean()
            season_names = {0: "Wet", 1: "Dry"}

            if "most polluted" in query_l or "polluted season" in query_l:
                s = season_avg.idxmax()
                return f"🌦️ Most polluted season: {season_names.get(s, s)} (Avg PM2.5: {season_avg[s]:.3f})"

            lines = ["📊 Season-wise pollution:"]
            for s, val in season_avg.items():
                lines.append(f"- {season_names.get(s, s)}: {val:.3f}")
            return "\n".join(lines)

        # -------- HEALTH --------
        if "health" in query_l or "tips" in query_l or "precautions" in query_l:
            return """❤️ Health tips:
- Wear an N95 mask outdoors.
- Avoid heavy traffic hours if pollution is high.
- Keep windows closed during peak pollution.
- Drink enough water and reduce outdoor exercise in polluted hours."""

        return None

    # ================= LLM FUNCTION =================
    def ask_groq(query):
        context = """
        The dataset contains:
        - location-wise pollution data
        - PM2.5 values
        - hourly trends
        - date-wise values
        - pollution source labels
        - season-wise pollution patterns

        Important:
        - Use short, clear answers.
        - If the question is general, answer simply.
        - Do not invent exact dataset numbers unless provided.
        """

        prompt = f"""
        You are EnviroScan AI, a pollution assistant.

        Dataset context:
        {context}

        User question:
        {query}

        Rules:
        - Keep answer short and clear
        - Give only one direct answer
        - For general questions, answer naturally
        - Do not mention that you are an LLM
        """

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a helpful pollution assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_completion_tokens=250
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"⚠️ Groq API error: {str(e)}"

    # ================= MAIN CHATBOT =================
    def chatbot_response(query):
        rule_answer = rule_based_answer(query)

        if rule_answer:
            return rule_answer

        return ask_groq(query)

    # ================= EXAMPLE HELP =================
    st.caption("🤖 Ask: pollution in Central University • AQI Zoo Park 2026-02-12 20:00 • safest location • most polluted season • top polluted areas")

    # ================= USER INPUT =================
    user_input = st.chat_input("Ask anything about pollution...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        bot_reply = chatbot_response(user_input)

        st.session_state.messages.append({"role": "assistant", "content": bot_reply})

        with st.chat_message("assistant"):
            st.markdown(bot_reply)
