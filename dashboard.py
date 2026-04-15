import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import smtplib
import folium
from email.mime.text import MIMEText
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import re
from openai import OpenAI
import requests
from streamlit_autorefresh import st_autorefresh
import docx2txt
import base64
from PyPDF2 import PdfReader
from rules import handle_rules
from groq import Groq
import httpx


# ================= PAGE CONFIG =================
st.set_page_config(page_title="EnviroScan Dashboard", layout="wide")

# ==========================================
# GLOBAL ENVIROSCAN DARK POLLUTION THEME
# ==========================================
st.markdown("""
<style>
/* ===== PAGE BACKGROUND ===== */
.stApp {
    background: linear-gradient(rgba(15, 23, 42, 0.35), rgba(15, 23, 42, 0.35)),
            url("https://pplx-res.cloudinary.com/image/upload/pplx_search_images/4076553fbccce11ef6a75bfd766ba3cb7c2ce91d.jpg");                
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* ===== REMOVE ALL DEFAULT CONTAINERS ===== */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: transparent !important;
}

.block-container {
    max-width: 100% !important;
    padding-top: 2rem !important;
    padding-left: 4rem !important;
    padding-right: 4rem !important;
    padding-bottom: 2rem !important;
    background: transparent !important;
}

/* remove all streamlit wrapper backgrounds */
div[data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"],
div[data-testid="column"],
div[data-testid="stTabs"],
div[data-testid="stMarkdownContainer"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ===== HEADER ===== */
[data-testid="stHeader"] {
    background: transparent !important;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: rgba(30, 41, 59, 0.78) !important;
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* ===== TEXT ===== */
h1, h2, h3, h4, h5, h6, p, label, div, span {
    color: white !important;
}

/* ===== TITLE ===== */
.main-title {
    text-align: center;
    font-size: 4rem;
    font-weight: 900;
    color: white !important;
    text-shadow: 0 4px 18px rgba(0,0,0,0.55);
    margin-bottom: 0.3rem;
}

.sub-title {
    text-align: center;
    font-size: 1.3rem;
    color: #cbd5e1 !important;
    margin-bottom: 2rem;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.15) !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #f1f5f9 !important;
    font-weight: 600 !important;
}

.stTabs [aria-selected="true"] {
    color: #60a5fa !important;
    border-bottom: 3px solid #60a5fa !important;
}

.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding-top: 1rem !important;
}

/* ===== SELECTBOX ===== */
.stSelectbox div[data-baseweb="select"] > div {
    background: rgba(30, 41, 59, 0.75) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 14px !important;
    color: white !important;
}
.stSelectbox svg {
    fill: white !important;
}

/* ===== METRIC CARDS ONLY ===== */
.metric-box {
    background: rgba(15, 23, 42, 0.82);
    border-radius: 18px;
    padding: 1.5rem 1rem;
    text-align: center;
    border-top: 4px solid #ef4444;
    box-shadow: 0 10px 28px rgba(0,0,0,0.35);
}

/* ===== AQI BAR / ALERT BOXES ===== */
.aqi-bar {
    background: linear-gradient(90deg, #facc15, #f59e0b);
    color: white !important;
    padding: 1.2rem;
    border-radius: 18px;
    font-weight: 700;
    text-align: center;
    box-shadow: 0 8px 20px rgba(0,0,0,0.25);
}

/* ===== REMOVE EXPANDER / FORM BOX BACKGROUND ===== */
[data-testid="stExpander"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ===== OPTIONAL: hide top gap ===== */
section.main > div {
    padding-top: 0rem !important;
}

/* ===== 12. MAIN ACTION BUTTONS (REFRESH & DOWNLOAD) ===== */ 
.stButton > button, .stDownloadButton > button { 
    background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important; 
    color: #ffffff !important; 
    border: none !important; 
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 0.6rem 1.2rem !important;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover, .stDownloadButton > button:hover { 
    background: linear-gradient(135deg, #2563eb, #1e40af) !important; 
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6) !important;
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# MAIN DASHBOARD HEADER
# ==========================================
st.markdown('<div class="main-title">EnviroScan Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Air Pollution Monitoring & Prediction System</div>', unsafe_allow_html=True)



# ================= LOAD DATA =================
df = pd.read_csv("vij_hyd_labelled_dataset.csv")

model = joblib.load("models/gradient_boost_pollution_model.pkl")
le = joblib.load("models/label_encoder.pkl")

FEATURE_COLS = [
    "pm2_5","pm10","no2","o3","so2","co",
    "road_count","industrial_count","waste_count","farmland_count"
]

# ================= SIDEBAR =================
st.sidebar.markdown('<div class="sidebar-title">  Filters</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
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

# ================= TABS =================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Dashboard", 
    " Map", 
    " Reports",
    " Chatbot",
    " Live pollution detection"
])


import time

def animated_kpi(label, value, suffix=""):
    placeholder = st.empty()

    for i in range(0, 101, 5):
        display_val = (value * i) / 100
        placeholder.markdown(f"""
        <div class="metric-box">
            <div style="font-size:32px; font-weight:800;">
                {display_val:.2f}{suffix}
            </div>
            <div style="font-size:14px; color:#475569;">
                {label}
            </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.01)

    placeholder.markdown(f"""
    <div class="metric-box">
        <div style="font-size:32px; font-weight:800;">
            {value:.2f}{suffix}
        </div>
        <div style="font-size:14px; color:#475569;">
            {label}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================= GLOBAL BACKGROUND FUNCTION =================

with tab1:

    st.subheader(" Pollution Indicators")

    # ✅ DEFINE COLUMNS HERE (IMPORTANT FIX)
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        animated_kpi("PM2.5", filtered_df["pm2_5"].mean())

    with col2:
        animated_kpi("PM10", filtered_df["pm10"].mean())

    with col3:
        animated_kpi("NO2", filtered_df["no2"].mean())

    with col4:
        animated_kpi("O3", filtered_df["o3"].mean())

    with col5:
        animated_kpi("SO2", filtered_df["so2"].mean())

    with col6:
        animated_kpi("CO", filtered_df["co"].mean())

    st.markdown('</div>', unsafe_allow_html=True)

    # AQI
    st.subheader(" Air Quality Index (AQI)")

    def get_aqi_status(pm25):
        if pm25 <= 0.2:
            return "Good ", "Air quality is good. Safe for outdoor activities."
        elif pm25 <= 0.4:
            return "Moderate ", "Moderate air quality. Sensitive people should reduce outdoor exposure."
        elif pm25 <= 0.7:
            return "Unhealthy ", "Wear a mask and avoid prolonged outdoor exposure."
        else:
            return "Hazardous ", "Stay indoors. Avoid outdoor activity completely."

    avg_pm25 = filtered_df["pm2_5"].mean()
    status, advice = get_aqi_status(avg_pm25)
    st.markdown(f"""
    <div style="background: linear-gradient(135deg,#facc15,#f59e0b);
    padding:15px;border-radius:15px;color:white;text-align:center;">
    AQI Status: {status}
    </div>
    """, unsafe_allow_html=True)
    st.write(f"Average PM2.5: {avg_pm25:.2f}")
    st.subheader(" Health Recommendations")
    st.markdown(f"""
   <div style="background:#10b981;color:white;padding:15px;border-radius:15px;">
   {advice}
   </div>
""", unsafe_allow_html=True)
    
# ================= SAFE SLIDER FUNCTION =================
    def safe_slider(label, series, is_int=False):
        min_val = series.min()
        max_val = series.max()
        mean_val = series.mean()

        # 🔥 FIX: if min == max → create range
        if min_val == max_val:
            min_val = min_val - 5
            max_val = max_val + 5

        if is_int:
            return st.slider(label, int(min_val), int(max_val), int(mean_val))
        else:
            return st.slider(label, float(min_val), float(max_val), float(mean_val))

    # ================= SLIDERS =================
    st.subheader(" Adjust Pollution Parameters")

    col1, col2 = st.columns(2)

    with col1:
        pm2_5 = safe_slider("PM2.5", filtered_df["pm2_5"])
        pm10 = safe_slider("PM10", filtered_df["pm10"])
        no2 = safe_slider("NO2", filtered_df["no2"])
        o3 = safe_slider("O3", filtered_df["o3"])
        so2 = safe_slider("SO2", filtered_df["so2"])
        co = safe_slider("CO", filtered_df["co"])

    with col2:
        road = safe_slider("Road Count", filtered_df["road_count"], True)
        industrial = safe_slider("Industrial Count", filtered_df["industrial_count"], True)
        waste = safe_slider("Waste Count", filtered_df["waste_count"], True)
        farm = safe_slider("Farmland Count", filtered_df["farmland_count"], True)

    # ================= PREDICTION =================
    st.subheader(" Predict Pollution Source")

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


    # TABLE
    st.subheader(" Detailed Report")
    table_df = filtered_df.sort_values(by=pollutant, ascending=False)
    st.dataframe(table_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.subheader(" Download Report")

    csv = table_df.to_csv(index=False)

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{location}_pollution_report.csv",
        mime="text/csv"
    )
    

# ---------- EMAIL----------
    
    POLLUTANT_LIMITS = {
    "pm2_5": 0.6,
    "pm10": 0.5,
    "no2": 0.4,
    "co": 0.3,
    "so2": 0.3,
    "o3": 0.5
}


st.subheader("🚨 Automatic Multi-Pollutant Alert System")

# ✅ Prevent spam emails
if "email_sent" not in st.session_state:
    st.session_state.email_sent = False

try:
    exceeded_pollutants = []

    for pollutant, limit in POLLUTANT_LIMITS.items():
        if pollutant in filtered_df.columns:
            avg_value = filtered_df[pollutant].mean()

            if pd.notna(avg_value) and avg_value > limit:
                exceeded_pollutants.append(
                    f"{pollutant.upper()} → {avg_value:.2f} (Limit: {limit})"
                )

    # 🚨 Trigger alert if ANY pollutant exceeds
    if exceeded_pollutants and not st.session_state.email_sent:

        alert_details = "\n".join(exceeded_pollutants)

        email_body = f"""
🚨 Pollution Alert for {location}

The following pollutants have exceeded safe limits:

{alert_details}

⚠️ Health Advice:
Avoid outdoor activities, wear masks, and monitor AQI regularly.
"""

        msg = MIMEText(email_body)
        msg["Subject"] = "🚨 Multi-Pollutant Alert"
        msg["From"] = "chandusst987@gmail.com"
        msg["To"] = "chandusst987@gmail.com"

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()

        # 🔴 Use App Password
        server.login("chandusst987@gmail.com", "jiof corb rrbn gytq")

        server.send_message(msg)
        server.quit()

        st.session_state.email_sent = True

        st.error("🚨 Alert! Multiple pollutants exceeded safe limits. Email sent!")

    elif not exceeded_pollutants:
        st.success("✅ All pollutant levels are within safe limits.")
        st.session_state.email_sent = False

except Exception as e:
    st.error(f"Error: {e}")

# ================= MAP =================
with tab2:
    

    st.subheader(" Map")

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
    <span style="color:red;">● Industrial<br>
    <span style="color:blue;">●Vehicular<br>
    <span style="color:green;">● Agricultural<br>
    <span style="color:orange;">● Burning<br>
    <span style="color:purple;">● Natural<br>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # ================= DISPLAY =================
    st_folium(m, width=1000, height=900)

# ================= REPORT =================
with tab3:
    

    st.subheader(" Reports & Analysis")

    all_sources = ["Industrial","Vehicular","Agricultural","Burning","Natural"]

    source_counts = filtered_df["pollution_source"] \
        .value_counts() \
        .reindex(all_sources, fill_value=0)

    # TABLE
    st.write("###  Source Table")
    table_df = source_counts.reset_index()
    table_df.columns = ["Source", "Count"]
    st.dataframe(table_df)

    # PIE
    # PIE (FIXED CLEAN VERSION)
    st.write("###  Pie Chart")

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
    st.write("###  Trend")
    trend = filtered_df.groupby("hour")[pollutant].mean()
    st.line_chart(trend)

    # ================= TOP LOCATIONS =================
    st.subheader("Top Polluted Locations")

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

# ================= CHATBOT ===============#

with tab4:

    # --- 1. FILE UPLOADER & PROCESSING ---
    uploaded_files = st.file_uploader("Upload additional documents (PDF/CSV/DOCX)", type=["csv", "pdf", "docx"], accept_multiple_files=True)
    
    uploaded_context = ""
    if uploaded_files:
        for file in uploaded_files:
            try:
                if file.name.endswith('.csv'):
                    df_up = pd.read_csv(file)
                    uploaded_context += f"--- Data from {file.name} ---\n{df_up.head(50).to_csv(index=False)}\n\n"
                elif file.name.endswith('.pdf'):
                    import PyPDF2
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    uploaded_context += f"--- Text from {file.name} ---\n{text[:4000]}\n\n" 
                elif file.name.endswith('.docx'):
                    import docx
                    doc = docx.Document(file)
                    text = "\n".join([para.text for para in doc.paragraphs])
                    uploaded_context += f"--- Text from {file.name} ---\n{text[:4000]}\n\n"
            except Exception as e:
                st.warning(f"Could not read {file.name}. Ensure PyPDF2 and python-docx are installed. Error: {e}")

    # --- 2. CAPTURE INPUT FIRST ---
    prompt = st.chat_input("Ask about the dataset, uploaded files, or general knowledge...")

    if prompt:
        st.session_state.messages = [
            {"role": "assistant", "content": " Type **help** to see what I can do!"}
        ]
    elif "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": " Type **help** to see what I can do!"}
        ]

    # --- 3. RENDER THE CLEARED CHAT ---
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): 
            st.markdown(m["content"])

    # --- 4. COMMAND & ANALYSIS LOGIC ---
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if prompt.lower().strip() == "help":
            help_msg = """
            <div class="glass-card" style="text-align: left; padding: 20px;">
                <h3 style="color: #3b82f6; margin-top: 0; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px;">
                    📖 Master Assistant Guide
                </h3>
                <b style="color: #f8fafc;">✅ I can answer 3 types of questions:</b>
                <ul style="color: #cbd5e1; line-height: 1.8;">
                    <li><b>1. Base Dataset:</b> "Compare temperature and PM10."</li>
                    <li><b>2. Uploaded Files:</b> "Summarize the PDF I just uploaded"</li>
                    <li><b>3. General Knowledge:</b> "What is AQI?", "How does rain affect pollution?"</li>
                </ul>
            </div>
            """
            with st.chat_message("assistant"): st.markdown(help_msg, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": "Displayed help guide."})

        else:
            global_knowledge = ""
            context_data = ""
            
            # --- BASE DATASET FETCH ---
            try:
                chat_df = pd.read_csv("vij_hyd_labelled_dataset.csv")
            except Exception as e:
                chat_df = None

            if chat_df is not None:
                temp_df = chat_df.copy()
                temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp']) 
                temp_df['month'] = temp_df['timestamp'].dt.month_name()
                temp_df['hour'] = temp_df['timestamp'].dt.hour
                
                # THE FIX: Added weather columns to the allowed Pandas list!
                num_cols = ['co', 'pm2_5', 'pm10', 'no2', 'so2', 'o3', 'temperature_c', 'humidity', 'pressure_hpa', 'wind_speed_ms', 'wind_direction']
                available_cols = [c for c in num_cols if c in temp_df.columns]
                
                monthly_trends = temp_df.groupby('month')[available_cols].mean().round(2).to_csv()
                location_stats = temp_df.groupby('location')[available_cols].mean().round(2)
                
                if 'pm2_5' in available_cols:
                    most_polluted_loc = location_stats['pm2_5'].idxmax()
                    safest_loc = location_stats['pm2_5'].idxmin()
                    hourly_stats = temp_df.groupby('hour')['pm2_5'].mean()
                    worst_hour = f"{hourly_stats.idxmax()}:00"
                    safest_hour = f"{hourly_stats.idxmin()}:00"
                else:
                    most_polluted_loc, safest_loc, worst_hour, safest_hour = "N/A", "N/A", "N/A", "N/A"

                overall_source = temp_df['pollution_source'].mode()[0] if 'pollution_source' in temp_df.columns else "Unknown"
                
                global_knowledge = (
                    f"DATASET TIMEFRAME: Feb 1, 2026 to March 31, 2026.\n\n"
                    f"--- BASE DATASET EXTREMES ---\n"
                    f"- Most Polluted Location (Highest PM2.5): {most_polluted_loc}\n"
                    f"- Safest Location (Lowest PM2.5): {safest_loc}\n"
                    f"- Most Polluted Hour of Day: {worst_hour}\n"
                    f"- Safest Hour of Day: {safest_hour}\n"
                    f"- Most Common Pollution Source Overall: {overall_source}\n\n"
                    f"--- TRENDS BY MONTH (Base Dataset) ---\n{monthly_trends}\n"
                    f"--- AVERAGES BY LOCATION (Base Dataset) ---\n{location_stats.to_csv()}\n"
                )

                date_match = re.search(r'\d{4}-\d{2}-\d{2}', prompt)
                if date_match:
                    d = date_match.group(0)
                    day_data = temp_df[temp_df['timestamp'].dt.date == pd.to_datetime(d).date()]
                    
                    location_found = False
                    for loc in temp_df['location'].unique():
                        loc_short = str(loc).split(",")[0].lower().strip()
                        if loc_short in prompt.lower():
                            day_data = day_data[day_data['location'].str.lower().str.contains(loc_short, na=False)]
                            location_found = True
                            break
                    
                    if not day_data.empty:
                        cols_to_keep = ['timestamp', 'location'] + available_cols + ['pollution_source']
                        cols_to_keep = [c for c in cols_to_keep if c in day_data.columns]
                        
                        if location_found:
                            context_data = "HOURLY DATA FOR REQUESTED LOCATION:\n" + day_data[cols_to_keep].head(24).to_csv(index=False)
                        else:
                            agg_dict = {c: 'mean' for c in available_cols}
                            if 'pollution_source' in day_data.columns:
                                agg_dict['pollution_source'] = lambda x: x.value_counts().index[0] if not x.value_counts().empty else 'Unknown'
                            
                            summary_df = day_data.groupby('location').agg(agg_dict).round(2).reset_index()
                            context_data = f"DAILY AVERAGE PER LOCATION FOR {d}:\n" + summary_df.to_csv(index=False)

            try:
                client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                
                system_instr = (
                    "You are the EnviroScan Senior AI Environmental Consultant.\n\n"
                    "HOW TO ANSWER (Choose the rule that fits the user's prompt):\n"
                    "RULE 1 (BASE DATASET): If the user asks about trends, locations, extremes, comparisons (like weather vs pollutants), or dates in the dataset, use ONLY the 'GLOBAL KNOWLEDGE BASE' and 'SPECIFIC DATA TABLE' below.\n"
                    "RULE 2 (UPLOADED FILES): If the user asks about an uploaded document, PDF, or CSV, use ONLY the 'UPLOADED FILE CONTEXT' below to answer.\n"
                    "RULE 3 (GENERAL KNOWLEDGE): If the user asks a general scientific question, use your general training knowledge.\n\n"
                    "FORMATTING (For Data Queries Only):\n"
                    "If answering Rule 1 or Rule 2, format your response with: 'The Direct Answer', 'Public Health Implications', and 'Actionable Recommendations'.\n"
                    "If answering Rule 3 (General Knowledge), just write a normal, helpful explanation.\n\n"
                    f"--- GLOBAL KNOWLEDGE BASE (Base Dataset) ---\n{global_knowledge}\n\n"
                    f"--- SPECIFIC DATA TABLE (Base Dataset Dates) ---\n{context_data if context_data else 'No specific date queried.'}\n\n"
                    f"--- UPLOADED FILE CONTEXT ---\n{uploaded_context if uploaded_context else 'No files uploaded.'}"
                )

                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_instr},
                        {"role": "user", "content": prompt}
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3, 
                    max_tokens=1500
                ).choices[0].message.content
                
                with st.chat_message("assistant"): st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error communicating with AI: {e}")


with tab5:
    import streamlit as st
    import requests
    import pandas as pd

    st.markdown(" Live Pollution Monitoring")

    # ================= API KEY =================
    API_KEY = "c630fe86123ccf82de5644158b421b80"  # 🔴 replace this

    if not API_KEY or API_KEY == "YOUR_OPENWEATHER_API_KEY":
        st.warning("⚠️ Please add your OpenWeather API key to enable live data")
        st.stop()

    # ================= CITY DATA =================
    city_coords = {
        "Hyderabad":     {"lat": 17.3850, "lon": 78.4867},
        "Vijayawada":    {"lat": 16.5062, "lon": 80.6480},
        "Chennai":       {"lat": 13.0827, "lon": 80.2707},
        "Delhi":         {"lat": 28.6139, "lon": 77.2090},
        "Mumbai":        {"lat": 19.0760, "lon": 72.8777},
        "Bengaluru":     {"lat": 12.9716, "lon": 77.5946},
        "Visakhapatnam": {"lat": 17.6801, "lon": 83.2016}
    }

    city_list = list(city_coords.keys())

    # ================= SESSION =================
    if "live_city" not in st.session_state:
        st.session_state.live_city = "Hyderabad"

    if "live_city_widget" not in st.session_state:
        st.session_state.live_city_widget = st.session_state.live_city

    def update_city():
        st.session_state.live_city = st.session_state.live_city_widget

    # ================= API CALL =================
    @st.cache_data(ttl=300)
    def get_live_pollution(city, key):
        coords = city_coords[city]

        url = "http://api.openweathermap.org/data/2.5/air_pollution"
        params = {"lat": coords["lat"], "lon": coords["lon"], "appid": key}

        res = requests.get(url, params=params, timeout=20)
        res.raise_for_status()
        data = res.json()

        item = data["list"][0]

        return {
            "AQI": item["main"]["aqi"],
            "PM2.5": item["components"].get("pm2_5"),
            "PM10": item["components"].get("pm10"),
            "NO2": item["components"].get("no2"),
            "O3": item["components"].get("o3"),
            "SO2": item["components"].get("so2"),
            "CO": item["components"].get("co"),
            "time": pd.Timestamp.now()
        }

    # ================= FORECAST =================
    @st.cache_data(ttl=300)
    def get_forecast(city, key):
        coords = city_coords[city]

        url = "http://api.openweathermap.org/data/2.5/air_pollution/forecast"
        params = {"lat": coords["lat"], "lon": coords["lon"], "appid": key}

        res = requests.get(url, params=params, timeout=20)
        res.raise_for_status()
        data = res.json()

        rows = []
        for i in data["list"]:
            rows.append({
                "Time": pd.to_datetime(i["dt"], unit="s"),
                "AQI": i["main"]["aqi"],
                "PM2.5": i["components"].get("pm2_5"),
                "PM10": i["components"].get("pm10"),
                "So2": i["components"].get("so2"),
                "CO": i["components"].get("co"),
                "O3": i["components"].get("o3"),
                "NO2": i["components"].get("no2")
            })

        return pd.DataFrame(rows)

    # ================= AQI STATUS =================
    def get_aqi_status(aqi):
        return {
            1: "Good 🟢",
            2: "Fair 🟡",
            3: "Moderate 🟠",
            4: "Poor 🔴",
            5: "Very Poor 🛑"
        }.get(aqi, "Unknown")

    # ================= HEALTH =================
    def health_advice(aqi):
        if aqi == 1:
            return "Air quality is good. Enjoy outdoor activities."
        elif aqi == 2:
            return "Acceptable air quality. Sensitive individuals should be cautious."
        elif aqi == 3:
            return "Limit outdoor exposure. Consider wearing a mask."
        elif aqi == 4:
            return "Avoid outdoor activities. Wear N95 mask."
        else:
            return "Hazardous air quality. Stay indoors."

    # ================= UI =================
    col1, col2 = st.columns([1,5])

    with col1:
        if st.button(" Refresh"):
            st.cache_data.clear()
            st.rerun()

    st.selectbox(
        "Select City",
        city_list,
        key="live_city_widget",
        on_change=update_city
    )

    city = st.session_state.live_city
    st.write(f" Current City: **{city}**")

    # ================= DISPLAY =================
    try:
        data = get_live_pollution(city, API_KEY)

        def safe(x):
            return f"{x:.2f}" if x is not None else "N/A"

        # METRICS
        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)

        c1.metric("PM2.5", safe(data["PM2.5"]))
        c2.metric("PM10", safe(data["PM10"]))
        c3.metric("NO2", safe(data["NO2"]))
        c4.metric("O3", safe(data["O3"]))
        c5.metric("SO2", safe(data["SO2"]))
        c6.metric("CO", safe(data["CO"]))

        # AQI
        st.subheader(" Air Quality Index")
        st.metric("AQI", data["AQI"])
        st.success(get_aqi_status(data["AQI"]))

        # HEALTH
        st.subheader(" Health Advice")
        st.info(health_advice(data["AQI"]))

        st.caption(f"Updated: {data['time']}")

        # FORECAST
        st.subheader(" 4-Day Forecast")
        forecast = get_forecast(city, API_KEY)

        if not forecast.empty:
            st.dataframe(forecast, use_container_width=True)
            st.line_chart(forecast.set_index("Time"))
        else:
            st.warning("No forecast data available")

    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
