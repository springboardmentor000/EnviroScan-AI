# EnviroScan-AI
EnviroScan AI-Powered Pollution Source Identifier using Geospatial Analytics
An AI-powered pollution source identification system for Hyderabad, India. EnviroScan collects real-time air quality data, weather parameters, and geospatial features to predict whether pollution is coming from vehicular traffic, industrial activity, agricultural burning, open waste, or natural sources.
Live Dashboard: https://enviroscanai-app.streamlit.app/

What It Does
Most pollution monitoring systems only tell you how much pollution is present. EnviroScan tells you where it is coming from — and displays the results on interactive maps with real-time alerts.

Collects live data from 14 TSPCB monitoring stations across Hyderabad
Predicts pollution source using a trained Random Forest model (96.7% accuracy)
Displays interactive heatmaps and source maps of Hyderabad
Alerts when PM2.5 crosses safe thresholds
Provides downloadable pollution reports


Project Structure
EnviroScan/
│
├── milestone1_module1.py          # Data collection (OpenAQ + OWM + OSM)
├── milestone1_module2.py          # Data cleaning and feature engineering
├── milestone2_module3.py          # Source labeling and simulation
├── milestone2_module4.py          # ML model training and evaluation
├── milestone3_module5.py          # Geospatial maps and heatmaps
├── milestone3_module6.py          # Streamlit dashboard
│
├── Hyderabad_pollution_latest.csv     # Raw collected data
├── Hyderabad_pollution_cleaned.csv    # Cleaned and engineered dataset
├── Hyderabad_pollution_labeled.csv    # Labeled dataset (214 rows)
│
├── models/
│   ├── random_forest.pkl          # Best model (96.7% CV accuracy)
│   ├── xgboost.pkl                # XGBoost classifier
│   ├── decision_tree.pkl          # Decision Tree classifier
│   └── label_encoder.pkl          # Label encoder for source classes
│
└── outputs/
    ├── map1_pm25_heatmap.html         # PM2.5 heatmap
    ├── map2_pollution_sources.html    # Source type map
    ├── map3_combined.html             # Combined map with layer toggle
    ├── module3_label_distribution.png # Source label charts
    └── module4_model_results.png      # Model performance charts

Pollution Sources
The system classifies pollution into 5 source categories:
SourceKey IndicatorsExample StationsVehicularHigh NO2, high CO, many roads nearbyZoo Park, Somajiguda, New MalakpetIndustrialHigh SO2, industrial zones nearbyBollaram, Nacharam TSIIC, SanathnagarAgriculturalHigh PM10, low NO2, farmland nearbyKompally Municipal OfficeBurningVery high CO, dump sites nearbyECIL KapraNaturalLow PM2.5, low NO2, high O3Low-density outskirt areas

Model Performance
ModelTest AccuracyCV Score (5-Fold)Random Forest100%96.7%XGBoost100%96.3%Decision Tree97.7%96.3%
Random Forest was selected as the best model and is used in the live dashboard.

Installation
1. Clone the repository
bashgit clone https://github.com/yourusername/enviroscan.git
cd enviroscan
2. Install dependencies
bashpip install requests pandas numpy scikit-learn xgboost joblib folium streamlit streamlit-folium matplotlib
3. Add your API keys
Open milestone1_module1.py and add your keys:
pythonOPENAQ_KEY      = "your_openaq_key_here"
OPENWEATHER_KEY = "your_openweathermap_key_here"
Get free keys at:

OpenAQ: https://api.openaq.org/
OpenWeatherMap: https://openweathermap.org/api


Running the Project
Run each module in order:
bash# Step 1: Collect data
py milestone1_module1.py

# Step 2: Clean and engineer features
py milestone1_module2.py

# Step 3: Label sources and simulate training data
py milestone2_module3.py

# Step 4: Train ML models
py milestone2_module4.py

# Step 5: Generate geospatial maps
py milestone3_module5.py

# Step 6: Launch the dashboard
streamlit run milestone3_module6.py

Data Sources
SourceWhat is collectedAPI KeyOpenAQ v3PM2.5, PM10, NO2, SO2, O3, COFree (required)OpenWeatherMapTemperature, Humidity, Wind Speed, Wind DirectionFree (required)OpenStreetMap (Overpass)Roads, industrial zones, dump sites, farmlandNot required

Dashboard Pages
PageDescriptionDashboardKPI cards, station table, PM2.5 chart, source pie chartMapsInteractive Folium heatmaps and source marker mapsPredict SourceEnter custom values and get instant ML predictionStation AnalysisDeep dive into any individual stationAlertsShows all stations exceeding PM2.5 thresholdDownload ReportExport full data as a timestamped CSV

AQI Scale (India Standard)
AQI CategoryPM2.5 RangeHealth ImpactGood0 - 30 µg/m³Minimal impactSatisfactory31 - 60 µg/m³Minor discomfort for sensitive peopleModerate61 - 90 µg/m³Breathing issues for asthma patientsPoor91 - 120 µg/m³Discomfort for general publicVery Poor121 - 250 µg/m³Serious health effectsSevere250+ µg/m³Emergency conditions

Key Findings

Sanathnagar (PM2.5: 157 µg/m³) is the most polluted station — Industrial source
Industrial activity is the dominant pollution source across Hyderabad (8 of 14 stations)
Zoo Park has the highest NO2 (88.2 µg/m³) — consistent with heavy vehicular traffic
ECIL Kapra has the cleanest air (PM2.5: 15.35) but shows Burning signals from nearby dump sites


Live Deployment
The dashboard is publicly deployed on Streamlit Cloud:
https://enviroscanai-app.streamlit.app/

Project Milestones
MilestoneWeekModulesMilestone 1Week 1-2Data Collection, Data CleaningMilestone 2Week 3-4Source Labeling, Model TrainingMilestone 3Week 5-6Geospatial Maps, Streamlit DashboardMilestone 4Week 7-8Documentation, Presentation, Deployment

Technologies Used
Python, pandas, numpy, scikit-learn, XGBoost, Folium, Streamlit, OpenAQ API, OpenWeatherMap API, OpenStreetMap (Overpass API), joblib, matplotlib
