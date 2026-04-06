#  EnviroScan-AI  
### AI-Powered Pollution Source Identifier using Geospatial Analytics  

---

##  1. Overview  

Air pollution is a major environmental issue in urban areas, caused by sources such as vehicular emissions, industrial activities, agricultural practices, and biomass burning. Monitoring pollution levels is essential; however, identifying the possible sources of pollution is equally important for effective environmental management and control.  

The EnviroScan project aims to analyze environmental pollution data and estimate possible pollution sources by integrating air quality data, weather conditions, and geospatial context. By combining data from multiple sources and applying data processing and machine learning techniques, the system generates a structured dataset for pollution analysis and intelligent source prediction.  

---

##  2. Motivation  

Air pollution has significant impacts on human health, ecosystems, and environmental sustainability. Exposure to pollutants such as particulate matter and harmful gases can lead to serious health issues including respiratory and cardiovascular diseases.  

The motivation behind this project is to develop an intelligent system that integrates pollution data, weather conditions, and geographic context to estimate potential pollution sources and support better decision-making.  

---

##  3. Objectives  

###  Main Objective  
To develop a system that analyzes environmental pollution data and identifies potential pollution sources using contextual information such as weather conditions and geographic features.  

###  Specific Objectives  
- Collect air pollution and weather data using public APIs  
- Extract geographic features (roads, industries, agricultural areas)  
- Clean and preprocess collected data  
- Generate spatial and temporal features  
- Apply rule-based techniques for source labeling  
- Train machine learning models for prediction  
- Visualize results using an interactive dashboard  

---

##  4. Data Sources  

The EnviroScan system integrates multiple reliable data sources to collect environmental, weather, and geospatial information required for pollution analysis and source identification.  

###  OpenWeatherMap – Air Pollution Data  
Provides pollutant parameters such as PM2.5, PM10, NO₂, CO, SO₂, and O₃. These parameters are essential for understanding pollution intensity.  

###  OpenWeatherMap – Weather Data  
Includes temperature, humidity, pressure, wind speed, and wind direction, which influence pollutant dispersion.  

###  OpenStreetMap (Overpass API) – Geospatial Data  
Provides environmental features such as roads, industrial zones, dump sites, and agricultural land to identify pollution sources based on location.  

---

##  5. System Architecture

Data Collection  
↓  
Data Cleaning & Feature Engineering  
↓  
Source Labeling  
↓  
Model Training & Prediction  
↓  
Geospatial Visualization  
↓  
Dashboard & Alerts

---

##  6. Implementation  

###  Module 1 — Data Collection  
- Pollution data: PM2.5, PM10, NO₂, CO, SO₂, O₃  
- Weather data: Temperature, Humidity, Wind Speed, Wind Direction  
- Attributes: Latitude, Longitude, Timestamp, City  

**Selected Cities:**  
- Delhi  
- Mumbai  
- Pune  
- Nagpur  
- Bangalore  

**Geospatial Features:**  
- Roads  
- Industrial Areas  
- Agricultural Land  
- Dump Sites  

**Output:** `data/raw_data.csv`  

---

###  Module 2 — Data Cleaning & Feature Engineering  

**Cleaning Steps:**  
- Remove duplicates  
- Convert timestamps  
- Handle invalid values  
- Maintain hourly data  

**Missing Value Handling:**  
- Interpolation  
- City-wise median  
- Global median  

**Feature Engineering:**  
- Hour, Day, Season  
- Spatial distances  

**Pollution Classification:**  
- Low (≤30)  
- Moderate (≤60)  
- High (≤120)  

**Output:** `data/cleaned_data.csv`  

---

###  Module 3 — Source Labeling & Simulation  

This module assigns pollution source labels using a rule-based approach.  

**Source Categories:**  
- Burning  
- Vehicular  
- Industrial  
- Agricultural  
- Natural  

**Labeling Logic:**  
- Burning → High PM2.5 + CO  
- Vehicular → Near roads + High NO₂/CO  
- Industrial → High SO₂ or CO  
- Agricultural → Moderate PM2.5 in summer  
- Natural → Default  

A new column **`pollution_source`** is created.  

**Output:** `data/labeled_data.csv`  

---

###  Module 4 — Model Training  

**Features Used:**  
- Location: Latitude, Longitude  
- Pollution: PM2.5, PM10, NO₂, CO, SO₂, O₃  
- Weather: Temperature, Humidity, Wind Speed  
- Spatial: Distance to road, industry, dump, farm  

**Models:**  
- Decision Tree  
- Random Forest  
- XGBoost  

**Evaluation:**  
- Accuracy  
- Cross-validation  
- Confusion matrix  

**Outputs:**  
- `models/best_model.pkl`  
- `outputs/predictions.csv`  

---

###  Module 5 — Visualization  

- Interactive map using Folium  
- Heatmap (PM2.5 intensity)  
- City markers with source types  

**Color Coding:**  
- Vehicular → Blue  
- Industrial → Red  
- Agricultural → Green  
- Burning → Orange  
- Natural → Purple  

**Output:** `outputs/pollution_map.html`  

---

###  Module 6 — Dashboard  

Built using Streamlit  

**Features:**  
- City filtering  
- Pollution indicators (PM2.5, NO₂, CO)  
- Trend graphs  
- Source distribution  

**Prediction Modes:**  
- Quick Mode  
- Advanced Mode  

**Sections:**  
- Dashboard  
- Prediction  
- Map  
- Download  

**Technologies:**  
- Streamlit  
- Pandas  
- Plotly  
- Joblib  

---

##  Final Outcome  

EnviroScan provides:  
- Pollution monitoring  
- Source prediction  
- Interactive visualization  
- Decision-support insights  

---

##  Technologies Used  

- Python  
- Machine Learning  
- Geospatial Analysis  
- Streamlit  
- OpenWeatherMap API  
- OpenStreetMap  

---

##  Conclusion  

This project combines data analytics, machine learning, and geospatial visualization to create a powerful system for pollution monitoring and source identification, making environmental analysis more accessible and actionable.
