#  EnviroScan: Pollution Source Detection System

EnviroScan is an end-to-end data science and machine learning project designed to monitor air pollution, analyze environmental patterns, and predict pollution sources using real-time and historical data.

---

##  Project Overview

Air pollution is influenced by multiple factors such as vehicular emissions, industrial activity, agricultural burning, and waste disposal. This project integrates multiple data sources, applies feature engineering, and uses machine learning models to classify pollution sources.

The system provides:
- Real-time pollution insights
- Historical trend analysis
- Source prediction using ML
- Interactive visualization dashboard

---

## Objectives

- Collect environmental data from multiple sources  
- Process and engineer meaningful features  
- Label pollution sources using domain-based rules  
- Train machine learning models for classification  
- Build an interactive dashboard for visualization  

---

#  Milestone 1: Data Collection and cleaning 

### Module 1: Data Collection

### Objective
Gather environmental data required for analysis and modeling.

### Files
- collect_air_quality.py  
- collect_weather.py  
- collect_osm_features.py  
- config.py  

### Data Collected

Air Quality:
- PM2.5, PM10, NO2, SO2, CO, O3  

Weather:
- Temperature, Humidity, Wind Speed  

Geographical:
- Road density, Industry zones, Agriculture areas, Dump areas  


### Module 2: Data Processing & Feature Engineering

### Files
- merge_data.py  
- feature_engineering.py  
- data_cleaning.py  

### Steps

1. Merge Data  
   Combined datasets using latitude, longitude, and timestamp  
   Output: merged_dataset.csv  

2. Feature Engineering  
   - Time Features: hour, day, month, weekday  
   - Pollution Features: total_pollution, pm_ratio  
   - Interaction Features: wind_dispersion, humidity_pm  

   Output: feature_dataset.csv  

3. Data Cleaning  
   - Removed unnecessary columns (timestamp_x, timestamp_y, unit)  
   - Handled missing values  
   - Removed duplicates  

   Output: cleaned_dataset.csv  

---

#  Milestone 2: Pollution Source Labeling & Model traing


### Module 3: Pollution Source Labeling

### File
- pollution_sources.py  

### Approach
Rule-based labeling using:
- Pollutant values  
- Road density  
- Industrial presence  
- Agricultural activity  

### Labels

0 -> Agricultural Buring  
1 -> Industrial  
2 -> Natural  
3 -> Vehicular    

Output: labeled_dataset.csv  



### Module 4: Model Training

### File
- train_model.py  

### Models Used
- Decision Tree  
- Random Forest  

### Process
- Train-test split (80/20)  
- Feature selection  
- Hyperparameter tuning  
- Class imbalance handling  

### Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

Output: final_model.pkl  

---

#  Milestone 3: Visualization & Deployment

### Files
- app.py  
- live_data.py  

### Features

- City-based pollution analysis  
- Real-time data using OpenWeather API  
- Heatmap visualization  
- 5-day pollutant trend graphs  
- ML-based pollution source prediction  



### Key Features

- Real-time pollution monitoring  
- Multi-city heatmap visualization  
- Historical trend analysis  
- Machine learning prediction  
- Interactive dashboard  
---
#  Milestone 4: app.py



###  Streamlit Dashboard (`app.py`)


###  Home Page

### Features:
- User inputs city name  
- Fetches real-time pollution data  
- Displays:

#### Current Pollution Metrics
- PM2.5, PM10, NO2, SO2, CO, O3  

####  Heatmap
- Visualizes pollution distribution within city  

####  5-Day Trend Analysis
- Separate graphs for:
  - PM2.5  
  - NO2  
  - SO2  
  - CO  
  - O3  

#### Pollution Source Prediction
- Uses trained ML model  
- Predicts:
  - Vehicular  
  - Industrial  
  - Agricultural  
  - Natural  



###  Map Page

### Features:

### 1. Pollution Heatmap
- Displays intensity of pollution across India  


### 2. Pollution Source Map
- Uses ML model predictions  
- Displays markers with:
  - Color-coded sources  
  - Icons (car, industry, leaf, cloud)  


### 3. Combined View
- Heatmap + Source markers together  


###  Interactive Popups

Each marker displays:

- City name  
- Predicted pollution source  
- Pollution values (PM2.5, PM10, NO2, etc.)  
- Weather data  
- Geographic features  


### Manual Data Refresh

- Button to trigger live data collection  
- Updates map instantly  
- Ensures latest pollution data is used  


#  Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  
- Folium  
- Plotly  
- OpenWeather API  
- Joblib  

---

#  How to Run

1. Install dependencies  
pip install pandas numpy scikit-learn streamlit folium plotly requests joblib  

2. Run app  
streamlit run app.py  
 
---

#  Future Improvements

- AQI-based color indicators  
- Time-series forecasting  
- Cloud deployment  
- Improved model accuracy  

