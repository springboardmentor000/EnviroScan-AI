#  EnviroScan: AI-Based Pollution Source Identification System using Geospatial Analytics

---

## 1. Introduction

Air pollution is one of the most serious environmental challenges affecting urban and rural areas. While many systems measure pollutant levels like PM2.5 or NO2, they **do not identify the actual source of pollution**.

This creates a major gap:
 - We know pollution levels  
 - But we don’t know the cause  

EnviroScan solves this problem by combining:
- Machine Learning
- Environmental Data
- Geospatial Visualization

 The system not only measures pollution but also predicts:
✔ What caused the pollution  
✔ Where it is concentrated  
✔ How severe it is  

---

##  2. Problem Statement

Traditional systems:
- Only show pollutant values
- Do not classify pollution sources
- Do not provide actionable insights

EnviroScan provides:
-  Source identification  
-  Data-driven insights  
- Visual representation (maps + dashboard)  



##  3. Objectives

The main objectives of this project are:

1. To collect real-world environmental data  
2. To preprocess and structure the dataset  
3. To classify pollution sources using rule-based logic  
4. To train a machine learning model  
5. To visualize pollution using maps  
6. To build a real-time interactive dashboard  

---

##  4. System Architecture
Data Collection → Data Cleaning → Feature Engineering → Source Labeling → Model Training → Dashboard + Map


### Explanation:

1. Data is collected from APIs  
2. Cleaned and converted into structured format  
3. Features like AQI, time, location are created  
4. Pollution source is labeled  
5. Model is trained  
6. Results are visualized  

---

#  5. MODULE-WISE EXPLANATION

---

## 🔹 Module 1: Data Collection

This module gathers raw environmental data.

###  Data Collected:

####  Air Pollution Data
- PM2.5
- PM10
- NO2
- CO
- SO2
- O3

####  Weather Data
- Temperature
- Humidity
- Wind Speed

####  Location Data
- Latitude
- Longitude
- Timestamp

---

## 🔹 Module 2: Data Cleaning & Feature Engineering

Raw data is not directly usable. It must be cleaned and enhanced.

###  Cleaning Steps:
- Removed duplicate rows
- Handled missing values
- Standardized column formats

---

###  Feature Engineering

####  Time Features
- Hour → helps detect peak pollution time
- Day → pattern analysis
- Month → seasonal trends

####  Spatial Features
- Latitude
- Longitude

####  AQI Score Calculation
AQI_score = (PM2_5 + PM10 + NO2 + CO + SO2 + O3) / 6


This gives a normalized pollution indicator.

---

## 🔹 Module 3: Source Labeling (VERY IMPORTANT)

This is the core logic that creates the **target variable**.

###  Why needed?
Machine learning requires labeled data. Since real labels are not available, we create them using rules.

---

###  Classification Logic

| Source | Condition | Meaning |
|------|----------|--------|
| Vehicular | High NO2 + CO | Traffic pollution |
| Industrial | High SO2 | Factory emissions |
| Agricultural | High PM2.5 | Crop burning |
| Urban Dust | High PM10 | Road dust |
| Natural | High humidity + low wind | Fog/stagnation |

---

### Output Columns Created

- `source_domain` → type of pollution  
- `cause_type` → human / natural / burning  
- `confidence` → strength of prediction  

---

## 🔹 Module 4: Model Training

This module builds the machine learning model.

---

###  Dataset
Infosys_dataset.csv 

###  Data Preparation

- Removed duplicates  
- Removed missing values  
- Selected important features  

###  Features Used
PM2_5, PM10, NO2, CO, SO2, O3,
AQI_score,
temperature_C, humidity_%, wind_speed_mps


---

###  Target Variable
source_domain


---

###  Models Used

#### 1. Random Forest
- Ensemble model (multiple decision trees)
- Gives high accuracy
- Handles non-linear data

#### 2. Decision Tree
- Simple and interpretable
- Used for comparison

---

###  Hyperparameter Tuning

GridSearchCV used to find best parameters:

- n_estimators → number of trees  
- max_depth → depth of tree  

---

###  Model Evaluation

- Accuracy Score  
- Classification Report (Precision, Recall, F1)  
- Confusion Matrix  

---

###  Model Output
pollution_model.pkl


---

## 🔹 Module 5: Geospatial Visualization

This module visualizes pollution on a map.

---

###  Features

✔ Heatmap (PM2.5 intensity)  
✔ Colored markers (based on source)  
✔ Popup details (city, AQI, source)  
✔ Legend for understanding  

---

###  Color Mapping

| Source | Color |
|------|------|
| Vehicular | Blue |
| Industrial | Red |
| Agricultural | Green |
| Urban Dust | Orange |
| Natural | Purple |

---

###  Output
pollution_heatmap.html

- Open this file in browser to view map

---

## 🔹 Module 6: Dashboard (Streamlit)

This is the user interface of the system.

---

###  Sections


### 🔎 1. Filters
- Select city
- View specific data

---

###  2. KPI Metrics
- Average PM2.5  
- Average AQI  
- Dominant source  

---

###  3. Alert System

| AQI Score | Status |
|----------|--------|
| > 0.7 | High Pollution |
| > 0.4 | Moderate |
| Else | Normal |

---

###  4. Data Table
- Shows recent records

---

###  5. Graphs

#### Line Graph
Shows pollution trend

#### Histogram
Shows PM2.5 distribution

#### Correlation Heatmap
Shows feature relationships

---

###  6. Interactive Map
- Displays pollution by location

---

###  7. Prediction System

User inputs:
- Pollution values
- Weather values

Model predicts:
    -  Pollution Source



###  8. Download Feature
- Export filtered dataset


#  6. Technologies Used

###  Language
Python

###  Libraries
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Folium
- Joblib


#  7. Project Structure
Infosys/

app.py
module_4_training.py
module_5_map.py
Infosys_dataset.csv
pollution_model.pkl
pollution_heatmap.html
README.md


---

#  8. Challenges Faced

| Issue | Solution |
|------|--------|
| Missing source column | Fixed in Module 3 |
| Graphs empty | Improved filtering |
| Duplicate data | Removed |
| API timeout | Retry logic |

---

#  9. Key Features

✔ AI-based pollution source detection  
✔ Real-time dashboard  
✔ Geospatial heatmap  
✔ Interactive graphs  
✔ Smart alerts  

---

#  10. Future Scope

- Real-time API integration  
- Mobile app  
- Deep learning models  
- Time-series prediction  
- Government integration  

---

#  11. Conclusion

EnviroScan demonstrates how:

  -  AI + Environmental Data + Maps  

can solve real-world pollution problems by:

✔ Identifying causes  
✔ Visualizing hotspots  
✔ Helping decision-making  

---

#  Author

Paturi Bhavana  
B.Tech CSE – SRM University AP  

---
