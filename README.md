EnviroScan – AI Pollution Monitoring & Source Detection System
________________________________________
1. Introduction
Air pollution is one of the most serious environmental challenges affecting modern cities and regions. Although existing systems can measure pollutant levels such as PM2.5 and NO₂, they are unable to identify the actual source of pollution, which limits effective decision-making.
To address this limitation, EnviroScan is developed as an intelligent system that integrates:
* Machine Learning 
* Environmental Data Analysis 
* Geospatial Visualization 
The system not only monitors pollution levels but also:
*	Identifies the source of pollution 
*	Detects high-risk pollution zones 
*	Provides real-time analytical insights 
________________________________________
2. Problem Statement
Traditional pollution monitoring systems:
*	Only display pollutant concentrations 
*	Do not classify pollution sources 
*	Lack real-time actionable insights 
As a result, it becomes difficult for authorities and individuals to understand the root cause of pollution and take corrective actions.
EnviroScan overcomes these challenges by:
*	Predicting pollution sources using AI 
*	Integrating multiple environmental datasets 
*Providing interactive dashboards and visual maps 
________________________________________
3. Objectives
The key objectives of this project are:
 *To collect environmental data using APIs 
* To preprocess and clean the dataset 
* To extract meaningful features from raw data 
*	To classify pollution sources using rule-based logic 
*	To train machine learning models for prediction 
*	To visualize pollution data using maps 
*	To develop a real-time interactive dashboard 
________________________________________
4. System Architecture
Workflow
Data Collection → Data Cleaning → Feature Engineering → Source Labeling → Model Training → Visualization (Dashboard & Map)
Explanation
*	Data is collected from APIs such as OpenWeather and OpenAQ 
* The collected data is cleaned and structured 
*	Features like time, season, and spatial distances are generated 
*Pollution sources are labeled using rule-based logic 
* Machine learning models are trained on labeled data 
*	Results are visualized through an interactive dashboard and map 
________________________________________
 Module-Wise Documentation
Module-wise Implementation
Module 1: Data Collection

Objective
Collect air quality, weather, and geospatial data from multiple sources to build the base dataset.

Description
This module gathers environmental data required for pollution analysis. Air quality parameters such as PM2.5, PM10, NO₂, SO₂, CO, and O₃ are collected along with weather attributes including temperature, humidity, and wind speed. Geospatial context is extracted using OpenStreetMap to understand surrounding infrastructure.

Data Collected

Air Quality Parameters: PM2.5, PM10, NO₂, SO₂, CO, O₃
Weather Data: Temperature, Humidity, Wind Speed
Geospatial Data: Road networks, industrial zones, farmland, waste sites

Metadata

Latitude
Longitude
Timestamp
Monitoring station

Output

Raw dataset stored as enviro_scan_dataset.csv
Module 2: Data Cleaning and Feature Engineering

Objective
Prepare the dataset for machine learning by cleaning and generating meaningful features.

Description
This module ensures data quality and transforms raw inputs into structured features. Duplicate records are removed, missing values are handled, and timestamps are standardized. Additional temporal and spatial features are created to improve model performance.

Key Steps

Removal of duplicate records
Handling missing values using filling and simulation
Standardization of timestamps (IST)

Feature Engineering

Temporal Features: Hour of day, day of week
Spatial Features:
road_count
industry_count
farmland_count
dump_count

Output

Cleaned and feature-rich dataset
Module 3: Source Labeling and Simulation

Objective
Generate labeled data for supervised learning using rule-based logic.

Description
Since real-world pollution source labels are not available, this module applies heuristic rules based on environmental knowledge to assign source categories.

Labeling Logic

Source Type	Condition
Vehicular	High NO₂ and high road density
Industrial	High SO₂ and nearby industries
Agricultural	High particulate matter and farmland
Waste Burning	Very high PM2.5 levels
Natural	Default classification

Output

Labeled dataset for model training
Module 4: Model Training and Source Prediction

Objective
Train machine learning models to predict pollution sources.

Description
This module builds and evaluates classification models using the prepared dataset. Multiple models are trained and compared to select the best-performing one.

Models Used

Decision Tree
Random Forest
XGBoost

Methodology

Train-test split (75/25)
Feature scaling using StandardScaler
Cross-validation (5-fold)
Hyperparameter tuning

Evaluation Metrics

Accuracy
Precision
Recall
F1 Score

Output

Trained model file (.pkl)
Scaler and encoder files
Module 5: Geospatial Mapping and Visualization

Objective
Visualize pollution data and predictions using maps.

Description
This module integrates geospatial visualization to represent pollution levels and sources across locations. Interactive maps allow users to explore pollution distribution and identify high-risk zones.

Features

Map visualization using latitude and longitude
Heatmaps for pollution intensity
Marker-based display with detailed information
Identification of high pollution zones

Output

Interactive geospatial visualizations
Module 6: Interactive Dashboard and Real-Time Monitoring

Objective
Develop a user-friendly interface for real-time pollution monitoring and prediction.

Description
This module provides a web-based dashboard for users to interact with the system. It integrates real-time APIs and allows users to input location data for pollution analysis.

Features

Real-time air quality and weather data integration
Pollution source prediction using trained model
Display of environmental metrics
Alert system based on pollution thresholds
Dataset exploration functionality

Technologies Used

Streamlit
OpenAQ API
OpenWeather API

Output

Fully functional interactive dashboard
Module 7: Deployment and Integration

Objective
Deploy the system for real-world usage.

Description
The final module focuses on deploying the application using cloud platforms and integrating all components into a complete system.

Key Tasks

Deployment using Streamlit Cloud
Integration with GitHub repository
Dependency management using requirements.txt
Model and dataset integration

Output

Live deployed application accessible via URL
________________________________________
5. Technologies Used
•	Python

•	Pandas, NumPy 

•	Scikit-learn 

•	XGBoost 


•	Streamlit 

•	Plotly 

•	Folium 

•	Joblib 

•	OpenWeather API 

•	OpenAQ API 
________________________________________
6. Conclusion
EnviroScan is a comprehensive AI-based system that integrates:
•	Data collection 
•	Machine learning 
•	Geospatial visualization 
•	Real-time monitoring 
The system helps in:
•	Identifying pollution sources 
•	Detecting pollution hotspots 
•	Supporting informed environmental decisions
