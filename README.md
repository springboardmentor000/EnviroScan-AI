EnviroScan – AI Pollution Monitoring & Source Detection System
________________________________________
1. Introduction
Air pollution is one of the most serious environmental challenges affecting modern cities and regions. Although existing systems can measure pollutant levels such as PM2.5 and NO₂, they are unable to identify the actual source of pollution, which limits effective decision-making.
To address this limitation, EnviroScan is developed as an intelligent system that integrates:
•	Machine Learning 
•	Environmental Data Analysis 
•	Geospatial Visualization 
The system not only monitors pollution levels but also:
•	Identifies the source of pollution 
•	Detects high-risk pollution zones 
•	Provides real-time analytical insights 
________________________________________
2. Problem Statement
Traditional pollution monitoring systems:
•	Only display pollutant concentrations 
•	Do not classify pollution sources 
•	Lack real-time actionable insights 
As a result, it becomes difficult for authorities and individuals to understand the root cause of pollution and take corrective actions.
EnviroScan overcomes these challenges by:
•	Predicting pollution sources using AI 
•	Integrating multiple environmental datasets 
•	Providing interactive dashboards and visual maps 
________________________________________
3. Objectives
The key objectives of this project are:
•	To collect environmental data using APIs 
•	To preprocess and clean the dataset 
•	To extract meaningful features from raw data 
•	To classify pollution sources using rule-based logic 
•	To train machine learning models for prediction 
•	To visualize pollution data using maps 
•	To develop a real-time interactive dashboard 
________________________________________
4. System Architecture
Workflow
Data Collection → Data Cleaning → Feature Engineering → Source Labeling → Model Training → Visualization (Dashboard & Map)
Explanation
•	Data is collected from APIs such as OpenWeather and OpenAQ 
•	The collected data is cleaned and structured 
•	Features like time, season, and spatial distances are generated 
•	Pollution sources are labeled using rule-based logic 
•	Machine learning models are trained on labeled data 
•	Results are visualized through an interactive dashboard and map 
________________________________________
 Module-Wise Documentation
________________________________________
 Module 1: Data Collection


module1.py
Objective:
To generate a comprehensive dataset by collecting pollution and weather data from multiple Indian cities.
Key Functionalities:
•	Integrates OpenWeather and OpenAQ APIs 
•	Covers multiple states and cities across India 
•	Generates a dataset with 150 records 
Data Collected:
•	Pollutants: PM2.5, PM10, NO₂, CO, SO₂, O₃ 
•	Weather: Temperature, Humidity, Pressure, Wind Speed 
•	Location: Latitude, Longitude, Timestamp 
Additional Features:
•	Distance to road 
•	Distance to industrial areas 
•	Distance to dump sites 
Special Handling:
•	Uses fallback values when API data is unavailable 
•	Adds slight geographic variation for realism 
Output:
dataset.csv
________________________________________
 Module 2: Data Cleaning & Feature Engineering
 
 
 module2.py
Objective:
To clean raw data and transform it into a structured format suitable for machine learning.
Processing Steps:
•	Removal of duplicate records 
•	Handling missing values using interpolation and median filling 
•	Conversion of timestamps into standard datetime format 
Feature Engineering:
•	Temporal features: hour, day, month, day of week 
•	Seasonal classification: Winter, Summer, Monsoon, Post-Monsoon 
•	Validation: removal of invalid (negative) values 
•	Feature scaling using MinMaxScaler 
Output:
clean_environment_dataset.csv
________________________________________
Module 3: Source Labeling


module3.py
Objective:
To assign pollution source labels using rule-based logic.
Approach:
Since real-world labels are unavailable, heuristic rules are used.
Labeling Criteria:
•	Vehicular → Near roads with high NO₂ 
•	Industrial → Near industries with high SO₂ 
•	Burning → Near dump areas with high PM2.5 
•	Agricultural → High PM2.5 during specific seasons 
•	Natural → Default category 
Additional Feature:
•	Confidence score (0.7 – 0.95 range) 
Output:
labeled_environment_dataset.csv
________________________________________
 Module 4: Model Training & Prediction
 
 
 module4.py
Objective:
To train machine learning models for predicting pollution sources.
Models Used:
•	Decision Tree 
•	Random Forest 
•	XGBoost (optional) 
Process:
•	Encoding categorical variables 
•	Splitting dataset into training and testing sets 
•	Applying 5-fold cross-validation 
•	Training and evaluating multiple models 
Evaluation Metrics:
•	Accuracy 
•	Classification report 
Outputs:
•	pollution_source_model.pkl 
•	label_encoder.pkl 
•	city_encoder.pkl 
Additional Features:
•	User input-based prediction 
•	Pollution alert system 
•	Suggestion system based on prediction 
________________________________________
 Module 5: Geospatial Visualization
 
 
 module5.py
Objective:
To visualize pollution data and predicted sources using interactive maps.
Features:
•	Interactive map using Folium 
•	Heatmap showing PM2.5 intensity 
•	Color-coded markers representing pollution sources 
•	Popups displaying location and pollution details 
•	Highlighting high pollution zones 
Output:
pollution_map.html
________________________________________
 Module 6: Interactive Dashboard
 module6.py
Objective:
To provide a user-friendly interface for real-time pollution monitoring and analysis.
Technology Used:
Streamlit
Key Features:
•	Home page with modern UI 
•	Personal pollution analysis 
•	Real-time dashboard 
User Mode:
•	Accepts pollution input values 
•	Predicts pollution source 
•	Displays confidence score and forecast graph 
•	Dynamic background based on pollution level 
Dashboard Mode:
•	City and state selection 
•	Real-time pollution metrics display 
•	Source prediction 
•	Trend and distribution graphs 
•	Interactive heatmap visualization 
•	High pollution alerts 
•	Dataset download option 
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
