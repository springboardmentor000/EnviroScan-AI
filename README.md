# EnviroScan
EnviroScan: AI-Powered Pollution Source Identifier using Geospatial Analytics

Project Overview:

Air pollution monitoring systems generally measure pollutant levels but do not identify the specific sources of pollution. This limitation makes it difficult for authorities and urban planners to implement targeted mitigation strategies. The EnviroScan system uses machine learning, weather analytics, and geospatial data to identify the most likely source of pollution such as vehicular emissions, industrial activity, agricultural burning, waste burning, or natural causes. The system integrates multiple data sources including: Air quality monitoring data, Weather information, Geospatial infrastructure features. Using this integrated dataset, the system predicts pollution sources and prepares the data for further visualization and analysis.

Milestone 1 (Week 1–2)

This milestone includes:
Module 1 – Data Collection
Module 2 – Data Cleaning and Feature Engineering

Module 1: Data Collection from APIs and Location Databases

Objective:
The objective of this module is to collect air quality data, weather data, and geospatial environmental features from multiple sources. These datasets form the base for pollution source identification.

The following pollutant measurements are collected: PM2.5, PM10, NO₂, SO₂, CO, O₃. These pollutants are widely used indicators of urban air pollution.

Weather Data Collection: Weather information is collected using the OpenWeatherMap API.
The following weather parameters are retrieved: Temperature, Humidity, Wind Speed, Wind Direction.
Weather conditions significantly influence pollutant dispersion.

Geospatial Data Collection: Environmental context is extracted using OSMnx, which retrieves geospatial data from OpenStreetMap. The following spatial features are collected: Road Networks, Industrial Zones, Agricultural Fields, Waste Disposal Sites. These spatial features provide environmental context for identifying pollution sources.

Metadata Tagging: Each collected data point is tagged with these metadata Latitude, Longitude, Timestamp, Monitoring station name. This ensures the dataset contains both spatial and temporal context.

Data Storage: The collected data is stored in a structured dataset enviro_scan_dataset.csv.
This dataset is later used for preprocessing and machine learning.

Module 2: Data Cleaning and Feature Engineering

Objective:
The objective of this module is to preprocess the collected data and generate meaningful features for machine learning models.

Duplicate Removal: Duplicate records are removed to ensure dataset integrity.
Example: df.drop_duplicates()

Handling Missing Values: API responses may sometimes contain missing values.
To address this missing pollutant values are replaced with realistic simulated values.
Numerical values are filled using median values where necessary.
This ensures dataset completeness.

Standardization of Timestamps and Data: All timestamps are converted into a standardized datetime format.
Time is represented in Indian Standard Time (IST).
Several temporal features are derived from timestamps: Hour, Day of Week, Month, Season.
These features help capture time-based pollution patterns.

Spatial Feature Engineering: Spatial proximity features are extracted using OpenStreetMap data through OSMnx.
The following spatial attributes are generated: road_count, industry_count, farmland_count, dump_count.
These features represent the number of environmental infrastructures around monitoring locations.

Dataset Integration: All datasets are merged into a single feature-rich DataFrame containing pollution measurements, weather conditions, spatial environmental features, temporal features                              

Milestone 2(Week 3–4):
Milestone 2 focuses on pollution source labeling and machine learning model development.
Modules included:
Module 3 – Source Labeling and Simulation
Module 4 – Model Training and Source Prediction

Module 3: Source Labeling and Simulation
Objective:
This module assigns pollution source labels using rule-based heuristics derived from environmental knowledge.
Since real pollution source labels are not directly available, heuristic rules are used to simulate labeled training data.

Pollution Source Labeling Rules:
1.Vehicular Pollution
Condition: High road density, High NO₂ concentration
Explanation:
Vehicles emit nitrogen dioxide during fuel combustion.

2.Industrial Pollution
Condition: Industrial zones nearby, High SO₂ concentration
Explanation:
Industrial processes emit sulfur dioxide.

3.Agricultural Pollution
Condition: Farmland nearby, Dry season, High particulate matter
Explanation:
Crop residue burning produces particulate emissions.

4.Waste Burning
Condition: Dump sites nearby, High PM2.5 values
Explanation:
Burning waste generates particulate pollution.

5.Natural Pollution
If none of the above conditions are satisfied, pollution is classified as Natural.

Dataset Preparation:
The labeled dataset is prepared and saved as: enviro_scan_dataset.csv.
This dataset serves as the training dataset for machine learning models.

Module 4: Model Training and Source Prediction

Objective:
The objective of this module is to train machine learning models capable of predicting pollution sources using environmental data.
1.Train-Test Split
The dataset is divided into training and testing datasets using an 80-20 split.
Dataset	Portion
Training Data	80%
Testing Data	20%

Machine Learning Models Used.
The following classification models are implemented:
1.Decision Tree: A rule-based model that classifies data using hierarchical decision structures.
2.Random Forest: An ensemble learning algorithm combining multiple decision trees to improve prediction accuracy.
3.XGBoost: A gradient boosting algorithm designed for high performance and scalability.
4.Cross Validation: To ensure model reliability, 5-Fold Cross Validation is used.
The dataset is divided into five subsets:
1. Train on four subsets
2. Test on one subset
3. Repeat five times
4. Average accuracy is calculated

Model Evaluation Metrics
Model performance is evaluated using: Accuracy, Precision, Recall, F1 Score, Confusion Matrix.
These metrics measure how effectively the model predicts pollution sources.
Model Export: The trained model is saved using joblib.
joblib.dump(best_model,"pollution_model.pkl")

Output file:
pollution_model.pkl
This model is integrated into dashboard.

Final Outputs of the System:
The EnviroScan pipeline generates: Dataset enviro_scan_dataset.csv
Trained Model: pollution_model.pkl
These outputs enable automated pollution source prediction and support environmental monitoring systems.

Milestone 3 (Week 5–6):
Module 5: Geospatial Mapping and Heatmap Visualization
Module 6: Interactive Dashboard and Real-Time Monitoring

Module 5: Geospatial Mapping and Heatmap Visualization
Objective:
To visualize pollution data and predicted sources using geospatial mapping techniques, enabling identification of pollution hotspots and spatial distribution patterns.
 
Geospatial Map Integration:
An interactive map is created using the Folium library.
The map displays pollution data across different locations using latitude and longitude coordinates.
The map is dynamically centered based on the available dataset.

Heatmap Visualization:
A pollution heatmap is generated using PM2.5 values.
Heatmaps represent pollution intensity using color gradients:
Green → Low pollution
Orange → Moderate pollution
Red → High pollution
This helps in quickly identifying pollution hotspots.

Marker-Based Visualization:
Location points are displayed using clustered markers.
Each marker includes:
Tooltip (on hover) → shows PM2.5 value
Popup (on click) → shows:
PM2.5, PM10, NO₂, SO₂, CO, O₃ pollution source.

Filtering Capability:
The dashboard allows filtering based on:
Location (city), Pollution source category.
This enables interactive exploration of pollution data.

High-Risk Zone Identification:
Locations with higher PM2.5 values are highlighted using color-coded markers, helping identify critical pollution zones.

Map Embedding:
The map is embedded in the Streamlit dashboard using: st_folium()
This allows real-time user interaction with the map.

Module 6: Interactive Dashboard and Real-Time Monitoring
Objective:
To develop an interactive web-based dashboard for real-time pollution monitoring, visualization, and AI-based source detection.

Dashboard Development:
The dashboard is built using Streamlit.
It provides a user-friendly interface with multiple sections: Dashboard, Source Detection, Health Audit, Dataset Explorer.

Real-Time Data Integration:
Air Quality Data (OpenAQ API):
Real-time pollutant data is fetched including: PM2.5, PM10, NO₂, SO₂, CO, O₃.
Weather Data (OpenWeather API):

Real-time weather parameters are retrieved: Temperature, Humidity, Wind Speed.
These real-time inputs are used for dynamic prediction.

Source Detection:
Users can input latitude and longitude to analyze pollution at any location.
The system uses a trained Random Forest model to predict the pollution source.

Real-Time Metrics Display:
The dashboard displays environmental metrics in real time: Pollutant concentrations, Weather conditions, Area features (road, industry, farmland counts).
Displayed using structured metric cards for readability.

   Pollution Insights Visualization:
   Pie Chart (Source Distribution): Shows contribution of different pollution sources based on PM2.5 values.

Alert System:
The system provides real-time alerts:
High pollution (PM2.5 > 100)
Moderate pollution (PM2.5 > 60)
Safe conditions

Dataset Explorer:
Users can view the complete dataset in tabular format.
The dataset can be downloaded as a CSV file for further analysis.

Health Audit Module:
This module identifies unsafe pollution conditions.
Displays records where PM2.5 exceeds safe limits.
Highlights pollution violations.
Provides system safety status.

User Interaction Features:
Dropdown filters (city, source type), Real-time analysis input (latitude, longitude), Interactive map navigation.

System Integration:
The dashboard integrates: Dataset (enviro_scan_dataset.csv).
Real-time APIs (OpenAQ, OpenWeather)
