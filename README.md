**EnviroScan AI-Powered Pollution Source Identifier using Geospatial Analytics**

**Project Overview**

EnviroScan is an intelligent environmental monitoring system that uses machine learning and geospatial analytics to identify the likely sources of air pollution.

Traditional air quality monitoring systems measure pollutant concentrations but often fail to determine the actual source of pollution. EnviroScan solves this problem by integrating pollution data, weather conditions, and geographic information to classify the most probable pollution source.

The system also provides data visualizations and a prediction dashboard to help environmental researchers, urban planners, and government agencies understand pollution patterns and take preventive measures.


**Objectives**

Collect air pollution data from environmental APIs
Integrate weather and geospatial data
Perform data cleaning and feature engineering
Label pollution sources using rule-based logic
Train machine learning models to classify pollution sources
Visualize pollution patterns through charts and maps
Build an interactive dashboard for pollution monitoring


**Dataset Information**

The dataset used in this project was created by combining air quality data, weather data, and geographic features.

| Attribute      | Value                          |
| -------------- | ------------------------------ |
| Total Records  | 7056                           |
| Total Features | 36                             |
| Locations      | Vijayawada, Hyderabad



**Main Features**

PM2.5
PM10
NO₂
SO₂
CO
O₃
Temperature
Humidity
Wind Speed
Wind Direction
Road Count
Industrial Count
Farmland Count
Waste Count
Time Based Features (Hour, Day, Month, Season)


**Technologies Used**

Python Programming language 

**Libraries**

Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
XGBoost
Folium
OSMnx
Joblib
os

**Tools**

Visual Studio Code
GitHub
Command prompt
Streamlit

**APIs**

OpenAQ API – Air pollution data
OpenWeatherMap API – Weather data
OpenStreetMap – Geographic features

-------------------------------------------------------------------------------------------------------------------------------------------------------
**System architecture**

Data Collection
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Source Labeling
      ↓
Model Training
      ↓
Hyperparameter Tuning
      ↓
Cross Validation
      ↓
Prediction Dashboard



**Project Milestones and Modules
Milestone 1 – Data Collection and Data Preparation
Module 1: Data Collection**

• Collected air quality data (PM2.5, PM10, NO2, CO, SO2, O3) from the OpenAQ API for selected monitoring locations.

• Collected weather data (temperature, humidity, wind speed, wind direction) from the OpenWeatherMap API.

• Extracted nearby environmental features such as roads, industrial zones, dump sites, and agricultural fields using OpenStreetMap via the OSMnx library.

• Tag each data point with latitude, longitude, timestamp, and source API metadata.

• Stored the collected data in structured CSV or JSON format for preprocessing and modeling.


**Module 2: Data Cleaning and Feature Engineering**

• Remove duplicate entries and invalid records from the collected datasets.

• Handled missing values using interpolation and mean/median imputation techniques.

• Standardize timestamps, GPS coordinates, and pollutant units to maintain consistency.

• Normalize pollutant concentrations and weather variables to ensure consistent model input scaling.

•To prepare the dataset for machine learning algorithms, numerical features are scaled using Scikit-learn, MinMaxScaler. Normalization converts values into a 0–1 range

• Calculate spatial proximity features such as distance to nearest road, industrial area, or waste dumping site.

• Derive temporal features including hour of day, day of week, and seasonal indicators to capture pollution patterns.

• Combine pollution, weather, and geographic datasets into a unified feature-rich dataset for machine learning.



**Milestone 2 – Pollution Source Identification and Model Training
Module 3: Source Labeling and Simulation**

• Define rule-based conditions to label pollution sources based on environmental features and pollutant levels.

**Examples of labeling rules:**

• Close to main road + high NO₂ or CO → Vehicular pollution

• Near industrial zones + high SO₂ or PM2.5 → Industrial pollution

• Near farmland + dry season + high PM10 or PM2.5 → Agricultural pollution

• Near waste sites + high CO or PM → Burning pollution

• Low pollutant levels → Natural pollution

• Apply heuristic rules to assign labels such as:

Vehicular

Industrial

Agricultural

Burning

Natural

• Simulate labeled training data when ground-truth labels are not available.

• Validate labeling logic using environmental domain knowledge and literature references.

• Generate a final labeled dataset for machine learning model training.



**Data Analysis using Plots**

Before training the machine learning models, an exploratory data analysis (EDA) was performed to understand the patterns and relationships within the pollution dataset.

1)Boxplot:

From the plot, we can see that PM2.5 and PM10 have larger variation, indicating fluctuating particulate pollution. Other pollutants like SO2 and NO2 show lower values with fewer variations, suggesting relatively stable levels across stations.

2)Line graph:

Each line graph represents the hourly average concentration of a pollutant. The shaded areas highlight morning rush hours (around 7–10 AM) and evening rush hours (around 5–8 PM). During these periods, pollutants like PM2.5, PM10, and CO tend to increase, likely due to traffic and human activities. Ozone (O3) generally increases during the afternoon due to sunlight and chemical reactions in the atmosphere.

3)Pie chart:

The bar chart displays the total count of records for each source, while the pie chart shows the percentage share. The results indicate that Natural sources contribute the highest share (about 37.5%), followed by Industrial (21.8%) and Burning activities (18.4%). Agricultural and vehicular sources contribute smaller portions. This distribution helps understand which sources are most responsible for pollution in the dataset.

4)Heatmap:

The correlation heatmap illustrates the relationship between different variables such as pollutants and weather conditions. Values range from -1 to 1, where values close to 1 indicate strong positive correlation and values close to -1 indicate strong negative correlation. The heatmap shows a strong positive correlation between PM2.5 and PM10, meaning they tend to increase together. Temperature and humidity show a strong negative correlation, indicating that when temperature increases, humidity generally decreases

5)Bar chart  

This figure shows the average pollutant levels for each day of the week. Each subplot represents a pollutant such as PM2.5, PM10, NO2, SO2, CO, and O3. The bars compare weekday values (Monday–Friday) with weekend values (Saturday–Sunday). In many cases, pollutant levels slightly increase during mid-week due to higher industrial and transportation activity. Weekend levels are often slightly lower, indicating reduced human activity. This analysis helps identify weekly patterns in pollution levels.


**Module 4: Model Training and Source Prediction**

• Split the labeled dataset into training and testing sets (80/20 split).

• Train classification models including:

1)Random Forest

2)Decision Tree

3)XGBoost

4)Gradient Boosting

• Use pollutant concentrations, weather variables  features as input features.
• Predict the target variable: pollution_source
• Perform hyperparameter tuning using GridSearchCV and RandomizedSearchCV to optimize model performance.
• Apply Stratified K-Fold Cross Validation to ensure balanced class distribution during training.
• Evaluate model performance using:
Accuracy
Precision
Recall
F1 Score

Confusion Matrix

• Select the best-performing model based on evaluation metrics.
• Export the trained model using Joblib or Pickle for integration with the EnviroScan dashboard.

**Machine Learning Models**

**Decision Tree**

A Decision Tree is a supervised learning algorithm that splits the dataset based on feature conditions to create a tree-like structure of decisions.
In EnviroScan, the Decision Tree model learns decision rules based on pollutant concentrations, weather conditions, and spatial features to classify pollution sources.

Hyperparameters such as maximum depth, minimum samples split, and minimum samples leaf were tuned using RandomizedSearchCV. Model performance was validated using Stratified K-Fold Cross Validation.

**Random Forest**

Random Forest is an ensemble learning algorithm that builds multiple decision trees and combines their predictions to improve accuracy and reduce overfitting.
In this project, Random Forest analyzes complex relationships between pollutants and environmental factors.

Hyperparameters such as:number of trees , maximum depth , minimum samples split were optimized using GridSearchCV. Cross-validation was used to ensure robust performance.

**XGBoost**

XGBoost (Extreme Gradient Boosting) is a powerful gradient boosting algorithm that builds trees sequentially to minimize prediction errors.
Each tree learns from the mistakes of the previous trees, improving overall model accuracy.

Hyperparameters such as: learning rate, number of estimators, maximum tree depth, subsample ratio were tuned using GridSearchCV, and model validation was performed using Stratified K-Fold Cross Validation.

**Gradient Boosting**

Gradient Boosting builds models sequentially where each new model corrects the errors of previous models.
The model captures complex nonlinear relationships between pollutants, weather conditions, and geographic features.

Hyperparameter tuning was performed using GridSearchCV, and model stability was validated through cross-validation techniques.


| Model             | Training Accuracy | Testing Accuracy | Cross Validation F1 Score |
| ----------------- | ----------------- | ---------------- | ------------------------- |
| Decision Tree     | 95.27%            | 90.08%           | 0.915                     |
| Random Forest     | 99.10%            | 88.51%           | 0.905                     |
| XGBoost           | 97.54%            | 91.64%           | 0.942                     |
| Gradient Boosting | 96.64%            | 95.16%           | 0.939                     |



**Important predictors identified:**

PM2.5
PM10
NO₂
CO
O₃


**Milestone 3 - Geospatial Mapping and Heatmap Visualization And Real-Time Dashboard and Alerts
Module-5: Geospatial Mapping and Heatmap Visualization**

This module focuses on visualizing pollution predictions using an interactive map interface. It integrates geospatial data with machine learning outputs to provide a clear and intuitive understanding of pollution distribution across different locations.

A Folium-based map is generated, centered on Vijayawada, displaying multiple layers of environmental insights. A heatmap layer represents pollution severity using intensity gradients, where higher values indicate more critical pollution zones. This helps in quickly identifying high-risk areas.

In addition to the heatmap, the system uses emoji-based markers to represent different predicted pollution sources such as Industrial (🏭), Vehicular (🚗), Agricultural (🌾), Burning (🔥), and Natural (🌿). Each marker includes a detailed popup showing location name, predicted source, severity level, and model confidence score.

The module also supports layer controls, allowing users to toggle visibility of different pollution source categories for better analysis. A custom legend is added to enhance readability and user experience.

To improve realism, the map combines predefined city coordinates with actual dataset locations, ensuring both coverage and data authenticity. The final output is saved as an interactive HTML file that can be viewed in any web browser.

SHAP KernelExplainer is used to analyze the impact of input features such as pollutant levels (PM2.5, PM10, NO₂, O₃, SO₂, CO) and environmental factors (road count, industrial areas, waste sites, farmland). For each predicted pollution source, SHAP waterfall plots are generated to visualize how individual features push the prediction towards a specific class.

Additionally, a SHAP summary (beeswarm) plot is created using a sample of the dataset. This plot provides a global view of feature importance, highlighting which factors most influence the model across all predictions.

Overall, this module transforms raw prediction data into an insightful visual dashboard, enabling easy interpretation of pollution patterns and supporting better environmental decision-making.


**Module-6: Real-Time Dashboard and Alerts**

This module provides a user-friendly and interactive dashboard built using Streamlit for real-time pollution monitoring and analysis. It allows users to explore pollution data, visualize trends, and make predictions dynamically.

The dashboard is divided into three main sections: 
1)Dashboard
2)Map
3)Reports

In the Dashboard tab, users can view key pollution indicators such as PM2.5, PM10, NO₂, and O₃ for selected locations. Interactive sliders are provided to adjust pollutant and environmental parameters, enabling real-time prediction of pollution sources using the trained machine learning model. The predicted source and confidence score are displayed instantly.

The Map tab visualizes pollution data across multiple locations using an interactive Folium map. It includes a heatmap layer to represent pollutant intensity and emoji-based markers to indicate predicted pollution sources. This helps users easily identify pollution hotspots and understand spatial distribution.

In the Reports tab, users can analyze pollution source distribution through tables and pie charts. A trend analysis feature is also included, showing how pollutant levels vary over time. Additionally, users can download location-specific reports in CSV format for further analysis.

The module also includes an email alert system, which allows users to send pollution alerts instantly, making the application more practical for real-world monitoring scenarios.

Overall, this module enhances usability by combining data visualization, machine learning predictions, and interactive controls into a single, intuitive interface.


**Results**

The results demonstrate that combining pollution data, weather parameters, and geospatial features allows machine learning models to effectively identify pollution sources. 

Several models were evaluated using accuracy, precision, recall, F1-score, and confusion matrix to measure their performance.Among all models, Gradient Boosting and XGBoost showed the best balance between accuracy and generalization, making them suitable for integration into the EnviroScan prediction dashboard.

Gradient Boosting achieved the highest testing accuracy of 95.16%, showing strong predictive capability.

The EnviroScan system successfully predicts pollution sources using a Gradient Boosting model with reliable accuracy and confidence scores. The model effectively classifies pollution into categories such as Industrial, Vehicular, Agricultural, Burning, and Natural based on environmental and pollutant features.

The interactive dashboard enables real-time prediction, data filtering, and visualization, making it easy to analyze pollution patterns. The geospatial map highlights high-risk zones using heatmaps, while emoji-based markers improve interpretability of pollution sources.

SHAP analysis provides clear insights into feature importance, showing how different pollutants and environmental factors influence predictions. The reports section further supports analysis through charts, trends, and downloadable data.

Overall, the system delivers accurate predictions along with intuitive visualizations and explainability features.


**Conclusion**

EnviroScan is a comprehensive air pollution monitoring and prediction system that combines machine learning, geospatial visualization, and interactive dashboards. It not only predicts pollution sources but also explains the reasoning behind predictions, improving transparency and trust.

The integration of real-time prediction, heatmaps, and detailed reports makes the system highly useful for understanding pollution distribution and identifying critical areas. The addition of SHAP analysis enhances model interpretability, making the solution more reliable for decision-making.

This project demonstrates how data-driven approaches can be effectively used for environmental monitoring and can be further extended for real-time data integration, advanced analytics, and smart city applications.


**Future Scope**

• Real-time pollution monitoring
• Integration with IoT pollution sensors
• Expansion to multiple cities
• Satellite data integration
• Mobile and web applications
• Deep learning models for improved prediction
