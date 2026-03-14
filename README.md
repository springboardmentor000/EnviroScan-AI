EnviroScan AI-Powered Pollution Source Identifier using Geospatial Analytics

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
| Total Records  | 1914                           |
| Total Features | 33+                            |
| Locations      | HB Colony, Kanuru (Vijayawada) |



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

**Programming Language**

Python

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
| Gradient Boosting | 96.64%            | 93.16%           | 0.939                     |



**Important predictors identified:**

PM2.5
PM10
NO₂
CO
O₃


**Results**

The EnviroScan system successfully analyzed environmental data and predicted pollution sources using multiple machine learning models. The dataset was trained using pollutant concentrations, weather parameters, and geographic features to classify pollution sources.
Several models were evaluated using accuracy, precision, recall, F1-score, and confusion matrix to measure their performance.
• Gradient Boosting achieved the highest testing accuracy of 93.16%, showing strong predictive capability.
• XGBoost produced a high cross-validation F1-score of 0.942, indicating consistent performance across folds.
• Decision Tree provided good interpretability with clear decision rules for pollution source classification.
• Random Forest demonstrated strong training accuracy due to its ensemble learning structure.
The results demonstrate that combining pollution data, weather parameters, and geospatial features allows machine learning models to effectively identify pollution sources.Among all models, Gradient Boosting and XGBoost showed the best balance between accuracy and generalization, making them suitable for integration into the EnviroScan prediction dashboard.


**Future Scope**

• Real-time pollution monitoring
• Integration with IoT pollution sensors
• Expansion to multiple cities
• Satellite data integration
• Mobile and web applications
• Deep learning models for improved prediction
