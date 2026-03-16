# EnviroScan – AI Powered Pollution Source Identifier

## Project Overview

EnviroScan is a machine learning based environmental monitoring system designed to **identify the probable source of air pollution** using pollutant concentrations, weather parameters, and spatial proximity features.

The system integrates **real-time environmental datasets**, performs **data preprocessing and feature engineering**, and trains machine learning models to predict pollution sources such as vehicular emissions, industrial activity, agricultural burning, or natural sources.

The goal of this project is to support **smart city monitoring systems** and **data-driven environmental decision making**.

---

# Objectives

• Collect real-world environmental data from multiple APIs
• Engineer spatial and temporal features that influence pollution
• Label pollution sources using domain-based heuristics
• Train machine learning models for pollution source prediction
• Evaluate model performance using standard ML metrics
• Deploy an interactive dashboard for prediction visualization

---

# System Architecture

Data Collection → Data Cleaning → Feature Engineering → Source Labeling → Model Training → Prediction Dashboard

---

# Milestone 1 – Data Collection & Feature Engineering

## Module 1: Data Collection from APIs

Environmental data was collected from multiple sources:

### Air Quality Data

Collected using the **OpenAQ API**, including the following pollutants:

• PM2.5
• PM10
• NO2
• CO
• SO2
• O3

### Weather Data

Collected using **OpenWeatherMap API**, including:

• Temperature
• Humidity
• Wind Speed
• Wind Direction

### Geographic Features

Nearby environmental features were extracted using **OpenStreetMap via OSMnx**, including:

• Roads
• Industrial zones
• Dump sites
• Agricultural fields

Each data point was tagged with:

• Latitude
• Longitude
• Timestamp
• Source API metadata

All collected data was stored in structured **CSV format** for preprocessing.

---

# Module 2 – Data Cleaning and Feature Engineering

The raw datasets were cleaned and processed to create a feature-rich dataset suitable for machine learning.

## Data Cleaning

• Removed duplicate records
• Removed invalid or corrupted entries
• Handled missing values using median imputation
• Standardized timestamps and geographic coordinates

## Feature Engineering

Several important features were derived:

### Temporal Features

• Hour of day
• Day of week
• Month
• Season

These features help capture pollution patterns related to human activity and climate.

### Spatial Features

Distances to environmental structures were calculated:

• Distance to nearest road
• Distance to nearest industry
• Distance to farmland

These features help determine potential pollution sources.

### Data Normalization

Pollutant concentrations and weather variables were standardized to maintain consistent model input ranges.

The cleaned datasets were merged into a unified **feature-rich dataset**.

---

# Milestone 2 – Source Labeling & Machine Learning

## Module 3 – Pollution Source Labeling

Since real ground-truth labels are not always available, heuristic rules were used to simulate pollution source labels.

### Labeling Rules

Vehicular Pollution
If distance to road is small and NO2 concentration is high.

Industrial Pollution
If distance to industry is small and SO2 concentration is high.

Agricultural Pollution
If farmland is nearby during dry seasons and particulate matter levels are high.

Burning
If CO and PM2.5 levels are simultaneously high.

Natural Sources
If no specific pollution source conditions are satisfied.

Using these rules, each record was labeled as:

• Vehicular
• Industrial
• Agricultural
• Burning
• Natural

This created the **final labeled dataset for model training**.

---

# Module 4 – Model Training and Source Prediction

## Dataset Split

The dataset was divided into:

• 80% training data
• 20% testing data

## Machine Learning Models Used

Three classification models were trained:

• Random Forest Classifier
• Decision Tree Classifier
• XGBoost Classifier

These models use:

• pollutant concentrations
• weather conditions
• proximity features

to predict the pollution source.

---

## Hyperparameter Tuning

Random Forest hyperparameters were optimized using **GridSearchCV** to improve model performance.

Parameters tuned included:

• number of trees (n_estimators)
• maximum tree depth
• minimum samples split

---

## Model Evaluation

Model performance was evaluated using standard machine learning metrics:

• Accuracy
• Precision
• Recall
• F1 Score
• Confusion Matrix

Cross-validation was also used to validate the model's generalization performance.

---

## Model Export

The final trained model was exported using **Joblib** for deployment.

Saved file:

pollution_source_model.pkl

This model is later used by the prediction dashboard.

---

# Prediction Dashboard

An interactive **Streamlit dashboard** was developed to allow users to predict pollution sources based on environmental inputs.

Users can input:

• pollutant levels
• weather conditions
• proximity distances

The trained machine learning model then predicts the most likely pollution source.

This enables an intuitive interface for environmental monitoring and analysis.

---

# Technologies Used

Programming Language
Python

Libraries

• Pandas
• NumPy
• Scikit-learn
• XGBoost
• Matplotlib
• Seaborn
• Joblib
• Streamlit

Data Sources

• OpenAQ API
• OpenWeatherMap API
• OpenStreetMap (OSMnx)

---

# Project Structure

project/

module1_data_collection.py
module2_feature_engineering.py
module3_source_labeling.py
module4_model_training.py
pipeline_analysis.py
dashboard.py

datasets/
module3_labeled_dataset.csv

models/
pollution_source_model.pkl

---

# Future Improvements

• Integration with real-time air quality monitoring systems
• Real-time pollution prediction using live API data
• Geospatial pollution visualization using interactive maps
• Integration with smart city environmental monitoring platforms

---

# Conclusion

EnviroScan demonstrates how machine learning can be used to **identify and analyze pollution sources using environmental data**.

By integrating API-based data collection, feature engineering, heuristic labeling, and predictive modeling, the system provides an intelligent framework for **environmental monitoring and pollution source identification**.

This project highlights the potential of **AI-driven environmental analytics** in supporting sustainable urban development and public health initiatives.
