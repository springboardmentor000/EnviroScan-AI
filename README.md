#  EnviroScan – Air Pollution Monitoring System

## About Project
This project helps to:
- Collect pollution data  
- Clean and process data  
- Show graphs and maps  
- Predict pollution sources  

Simple meaning:  
This system analyzes air pollution and shows results visually
## Module 1: Data Collection
- Collects air pollution data (PM2.5, NO2, SO2, O3)  
- Collects weather data (temperature, humidity, wind speed)  
- Adds location (latitude, longitude)  

Output files:
- air_quality.csv  
- weather.csv  
- distance_data.csv  
##  Module 2: Data Cleaning
- Removes missing values  
- Fills empty data  
- Makes data clean  

Output files:
- cleaned_air_quality.csv  
- cleaned_weather.csv  
- cleaned_distance.csv  
##  Module 3: Feature Engineering
- Combines all datasets  
- Adds useful columns  
- Prepares final dataset  

Output file:
- final_dataset.csv  
##  Module 4: Prediction
- Uses if-else logic  
Example:
- High NO2 → Vehicular  
- High SO2 → Industrial  

### Machine Learning Method
- Uses Random Forest model  
- Learns from data  

Output files:
- labeled_dataset.csv  
- model.pkl  
##  Module 5: Map
- Shows pollution on map  
- Uses latitude and longitude  

Helps to:
- Identify polluted areas  
- Compare locations  

Output file:
- pollution_map.html  
## Module 6: Excel Export
- Converts dataset into Excel  

Output file:
- final_dataset.xlsx  
##  Dashboard

Built using Streamlit.

### Features:
- Select city  
- Select pollutant  
- View average values  

### Tabs:
1. Overview  
- Line graph → pollution over time  
- Pie chart → pollution sources  

2. Graphs  
- Scatter plot → relation between values  
- Heatmap → correlation  

3. Map  
- Shows pollution locations  

4. Prediction  
- Predicts pollution source  

- Model: Random Forest  
- Input: PM2.5, NO2, SO2, O3  
- Output: Pollution source  

Simple:  
Model predicts source based on pollution values.
## Tools Used
- Python  
- Pandas  
- Scikit-learn  
- Streamlit  
- Plotly  
- Folium  
## ▶ How to Run
Run project:
python main.py

Train model:
python scripts/mod4_model_training.py

Run dashboard:
python -m streamlit run dashboard.py
## 📁 Project Structure
Pollution_AI/
│
├── data/
├── scripts/
├── dashboard.py
├── main.py
├── model.pkl

## 🎯 Final Output
- Clean data  
- Graphs  
- Map  
- Dashboard  
- Prediction  
## Easy Explanation

This project collects pollution data, cleans it, shows graphs and maps, and predicts pollution sources using machine learning.

##  Future Improvements
- Real-time data  
- Better ML models  
- Mobile app  
- web app