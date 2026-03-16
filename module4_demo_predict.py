

import pandas as pd
import joblib

print("=== Pollution Source Prediction Demo ===\n")

# =========================
# Step 1: Load Trained Model
# =========================
try:
    model = joblib.load("pollution_source_model.pkl")
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("❌ Error: pollution_source_model.pkl not found. Make sure it exists in the folder.")
    exit()

# =========================
# Step 2: Prepare New Data for Prediction
# =========================
# Example: Replace values with your location's data
new_data = pd.DataFrame([{
    'PM2_5': 60,
    'PM10': 80,
    'NO2': 45,
    'CO': 0.8,
    'SO2': 12,
    'O3': 30,
    'temperature_C': 32,
    'humidity_%': 40,
    'wind_speed_mps': 2,
    'dist_to_road': 50,
    'dist_to_industry': 200,
    'dist_to_farmland': 300
}])

print("✅ New data prepared for prediction\n", new_data)

# =========================
# Step 3: Make Prediction
# =========================
predicted_source = model.predict(new_data)

print("\n✅ Predicted Pollution Source:", predicted_source[0])
print("\nDemo Completed Successfully!")