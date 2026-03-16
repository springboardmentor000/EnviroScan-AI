import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("Starting Final Data Processing...")

# Load dataset
df = pd.read_csv("module3_labeled_dataset.csv")

# ---------------------------------------------------
# 1 Remove duplicates
# ---------------------------------------------------

df = df.drop_duplicates()

# ---------------------------------------------------
# 2 Select required columns only
# ---------------------------------------------------

columns_needed = [
'city','latitude','longitude',
'temperature_C','humidity_%','wind_speed_mps',
'PM2_5','PM10','NO2','CO','SO2','O3',
'hour','day_of_week','month','season',
'dist_to_road','dist_to_industry','dist_to_farmland',
'pollution_source'
]

df = df[columns_needed]

# ---------------------------------------------------
# 3 Handle missing values
# ---------------------------------------------------

df.fillna(df.median(numeric_only=True), inplace=True)

# ---------------------------------------------------
# 4 Encode categorical variables
# ---------------------------------------------------

le = LabelEncoder()
df['city'] = le.fit_transform(df['city'])
df['season'] = le.fit_transform(df['season'])
df['pollution_source'] = le.fit_transform(df['pollution_source'])

# ---------------------------------------------------
# 5 Prepare ML data
# ---------------------------------------------------

X = df.drop("pollution_source", axis=1)
y = df["pollution_source"]

# ---------------------------------------------------
# 6 Train model
# ---------------------------------------------------

model = RandomForestClassifier(n_estimators=100)

# ---------------------------------------------------
# 7 Cross Validation
# ---------------------------------------------------

scores = cross_val_score(model, X, y, cv=5)

print("Cross Validation Scores:", scores)
print("Average Accuracy:", scores.mean())

# ---------------------------------------------------
# 8 Visualization
# ---------------------------------------------------

plt.figure()
sns.countplot(x=df["pollution_source"])
plt.title("Pollution Source Distribution")
plt.show()

plt.figure()
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

plt.figure()
plt.scatter(df["PM2_5"], df["NO2"])
plt.xlabel("PM2.5")
plt.ylabel("NO2")
plt.title("Pollution Relationship")
plt.show()

print("Pipeline Completed Successfully")