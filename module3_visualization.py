import pandas as pd
import matplotlib.pyplot as plt

print(" Running Module 3 Visualization...")

# Load labeled data
df = pd.read_csv("data/labeled_data.csv")

# Calculate counts
counts = df['pollution_source'].value_counts()

# Plot
plt.figure()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title("Pollution Source Distribution")

# Save graph (IMPORTANT)
plt.savefig("data/module3_pie_chart.png")

print(" Graph saved in data/module3_pie_chart.png")

