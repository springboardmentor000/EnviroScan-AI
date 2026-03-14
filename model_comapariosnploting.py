import matplotlib.pyplot as plt
import numpy as np
import os

# Create plots folder
os.makedirs("plots", exist_ok=True)

models = ["Random Forest","Decision Tree", "XGBoost", "Gradient Boosting"]

# ---------------------------
# Accuracy values
# ---------------------------

train_before = [95.4, 100.0, 94.74, 99.78]
test_before = [83.03, 88.25, 89.56, 96.89]

train_after = [99.1, 96.80, 97.54, 96.64]
test_after = [88.51, 93.45, 91.64, 93.16]


x = np.arange(len(models))
width = 0.13

plt.figure(figsize=(14,7))

bars1 = plt.bar(x - 2.5*width, train_before, width, label="Train Before")
bars2 = plt.bar(x - 1.5*width, test_before, width, label="Test Before")

bars3 = plt.bar(x + 0.5*width, train_after, width, label="Train After")
bars4 = plt.bar(x + 1.5*width, test_after, width, label="Test After")

plt.title("Model Performance Comparison (Before vs After Tuning)", fontsize=15)
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")

plt.xticks(x, models)
plt.ylim(80,101)

plt.legend(ncol=3)

# ---------------------------
# Add accuracy labels
# ---------------------------

def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                 height + 0.2,
                 f"{height:.2f}",
                 ha='center',
                 va='bottom',
                 fontsize=8)

for b in [bars1,bars2,bars3,bars4]:
    add_labels(b)

plt.tight_layout()

plt.savefig("plots/model_accuracy_full_comparison.png")

plt.show()

print("Graph saved in plots/model_accuracy_full_comparison.png")