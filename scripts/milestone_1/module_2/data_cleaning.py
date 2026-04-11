import pandas as pd

# Paths
INPUT = "../data/processed/feature_dataset.csv"
OUTPUT = "../data/processed/cleaned_dataset.csv"

# Load data
df = pd.read_csv(INPUT)

print("Original Shape:", df.shape)

# -----------------------------
# DROP UNNECESSARY COLUMNS
# -----------------------------
cols_to_drop = ["timestamp_x", "timestamp_y", "unit"]

for col in cols_to_drop:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------
# Fill numeric columns with mean
for col in df.select_dtypes(include=["float64", "int64"]).columns:
    df[col].fillna(df[col].mean(), inplace=True)

# -----------------------------
# REMOVE DUPLICATES
# -----------------------------
df.drop_duplicates(inplace=True)

# -----------------------------
# OPTIONAL: DROP NON-NUMERIC (if needed)
# -----------------------------
# Keep city if you want, otherwise drop
# df.drop(columns=["city"], inplace=True)

# -----------------------------
# FINAL CHECK
# -----------------------------
print("Cleaned Shape:", df.shape)

# Save cleaned data
df.to_csv(OUTPUT, index=False)

print("✅ Cleaned dataset saved")