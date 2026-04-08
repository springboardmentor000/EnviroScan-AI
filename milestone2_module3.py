import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os

print("=" * 60)
print("  EnviroScan — Module 3: Source Labeling & Simulation")
print("=" * 60)

os.makedirs("outputs", exist_ok=True)
random.seed(42)
np.random.seed(42)

# -------------------------------------------------
# STEP 1: Load cleaned data
# -------------------------------------------------
print("\n STEP 1: Loading cleaned data...")
df = pd.read_csv("Hyderabad_pollution_cleaned.csv")
print(f"   Rows: {len(df)}  |  Columns: {len(df.columns)}")

# -------------------------------------------------
# STEP 2: Rule-based source labeling
# -------------------------------------------------
print("\n  STEP 2: Applying rule-based source labeling...")

def label_source(row):
    no2   = row.get("no2",  0) or 0
    so2   = row.get("so2",  0) or 0
    co    = row.get("co",   0) or 0
    pm25  = row.get("pm25", 0) or 0
    pm10  = row.get("pm10", 0) or 0
    o3    = row.get("o3",   0) or 0

    roads      = row.get("road_count", 0) or 0
    industrial = row.get("industrial_zone_count", 0) or 0
    dumps      = row.get("dump_site_count", 0) or 0
    agri       = row.get("agricultural_count", 0) or 0
    name       = str(row.get("station_name", "")).lower()

    scores = {
        "Vehicular":    0,
        "Industrial":   0,
        "Agricultural": 0,
        "Burning":      0,
        "Natural":      0,
    }

    # ── Vehicular signals ──
    if no2 > 40:   scores["Vehicular"] += 3
    if no2 > 20:   scores["Vehicular"] += 1
    if co > 1.0:   scores["Vehicular"] += 2
    if roads > 200: scores["Vehicular"] += 3
    if roads > 50:  scores["Vehicular"] += 1
    if pm25 > 50 and no2 > 20: scores["Vehicular"] += 2
    if any(x in name for x in ["zoo", "park", "malakpet", "somajiguda", "diplomatic"]):
        scores["Vehicular"] += 2

    # ── Industrial signals ──
    if so2 > 10:        scores["Industrial"] += 3
    if industrial > 10: scores["Industrial"] += 3
    if industrial > 5:  scores["Industrial"] += 2
    if pm25 > 80 and so2 > 5: scores["Industrial"] += 2
    if any(x in name for x in ["industrial", "iala", "tsiic", "bollaram", "nacharam", "sanathnagar"]):
        scores["Industrial"] += 4
    if dumps > 0: scores["Industrial"] += 1

    # ── Agricultural signals ──
    if agri > 10:  scores["Agricultural"] += 3
    if agri > 5:   scores["Agricultural"] += 2
    if pm10 > 100 and no2 < 20: scores["Agricultural"] += 2
    if any(x in name for x in ["kompally", "kokapet"]):
        scores["Agricultural"] += 2

    # ── Burning signals ──
    if co > 500:   scores["Burning"] += 4
    if co > 200:   scores["Burning"] += 2
    if pm25 > 100 and co > 100: scores["Burning"] += 3
    if dumps > 2:  scores["Burning"] += 2

    # ── Natural signals ──
    if pm25 < 30 and no2 < 15 and so2 < 5: scores["Natural"] += 3
    if o3 > 100:   scores["Natural"] += 2
    if roads < 10 and industrial < 3:       scores["Natural"] += 2
    if any(x in name for x in ["university", "ecil", "kapra"]):
        scores["Natural"] += 1

    predicted  = max(scores, key=scores.get)
    total      = sum(scores.values())
    confidence = round(scores[predicted] / total, 2) if total > 0 else 0.0

    return predicted, confidence

print("\n   Labeling each station:\n")
sources, confidences = [], []

for _, row in df.iterrows():
    source, conf = label_source(row)
    sources.append(source)
    confidences.append(conf)
    print(f"   {row['station_name'][:45]:<45} → {source:<15} (confidence: {conf})")

df["pollution_source"]  = sources
df["source_confidence"] = confidences

# -------------------------------------------------
# STEP 3: Simulate additional training data
# -------------------------------------------------
print("\n STEP 3: Simulating additional training data...")

PROFILES = {
    "Industrial":    {"pm25":(60,150),  "pm10":(100,250), "no2":(15,50),  "so2":(20,80),  "co":(0.5,2.0),   "o3":(20,50),  "roads":(5,30),   "ind":(10,40), "dumps":(0,3), "agri":(0,2)},
    "Vehicular":     {"pm25":(40,100),  "pm10":(70,180),  "no2":(60,150), "so2":(5,20),   "co":(1.5,5.0),   "o3":(30,70),  "roads":(200,900),"ind":(0,10),  "dumps":(0,1), "agri":(0,2)},
    "Agricultural":  {"pm25":(20,60),   "pm10":(50,150),  "no2":(5,20),   "so2":(2,10),   "co":(0.1,0.5),   "o3":(10,40),  "roads":(5,50),   "ind":(0,3),   "dumps":(0,1), "agri":(10,30)},
    "Burning":       {"pm25":(100,300), "pm10":(200,500), "no2":(20,60),  "so2":(10,40),  "co":(300,1500),  "o3":(15,45),  "roads":(10,100), "ind":(0,5),   "dumps":(2,8), "agri":(0,5)},
    "Natural":       {"pm25":(5,30),    "pm10":(10,60),   "no2":(2,15),   "so2":(1,8),    "co":(0.0,0.3),   "o3":(50,120), "roads":(0,20),   "ind":(0,3),   "dumps":(0,1), "agri":(0,5)},
}

sim_records = []
for source, p in PROFILES.items():
    for i in range(40):
        rec = {
            "station_id":             f"SIM_{source[:3].upper()}_{i:03d}",
            "station_name":           f"Simulated_{source}_{i}",
            "latitude":               round(17.3850 + random.uniform(-0.15, 0.15), 6),
            "longitude":              round(78.4867 + random.uniform(-0.15, 0.15), 6),
            "timestamp":              "2026-03-10 12:00:00",
            "pm25":                   round(random.uniform(*p["pm25"]), 2),
            "pm10":                   round(random.uniform(*p["pm10"]), 2),
            "no2":                    round(random.uniform(*p["no2"]),  2),
            "so2":                    round(random.uniform(*p["so2"]),  2),
            "o3":                     round(random.uniform(*p["o3"]),   2),
            "co":                     round(random.uniform(*p["co"]),   3),
            "temp":                   round(random.uniform(22, 38), 1),
            "humidity":               random.randint(25, 80),
            "pressure":               1014,
            "wind_speed":             round(random.uniform(1, 8), 1),
            "wind_direction":         random.randint(0, 359),
            "road_count":             random.randint(*p["roads"]),
            "has_major_road":         1 if random.randint(*p["roads"]) > 50 else 0,
            "industrial_zone_count":  random.randint(*p["ind"]),
            "near_industrial_zone":   1 if random.randint(*p["ind"]) > 5 else 0,
            "dump_site_count":        random.randint(*p["dumps"]),
            "near_dump_site":         1 if random.randint(*p["dumps"]) > 1 else 0,
            "agricultural_count":     random.randint(*p["agri"]),
            "near_agricultural_area": 1 if random.randint(*p["agri"]) > 5 else 0,
            "hour":                   12,
            "day_of_week":            1,
            "month":                  3,
            "season":                 "Summer",
            "pollution_source":       source,
            "source_confidence":      1.0,
        }
        sim_records.append(rec)

sim_df = pd.DataFrame(sim_records)
print(f"    Simulated {len(sim_df)} records ({len(PROFILES)} sources × 40 samples)")

# -------------------------------------------------
# STEP 4: Combine real + simulated data
# -------------------------------------------------
print("\n🔗 STEP 4: Combining real + simulated data...")

combined = pd.concat([df, sim_df], ignore_index=True)

# Fill any remaining NaN in key feature columns with median
feature_cols = ["pm25","pm10","no2","so2","o3","co",
                "road_count","industrial_zone_count","dump_site_count","agricultural_count"]
for col in feature_cols:
    combined[col] = combined[col].fillna(combined[col].median())

print(f"   Real data rows     : {len(df)}")
print(f"   Simulated rows     : {len(sim_df)}")
print(f"   Combined total     : {len(combined)}")

# -------------------------------------------------
# STEP 5: Validate label distribution
# -------------------------------------------------
print("\n STEP 5: Label distribution in combined dataset...")
dist = combined["pollution_source"].value_counts()
for source, count in dist.items():
    bar = "█" * count
    print(f"   {source:<15}: {count:>3}  {bar}")

# -------------------------------------------------
# STEP 6: Save labeled dataset
# -------------------------------------------------
combined.to_csv("Hyderabad_pollution_labeled.csv", index=False)
print(f"\n Saved → Hyderabad_pollution_labeled.csv")

# -------------------------------------------------
# STEP 7: Visualizations
# -------------------------------------------------
print("\n STEP 7: Generating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("EnviroScan — Module 3: Source Label Distribution", fontsize=14, fontweight="bold")

# Pie chart
colors = ["#E74C3C","#3498DB","#2ECC71","#F39C12","#9B59B6"]
axes[0].pie(dist.values, labels=dist.index, autopct="%1.1f%%",
            colors=colors, startangle=140, textprops={"fontsize": 11})
axes[0].set_title("Source Distribution (Combined Dataset)")

# Bar chart — real data only
real_dist = df["pollution_source"].value_counts()
axes[1].bar(real_dist.index, real_dist.values, color=colors[:len(real_dist)], edgecolor="white")
axes[1].set_title("Source Labels — Real Hyderabad Stations")
axes[1].set_xlabel("Pollution Source")
axes[1].set_ylabel("Number of Stations")
axes[1].tick_params(axis="x", rotation=15)
for i, v in enumerate(real_dist.values):
    axes[1].text(i, v + 0.05, str(v), ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/module3_label_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved → outputs/module3_label_distribution.png")

# -------------------------------------------------
# STEP 8: Print final labeled real stations
# -------------------------------------------------
print("\n STEP 8: Final labels for real Hyderabad stations:\n")
print(f"   {'Station':<45} {'Source':<15} {'Confidence'}")
print(f"   {'-'*45} {'-'*15} {'-'*10}")
for _, row in df.iterrows():
    print(f"   {row['station_name'][:45]:<45} {row['pollution_source']:<15} {row['source_confidence']}")

print(f"\n{'='*60}")
print(" Module 3 Complete!")
print("   Output: Hyderabad_pollution_labeled.csv")
print("   Chart:  outputs/module3_label_distribution.png")
print(f"{'='*60}")