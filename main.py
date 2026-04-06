import subprocess
import sys

print(" Starting AI EnviroScan Pipeline...\n")

# =========================================
#  MODULES TO RUN 
# =========================================
steps = [

    ("Module 2: Data Cleaning", "src/data_cleaning.py"),
    ("Module 2: Visualization", "src/module2_visualization.py"),  
    
    ("Module 3: Source Labeling", "src/module3_labeling.py"),
    ("Module 3: Visualization", "src/module3_visualization.py"),
    ("Module 4: Model Training", "src/module4_model_training.py"),
    ("Module 4: Visualization", "src/module4_visualization.py"),
    ("Module 5: Mapping", "src/module5_mapping.py"),

]

# =========================================
#  EXECUTE MODULES
# =========================================
for name, script in steps:
    print(f"▶ Running {name}...\n")

    try:
        subprocess.run([sys.executable, script], check=True)
        print(f" {name} completed successfully\n")

    except subprocess.CalledProcessError:
        print(f" Error in {name}. Stopping execution.\n")
        break

# =========================================
#  FINAL MESSAGE
# =========================================
print(" Pipeline Execution Completed")
print(" Modules Completed: 1, 2, 3, 4, 5")
print(" Output Files Saved in /data folder")