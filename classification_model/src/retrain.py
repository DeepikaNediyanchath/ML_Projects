# src/retrain.py
print("Checking drift...")
os.system("python src/monitoring.py")

if drift_detected:
    print("Retraining...")
    os.system("python src/training.py")
