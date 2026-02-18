import pandas as pd
import numpy as np

REFERENCE_PATH = "data/iris.csv"
NEW_DATA_PATH = "data/new_data.csv"

THRESHOLD = 0.2  # 20% drift

def detect_drift():
    ref = pd.read_csv(REFERENCE_PATH)
    new = pd.read_csv(NEW_DATA_PATH)

    drift_scores = []

    for col in ref.columns[:-1]:
        ref_mean = ref[col].mean()
        new_mean = new[col].mean()
        drift = abs(ref_mean - new_mean) / ref_mean
        drift_scores.append(drift)

    avg_drift = np.mean(drift_scores)
    print("Drift:", avg_drift)

    if avg_drift > THRESHOLD:
        print("Retraining needed")
        return True
    else:
        print("Model healthy")
        return False

if __name__ == "__main__":
    detect_drift()
