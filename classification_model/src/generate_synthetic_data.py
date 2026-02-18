import pandas as pd
import numpy as np
import os
DATA_PATH = os.path.join(os.getcwd(), "data")

ref = pd.read_csv(DATA_PATH+"/iris.csv")
print(ref.head())


new = ref.sample(30).copy()

# Inject small drift
new["sepal length (cm)"] += np.random.normal(0.5, 0.2, size=len(new))
new["sepal width (cm)"]  += np.random.normal(0.2, 0.1, size=len(new))

new.to_csv(DATA_PATH+"/new_data.csv", index=False)
print("new_data.csv generated")
