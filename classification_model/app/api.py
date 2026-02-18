# app/api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Iris Prediction API")

MODEL_PATH = os.path.join(os.getcwd(), "model", "iris_pipeline.pkl")

model = joblib.load(MODEL_PATH)

SPECIES = ["Setosa", "Versicolor", "Virginica"]

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisOutput(BaseModel):
    prediction: str

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/v1/predict", response_model=IrisOutput)
def predict(data: IrisInput):
    features = np.array([
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]).reshape(1, -1)

    pred_idx = model.predict(features)[0]
    pred_label = SPECIES[pred_idx]

    return {"prediction": pred_label}

