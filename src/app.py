from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn import datasets

# Load trained model once
model = joblib.load("models/best_model.pkl")
iris = datasets.load_iris()
class_names = iris.target_names

# FastAPI app
app = FastAPI(title="Iris Classifier API")

# Request schema
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "Welcome to Iris Classifier API 🌸"}

@app.post("/predict")
def predict(features: IrisFeatures):
    sample = np.array([[features.sepal_length,
                        features.sepal_width,
                        features.petal_length,
                        features.petal_width]])
    prediction = model.predict(sample)[0]
    return {"prediction": class_names[prediction]}