
import os
import pickle

import pandas as pd
from fastapi import FastAPI

from api.schemas import (
    DeliveryPredictionInput,
    DeliveryPredictionResponse,
)

app = FastAPI(
    title="Swiggy Delivery Time Prediction API",
    description="Predict food delivery time using Machine Learning.",
    version="1.0.0",
)

# Get project root directory
ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Model path
MODEL_PATH = os.path.join(ROOT_DIR, "model.pkl")

# Load trained pipeline
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


@app.get("/")
def home():
    return {
        "project": "Swiggy Delivery Time Prediction",
        "status": "API is running successfully",
        "docs": "/docs",
        "prediction_endpoint": "/predict"
    }


@app.post( "/predict", response_model=DeliveryPredictionResponse,)
def predict(user_input: DeliveryPredictionInput):

    # Convert input to dictionary
    input_data = user_input.model_dump()

    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([input_data])

    # Predict delivery time
    prediction = model.predict(input_df)[0]

    return {
        "predicted_time_taken": round(float(prediction), 2)
    }
