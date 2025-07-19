from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Step 1: Initialize FastAPI app
app = FastAPI()

# Step 2: Load the trained ML model
model = joblib.load("rain_predictor_model.pkl")

# Step 3: Create input model for API request
class WeatherInput(BaseModel):
    Temperature: float
    Humidity: float
    WindSpeed: float
    CloudCover: float

# Step 4: Root endpoint to check if API is working
@app.get("/")
def read_root():
    return {"message": "API is working!"}

# Step 5: Prediction endpoint
@app.post("/predict")
def predict_rain(data: WeatherInput):
    # Convert input into model format
    input_data = np.array([[data.Temperature, data.Humidity, data.WindSpeed, data.CloudCover]])
    
    # Predict using loaded model
    prediction = model.predict(input_data)[0]
    
    # Return result
    return {"WillRain": bool(prediction)}
