from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib  
import random
from datetime import datetime
import os
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Suppress scikit-learn version warnings
warnings.filterwarnings('ignore', category=UserWarning)

app = FastAPI()

# Allow dashboard frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model and scaler
model = None
scaler = None

# Default feature ranges for scaling if model loading fails
feature_ranges = {
    'turbidity': (50, 150),
    'tds': (500, 1500),
    'flow1': (150, 250),
    'flow2': (150, 250)
}

# Load model and scaler with error handling
try:
    model = joblib.load("rf_clogging_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("Model and scaler loaded successfully")
except Exception as e:
    print(f"Error loading model or scaler: {str(e)}")
    # Create dummy model and scaler with sample data
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    scaler = StandardScaler()
    
    # Create sample data for fitting
    sample_data = np.array([
        [100, 1000, 200, 200],  # Average values
        [50, 500, 150, 150],    # Min values
        [150, 1500, 250, 250]   # Max values
    ])
    
    # Fit the scaler with sample data
    scaler.fit(sample_data)
    
    # Fit the model with sample data (predicting random days between 1-10)
    model.fit(sample_data, np.random.uniform(1, 10, size=3))
    print("Using dummy model and scaler with sample data")

# Set alert threshold
ALERT_THRESHOLD = 5  # days

# Define the request body schema
class SensorData(BaseModel):
    turbidity: float
    tds: float
    flow1: float
    flow2: float

    class Config:
        schema_extra = {
            "example": {
                "turbidity": 100.0,
                "tds": 1000.0,
                "flow1": 200.0,
                "flow2": 200.0
            }
        }

@app.get("/sensor-data")
def get_sensor_data():
    try:
        # Read the latest data from CSV
        if os.path.exists("yamuna_sensor_data_all_columns.csv"):
            df = pd.read_csv("yamuna_sensor_data_all_columns.csv")
            latest_data = df.iloc[-1]  # Get the latest row
            
            # Format the data
            data = {
                "turbidity": float(latest_data['turbidity']),
                "tds": float(latest_data['tds']),
                "flow1": float(latest_data['flow1']),
                "flow2": float(latest_data['flow2']),
                "timestamp": datetime.now().isoformat()
            }
            return data
    except Exception as e:
        print(f"Error reading sensor data: {str(e)}")
    
    # Fallback to simulated data
    return {
        "turbidity": round(random.uniform(feature_ranges['turbidity'][0], feature_ranges['turbidity'][1]), 2),
        "tds": round(random.uniform(feature_ranges['tds'][0], feature_ranges['tds'][1]), 2),
        "flow1": round(random.uniform(feature_ranges['flow1'][0], feature_ranges['flow1'][1]), 2),
        "flow2": round(random.uniform(feature_ranges['flow2'][0], feature_ranges['flow2'][1]), 2),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
def predict(data: SensorData):
    try:
        # Validate input data
        if any(value < 0 for value in [data.turbidity, data.tds, data.flow1, data.flow2]):
            raise HTTPException(status_code=400, detail="All sensor values must be positive")
        
        # Convert to NumPy array
        input_data = np.array([[data.turbidity, data.tds, data.flow1, data.flow2]])
        
        # Scale the input
        if scaler is not None:
            input_scaled = scaler.transform(input_data)
        else:
            input_scaled = input_data
        
        # Predict clogging time
        predicted_days = max(0.1, model.predict(input_scaled)[0])  # Ensure prediction is positive
        
        # Determine alert status
        alert = predicted_days < ALERT_THRESHOLD

        return {
            "predicted_days": round(predicted_days, 2),
            "alert": alert,
            "status": "⚠️ Clogging Soon" if alert else "✅ Filter OK"
        }
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")
