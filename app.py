"""
CI/CD Pipeline Failure Prediction - FastAPI Application
RESTful API for real-time predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Initialize FastAPI app
app = FastAPI(
    title="CI/CD Failure Prediction API",
    description="Predicts pipeline failure probability using ML",
    version="1.0.0"
)

# Load model
MODEL_PATH = 'models/best_model.pkl'

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Feature names (must match training)
FEATURES = [
    'hour', 'day_of_week', 'month',
    'stage_encoded', 'job_encoded', 
    'task_encoded', 'environment_encoded'
]

# Pydantic models for request/response
class PipelineInput(BaseModel):
    """Input schema for prediction"""
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    stage_encoded: int = Field(..., ge=0, le=3, description="Pipeline stage (0-3)")
    job_encoded: int = Field(..., ge=0, le=4, description="Job type (0-4)")
    task_encoded: int = Field(..., ge=0, le=4, description="Task type (0-4)")
    environment_encoded: int = Field(..., ge=0, le=3, description="Environment (0-3)")
    
    class Config:
        schema_extra = {
            "example": {
                "hour": 14,
                "day_of_week": 2,
                "month": 11,
                "stage_encoded": 1,
                "job_encoded": 2,
                "task_encoded": 1,
                "environment_encoded": 1
            }
        }

class PredictionOutput(BaseModel):
    """Output schema for prediction"""
    prediction: str
    failure_probability: float
    success_probability: float
    confidence: str
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str

# API Endpoints

@app.get("/", response_class=HTMLResponse)
def root():
    """Root endpoint with API documentation"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CI/CD Failure Prediction API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #2c3e50; }
            .endpoint { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
            code { background: #34495e; color: #ecf0f1; padding: 2px 6px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>🚀 CI/CD Failure Prediction API</h1>
        <p>Machine Learning API for predicting CI/CD pipeline failures</p>
        
        <h2>Available Endpoints:</h2>
        
        <div class="endpoint">
            <h3>GET /health</h3>
            <p>Check API health status</p>
        </div>
        
        <div class="endpoint">
            <h3>POST /predict</h3>
            <p>Predict pipeline failure probability</p>
            <p>Required fields: hour, day_of_week, month, stage_encoded, job_encoded, task_encoded, environment_encoded</p>
        </div>
        
        <div class="endpoint">
            <h3>GET /docs</h3>
            <p>Interactive API documentation (Swagger UI)</p>
        </div>
        
        <div class="endpoint">
            <h3>GET /model-info</h3>
            <p>Get model metadata and feature information</p>
        </div>
        
        <p><a href="/docs">📚 Try the API interactively</a></p>
    </body>
    </html>
    """
    return html_content

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model-info")
def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "encoding_info": {
            "stage_encoded": "0=Analysis, 1=Build, 2=Deploy, 3=Test",
            "job_encoded": "0=build_and_test, 1=deploy_to_dev, 2=deploy_to_staging, 3=run_integration_tests, 4=run_unit_tests",
            "task_encoded": "0=analyze, 1=build, 2=deploy, 3=lint, 4=test",
            "environment_encoded": "0=development, 1=not_specified, 2=production, 3=staging"
        }
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PipelineInput):
    """
    Predict pipeline failure probability
    
    Returns:
    - prediction: 'failure' or 'success'
    - failure_probability: Probability of failure (0-1)
    - success_probability: Probability of success (0-1)
    - confidence: 'high', 'medium', or 'low'
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_dict = data.dict()
        features_df = pd.DataFrame([input_dict])
        
        # Ensure correct feature order
        features_df = features_df[FEATURES]
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        
        # Extract probabilities
        success_prob = float(probabilities[0])
        failure_prob = float(probabilities[1])
        
        # Determine confidence level
        max_prob = max(success_prob, failure_prob)
        if max_prob >= 0.8:
            confidence = "high"
        elif max_prob >= 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Prepare response
        result = {
            "prediction": "failure" if prediction == 1 else "success",
            "failure_probability": round(failure_prob, 4),
            "success_probability": round(success_prob, 4),
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict")
def batch_predict(data: list[PipelineInput]):
    """
    Batch prediction endpoint for multiple pipelines
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(data) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 predictions per batch")
    
    try:
        results = []
        
        for item in data:
            input_dict = item.dict()
            features_df = pd.DataFrame([input_dict])
            features_df = features_df[FEATURES]
            
            prediction = model.predict(features_df)[0]
            probabilities = model.predict_proba(features_df)[0]
            
            results.append({
                "prediction": "failure" if prediction == 1 else "success",
                "failure_probability": round(float(probabilities[1]), 4),
                "success_probability": round(float(probabilities[0]), 4)
            })
        
        return {
            "count": len(results),
            "predictions": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on API startup"""
    print("="*70)
    print("CI/CD FAILURE PREDICTION API STARTING")
    print("="*70)
    print(f"Model loaded: {model is not None}")
    print(f"Features: {FEATURES}")
    print("API ready at http://localhost:8000")
    print("Docs available at http://localhost:8000/docs")
    print("="*70)

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )