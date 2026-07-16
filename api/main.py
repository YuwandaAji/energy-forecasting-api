from fastapi import FastAPI, HTTPException
import pandas as pd

from app.schemas import PredictionRequest, PredictionResponse, HourlyPrediction
from app.model_loader import load_all_models
from app.features import get_context_and_future
from app.inference import run_prediction

app = FastAPI(title="Energy Demand Forecasting API", version="1.0")

# Load semua model SEKALI saat startup, bukan tiap request (biar cepat)
models = None

@app.on_event("startup")
def startup_event():
    global models
    print("Loading models...")
    models = load_all_models()
    print("Models loaded successfully.")


@app.get("/")
def root():
    return {"message": "Energy Demand Forecasting API is running", "status": "ok"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": models is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if models is None:
        raise HTTPException(status_code=503, detail="Models belum selesai loading, coba lagi sebentar")
    
    valid_models = ["lstm", "prophet", "chronos2", "ensemble"]
    if request.model not in valid_models:
        raise HTTPException(status_code=400, detail=f"model harus salah satu dari: {valid_models}")
    
    try:
        context_df, future_df, is_future = get_context_and_future(
            request.prediction_start, input_window=168, output_horizon=24
        )
        
        preds = run_prediction(request.model, context_df, future_df, models, output_horizon=24)
        
        timestamps = pd.date_range(request.prediction_start, periods=24, freq='h')
        predictions = [
            HourlyPrediction(timestamp=ts, predicted_demand_mw=float(p))
            for ts, p in zip(timestamps, preds)
        ]
        
        return PredictionResponse(
            model_used=request.model,
            prediction_start=request.prediction_start,
            predictions=predictions,
            is_future=is_future
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi error: {str(e)}")