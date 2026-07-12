from pydantic import BaseModel, Field
from datetime import datetime
from typing import List

class PredictionRequest(BaseModel):
    prediction_start: datetime = Field(
        ..., 
        description="Timestamp mulai prediksi (24 jam ke depan dari titik ini)",
        examples=["2026-07-12T00:00:00"]
    )
    model: str = Field(
        default="ensemble",
        description="Model yang dipakai: 'lstm', 'chronos2', 'prophet', atau 'ensemble'"
    )

class HourlyPrediction(BaseModel):
    timestamp: datetime
    predicted_demand_mw: float

class PredictionResponse(BaseModel):
    model_used: str
    prediction_start: datetime
    predictions: List[HourlyPrediction]
    is_future: bool