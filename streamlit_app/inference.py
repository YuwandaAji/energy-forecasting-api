import torch
import numpy as np
import pandas as pd
from chronos import Chronos2Pipeline

COVARIATE_COLS = ['temperature', 'cooling_degree', 'heating_degree',
                   'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
                   'month_sin', 'month_cos', 'is_weekend', 'is_holiday']
FEATURE_COLS = ['adjusted_demand', 'temperature', 'cooling_degree', 'heating_degree',
                'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
                'month_sin', 'month_cos', 'is_weekend', 'is_holiday']
TARGET_IDX = FEATURE_COLS.index('adjusted_demand')

_chronos_pipeline = None

def get_chronos_pipeline():
    global _chronos_pipeline
    if _chronos_pipeline is None:
        _chronos_pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    return _chronos_pipeline


def inverse_transform_target(scaled_values, scaler, target_idx, n_features):
    dummy = np.zeros((len(scaled_values), n_features))
    dummy[:, target_idx] = scaled_values
    return scaler.inverse_transform(dummy)[:, target_idx]


def predict_lstm(context_df, models, output_horizon=24):
    scaler = models['scaler']
    lstm_model = models['lstm_model']
    device = models['device']
    
    context_scaled = scaler.transform(context_df[FEATURE_COLS])
    X = torch.tensor(context_scaled[np.newaxis, :, :], dtype=torch.float32).to(device)
    
    lstm_model.eval()
    with torch.no_grad():
        pred_scaled = lstm_model(X).cpu().numpy().flatten()
    
    pred_real = inverse_transform_target(pred_scaled, scaler, TARGET_IDX, len(FEATURE_COLS))
    return pred_real


def predict_prophet(future_df, models, output_horizon=24):
    prophet_model = models['prophet_model']
    
    future_prophet = future_df.reset_index()[['local_time'] + COVARIATE_COLS].copy()
    future_prophet.columns = ['ds'] + COVARIATE_COLS
    
    forecast = prophet_model.predict(future_prophet)
    return forecast['yhat'].values


def predict_chronos2(context_df, future_df, output_horizon=24):
    pipeline = get_chronos_pipeline()
    
    context_prep = context_df.reset_index()[['local_time', 'adjusted_demand'] + COVARIATE_COLS].copy()
    context_prep.columns = ['timestamp', 'target'] + COVARIATE_COLS
    context_prep['id'] = 'PJM'
    
    future_prep = future_df.reset_index()[['local_time'] + COVARIATE_COLS].copy()
    future_prep.columns = ['timestamp'] + COVARIATE_COLS
    future_prep['id'] = 'PJM'
    
    pred_df = pipeline.predict_df(
        context_prep, future_df=future_prep, prediction_length=output_horizon,
        quantile_levels=[0.5], id_column="id", timestamp_column="timestamp", target="target"
    )
    return pred_df['0.5'].values


def predict_ensemble(lstm_preds, chronos_preds, weights):
    return weights['lstm'] * lstm_preds + weights['chronos2'] * chronos_preds


def run_prediction(model_name, context_df, future_df, models, output_horizon=24):
    """Fungsi utama, dipanggil dari endpoint FastAPI"""
    
    if model_name == "lstm":
        return predict_lstm(context_df, models, output_horizon)
    elif model_name == "prophet":
        return predict_prophet(future_df, models, output_horizon)
    elif model_name == "chronos2":
        return predict_chronos2(context_df, future_df, output_horizon)
    elif model_name == "ensemble":
        lstm_preds = predict_lstm(context_df, models, output_horizon)
        chronos_preds = predict_chronos2(context_df, future_df, output_horizon)
        weights = models['config']['ensemble_weights']
        return predict_ensemble(lstm_preds, chronos_preds, weights)
    else:
        raise ValueError(f"Model '{model_name}' tidak dikenali. Pilih: lstm, prophet, chronos2, ensemble")