import pandas as pd
import numpy as np
import requests
import holidays

FEATURE_COLS = ['adjusted_demand', 'temperature', 'cooling_degree', 'heating_degree',
                'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
                'month_sin', 'month_cos', 'is_weekend', 'is_holiday']
COVARIATE_COLS = ['temperature', 'cooling_degree', 'heating_degree',
                   'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
                   'month_sin', 'month_cos', 'is_weekend', 'is_holiday']

LATITUDE = 39.9526
LONGITUDE = -75.1652
BASE_TEMP = 18

_historical_data = None

def load_historical_data():
    global _historical_data
    if _historical_data is None:
        _historical_data = pd.read_parquet("data/pjm_features.parquet")
    return _historical_data


def generate_time_features(timestamps):
    df = pd.DataFrame(index=pd.DatetimeIndex(timestamps))
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    us_holidays = holidays.US(years=range(df.index.year.min(), df.index.year.max()+1))
    df['is_holiday'] = df.index.to_series().dt.date.astype(str).isin(
        [str(d) for d in us_holidays.keys()]
    ).astype(int)
    
    return df


def fetch_weather_forecast(start_time, end_time):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "temperature_2m",
        "start_date": start_time.strftime('%Y-%m-%d'),
        "end_date": end_time.strftime('%Y-%m-%d'),
        "timezone": "America/New_York"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    weather_df = pd.DataFrame({
        'local_time': pd.to_datetime(data['hourly']['time']),
        'temperature': data['hourly']['temperature_2m']
    }).set_index('local_time')
    
    return weather_df


def get_context_and_future(prediction_start, input_window=168, output_horizon=24):
    df = load_historical_data()
    
    is_future = prediction_start > df.index.max()
    
    if is_future:
        # Untuk masa depan: SELALU pakai 168 jam TERAKHIR yang tersedia sebagai context
        context_end = df.index.max() + pd.Timedelta(hours=1)
        context_start = context_end - pd.Timedelta(hours=input_window)
        context_df = df.loc[context_start:context_end - pd.Timedelta(hours=1)].copy()
        
        actual_prediction_start = df.index.max() + pd.Timedelta(hours=1)
        future_end = actual_prediction_start + pd.Timedelta(hours=output_horizon)
        
        weather_future = fetch_weather_forecast(actual_prediction_start, future_end)
        future_time_feats = generate_time_features(
            pd.date_range(actual_prediction_start, periods=output_horizon, freq='h')
        )
        future_df = future_time_feats.join(weather_future, how='left')
        future_df['temperature'] = future_df['temperature'].interpolate().ffill().bfill()
        future_df['cooling_degree'] = (future_df['temperature'] - BASE_TEMP).clip(lower=0)
        future_df['heating_degree'] = (BASE_TEMP - future_df['temperature']).clip(lower=0)
        future_df.index.name = 'local_time'
        
        return context_df, future_df[COVARIATE_COLS], is_future, actual_prediction_start
    
    else:
        context_end = prediction_start
        context_start = prediction_start - pd.Timedelta(hours=input_window)
        future_end = prediction_start + pd.Timedelta(hours=output_horizon)
        
        context_df = df.loc[context_start:context_end - pd.Timedelta(hours=1)].copy()
        
        if len(context_df) < input_window:
            raise ValueError(f"Data histori tidak cukup. Butuh {input_window} jam sebelum {prediction_start}, "
                              f"hanya tersedia {len(context_df)} jam.")
        
        future_df = df.loc[prediction_start:future_end - pd.Timedelta(hours=1)].copy()
        if len(future_df) < output_horizon:
            raise ValueError(f"Data historis untuk periode target tidak lengkap.")
        
        return context_df, future_df[COVARIATE_COLS], is_future, prediction_start