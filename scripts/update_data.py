import requests
import pandas as pd
import numpy as np
import holidays
import os
import sys

API_KEY = os.environ.get("EIA_API_KEY")
LATITUDE = 39.9526
LONGITUDE = -75.1652
BASE_TEMP = 18
PARQUET_PATH = "data/pjm_features.parquet"

if not API_KEY:
    print("ERROR: EIA_API_KEY tidak ditemukan di environment variable")
    sys.exit(1)


def fetch_eia_data(data_type="D", days_back=10):
    """Fetch data Demand (D) atau Demand Forecast (DF) beberapa hari terakhir"""
    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    params = {
        "api_key": API_KEY,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": "PJM",
        "facets[type][]": data_type,
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": days_back * 24
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    df = pd.DataFrame(data['response']['data'])
    df['local_time'] = pd.to_datetime(df['period'], format='%Y-%m-%dT%H')
    df['value'] = pd.to_numeric(df['value'], errors='coerce') / 1000  # MWh -> GWh
    
    return df[['local_time', 'value']].set_index('local_time').sort_index()


def fetch_weather(start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date.strftime('%Y-%m-%d'),
        "end_date": end_date.strftime('%Y-%m-%d'),
        "hourly": "temperature_2m",
        "timezone": "America/New_York"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    weather_df = pd.DataFrame({
        'local_time': pd.to_datetime(data['hourly']['time']),
        'temperature': data['hourly']['temperature_2m']
    }).set_index('local_time')
    
    return weather_df


def generate_features(df):
    """Generate semua fitur turunan (waktu, holiday, cuaca) untuk dataframe baru"""
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['dayofweek'] = df.index.dayofweek
    df['date'] = df.index.date
    df['hour'] = df.index.hour
    df['dayofyear'] = df.index.dayofyear
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
    
    df['is_pandemic_period'] = 0  # selalu 0 untuk data baru (sudah lewat masa pandemi)
    
    df['cooling_degree'] = (df['temperature'] - BASE_TEMP).clip(lower=0)
    df['heating_degree'] = (BASE_TEMP - df['temperature']).clip(lower=0)
    df['temp_squared'] = df['temperature'] ** 2
    
    return df


def main():
    print("Fetching data demand terbaru dari EIA...")
    demand_df = fetch_eia_data(data_type="D", days_back=10)
    demand_df.columns = ['adjusted_demand']
    
    print("Fetching demand forecast...")
    forecast_df = fetch_eia_data(data_type="DF", days_back=10)
    forecast_df.columns = ['demand_forecast']
    
    combined = demand_df.join(forecast_df, how='outer')
    
    print("Fetching data cuaca...")
    weather_df = fetch_weather(combined.index.min(), combined.index.max())
    combined = combined.join(weather_df, how='left')
    combined['temperature'] = combined['temperature'].interpolate().ffill().bfill()
    
    print("Generating fitur turunan...")
    combined = generate_features(combined)
    
    # Lag & rolling perlu dihitung ulang setelah digabung dengan data lama
    # (nanti di-generate ulang setelah join dengan parquet lama)
    
    print("Loading data historis yang sudah ada...")
    existing_df = pd.read_parquet(PARQUET_PATH)
    
    print("Menggabungkan data baru dengan data historis...")
    # Hapus overlap: baris yang timestamp-nya sudah ada di data lama, pakai yang baru
    combined_new_only = combined[~combined.index.isin(existing_df.index)]
    
    updated_df = pd.concat([existing_df, combined_new_only]).sort_index()
    updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
    
    # Re-generate lag & rolling features untuk KESELURUHAN data (biar konsisten)
    updated_df['lag_1h'] = updated_df['adjusted_demand'].shift(1)
    updated_df['lag_24h'] = updated_df['adjusted_demand'].shift(24)
    updated_df['lag_168h'] = updated_df['adjusted_demand'].shift(168)
    updated_df['rolling_mean_24h'] = updated_df['adjusted_demand'].rolling(window=24).mean()
    updated_df['rolling_std_24h'] = updated_df['adjusted_demand'].rolling(window=24).std()
    
    updated_df.to_parquet(PARQUET_PATH)
    
    print(f"Selesai! Data ter-update: {len(combined_new_only)} baris baru ditambahkan.")
    print(f"Total data sekarang: {len(updated_df)} baris, s.d. {updated_df.index.max()}")


if __name__ == "__main__":
    main()