import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(__file__))

from model_loader import load_all_models
from features import get_context_and_future
from inference import predict_lstm, predict_prophet, predict_chronos2, predict_ensemble

st.set_page_config(page_title="Energy Demand Forecasting", page_icon="⚡", layout="wide")

@st.cache_resource
def get_models():
    return load_all_models()

st.title("⚡ Energy Demand Forecasting Dashboard")
st.caption("Prediksi konsumsi listrik PJM 24 jam ke depan")

with st.spinner("Loading models..."):
    models = get_models()

col1, col2 = st.columns(2)
with col1:
    prediction_date = st.date_input("Tanggal mulai prediksi")
with col2:
    prediction_time = st.time_input("Jam mulai prediksi")

st.subheader("Pilih mode prediksi")
mode = st.radio("Mode", ["Evaluasi Historis (pilih tanggal manual)", "Prediksi Otomatis (24 jam berikutnya dari data terbaru)"])

if mode == "Prediksi Otomatis (24 jam berikutnya dari data terbaru)":
    prediction_start = None  # nanti otomatis dihitung di features.py
    st.info("Sistem akan otomatis memprediksi 24 jam setelah data terbaru yang tersedia.")
else:
    col1, col2 = st.columns(2)
    with col1:
        prediction_date = st.date_input("Tanggal mulai prediksi")
    with col2:
        prediction_time = st.time_input("Jam mulai prediksi")
    prediction_start = pd.Timestamp.combine(prediction_date, prediction_time).floor('h')

model_choice = st.selectbox("Pilih model", ["lstm", "prophet", "chronos2", "ensemble"])

if st.button("Jalankan Prediksi", type="primary"):
    try:
        with st.spinner(f"Menjalankan prediksi dengan {model_choice}..."):
            if mode == "Prediksi Otomatis (24 jam berikutnya dari data terbaru)":
                # paksa pakai titik waktu yang jelas-jelas di masa depan
                from features import load_historical_data
                df_check = load_historical_data()
                valid_max = df_check['adjusted_demand'].last_valid_index()
                prediction_start = valid_max + pd.Timedelta(hours=1)
            
            context_df, future_df, is_future, actual_prediction_start = get_context_and_future(
                prediction_start, input_window=168, output_horizon=24
            )
            
            if model_choice == "lstm":
                preds = predict_lstm(context_df, models)
            elif model_choice == "prophet":
                preds = predict_prophet(future_df, models)
            elif model_choice == "chronos2":
                preds = predict_chronos2(context_df, future_df)
            elif model_choice == "ensemble":
                lstm_preds = predict_lstm(context_df, models)
                chronos_preds = predict_chronos2(context_df, future_df)
                weights = models['config']['ensemble_weights']
                preds = predict_ensemble(lstm_preds, chronos_preds, weights)
            
            timestamps = pd.date_range(actual_prediction_start, periods=24, freq='h')
            st.write("Debug - tipe data preds:", type(preds))
            st.write("Debug - isi preds:", preds)
            result_df = pd.DataFrame({
                'Waktu': timestamps,
                'Prediksi Demand (MW)': preds
            })
            
            if is_future:
                st.info(f"📅 Prediksi masa depan dimulai dari {actual_prediction_start}")
            else:
                st.success("📊 Evaluasi terhadap data historis")
            
            st.line_chart(result_df.set_index('Waktu'))
            st.dataframe(result_df, use_container_width=True)
    
    except ValueError as e:
        st.error(f"Error: {str(e)}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")