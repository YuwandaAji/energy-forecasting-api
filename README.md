# ⚡ PJM Energy Demand Forecasting

End-to-end time series forecasting system for hourly electricity demand in the PJM Interconnection grid (13 U.S. states + Washington D.C.), combining classical statistical methods, deep learning, and a zero-shot time series foundation model — with automated daily data updates and a live interactive dashboard.

**🔗 Live Demo:** [Streamlit App](https://energy-forecasting-aji.streamlit.app/) 
**📊 Data Source:** [EIA-930 Hourly Electric Grid Monitor](https://www.eia.gov/electricity/data/eia930) (U.S. Energy Information Administration)

---

## 📌 Project Motivation

Electricity cannot be stored at scale — grid operators must predict demand *before* it happens to balance generation and avoid blackouts or wasted capacity. This project builds a day-ahead (24-hour horizon) demand forecasting system for PJM, one of the largest grid operators in the U.S., and benchmarks it against PJM's own official day-ahead forecast.

This is the 4th project in a portfolio series deliberately designed to cover different domains and paradigms:
1. Real-time crypto pipeline (Kafka + Spark Streaming)
2. MLOps crypto pipeline (MLflow + FastAPI)
3. Batch e-commerce analytics (Airflow + dbt + BigQuery)
4. **Energy demand forecasting (this project)** — first deep learning + foundation model project in the series

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   EIA-930 API    │     │  Open-Meteo API   │     │  GitHub Actions  │
│  (demand data)   │     │  (weather data)   │     │ (daily schedule) │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                         │
         └───────────┬────────────┘                         │
                      ▼                                      │
         ┌────────────────────────┐                          │
         │  scripts/update_data.py │◄─────────────────────────┘
         │  (fetch, clean, merge)  │
         └────────────┬────────────┘
                      ▼
         ┌────────────────────────┐
         │  pjm_features.parquet   │  (auto-committed daily)
         └────────────┬────────────┘
                      ▼
      ┌───────────────┴────────────────┐
      ▼                                ▼
┌──────────────┐              ┌──────────────────┐
│   FastAPI     │              │  Streamlit App     │
│ (Docker, local)│              │ (deployed, public) │
└──────────────┘              └──────────────────┘
      │                                │
      └──────────────┬─────────────────┘
                      ▼
        ┌─────────────────────────┐
        │  Prophet │ LSTM │ Chronos-2 │
        │       Ensemble (LSTM+Chronos)  │
        └─────────────────────────┘
```

---

## 📂 Data

| Source | Purpose | Notes |
|---|---|---|
| [EIA-930 API](https://www.eia.gov/opendata/) | Hourly demand (`Adjusted demand`) + PJM's own day-ahead forecast | Official U.S. government data, 2015–present |
| [Open-Meteo](https://open-meteo.com/) | Hourly temperature (Philadelphia, PA — largest load center proxy for PJM) | Free, no API key required |

**Why Philadelphia as weather proxy?** PJM spans 13 states, so no single city perfectly represents grid-wide weather. Philadelphia was chosen as a reasonable population-weighted proxy.

### Data Quality Findings
- **DST transitions**: consistent 1-hour gaps (spring forward) and duplicate timestamps (fall back) every March/November since 2015 — handled via reindexing + linear interpolation.
- **PJM's day-ahead forecast field is missing ~22–24 hours around every DST transition** — traced to a likely automated reporting disruption during clock shifts, not a data quality issue on our end.
- **COVID-19 impact (Mar–Jun 2020)**: demand dropped ~8% and the weekday/weekend gap flattened significantly (commercial load dropped, residential load rose), consistent with PJM's own published findings. Flagged via an `is_pandemic_period` binary feature rather than excluded, to preserve the full 11-year seasonal history.
- **Weak correlation between `heating_degree` and demand** (r=0.06) vs. strong correlation for `cooling_degree` (r=0.62) — attributed to most PJM-area heating being gas-based, while cooling (AC) is almost universally electric.

---

## 🔧 Feature Engineering

- Cyclical time encoding (hour, day-of-week, month) via sin/cos transforms
- US federal holiday flag (`holidays` library)
- Heating/Cooling Degree Hours (base 18°C) — standard energy-industry non-linear temperature features
- Lag features (1h, 24h, 168h) and rolling statistics (24h mean/std)
- Pandemic period flag

---

## 🤖 Models & Evaluation

Three approaches were benchmarked, each representing a different modeling philosophy, plus a weighted ensemble:

| Model | Approach | Walk-forward MAPE (avg, 5 folds) | Final Test MAPE (12-month holdout) |
|---|---|---|---|
| Prophet + Weather regressors | Classical statistical, interpretable | 5.24% | 5.21% |
| LSTM (PyTorch, 168h→24h) | Deep learning, sequence modeling | 2.68%* | 2.93% |
| Chronos-2 (zero-shot) | Time series foundation model, **no training** | 2.81% | 3.00% |
| **Ensemble (LSTM + Chronos-2, 50/50)** | — | **2.30%** | **2.50%** |

*\*Block-based evaluation (24h non-overlapping windows), aligned with Chronos-2's zero-shot evaluation scheme for fair comparison.*

### Validation Methodology
- **Walk-forward validation** with 5 expanding-window folds (never randomly split — time series data must respect chronological order)
- **Sealed 12-month final test set**, untouched during all development/tuning, used exactly once for final reporting
- Multiple metrics tracked (MAE, RMSE, MAPE) across all models, logged via **MLflow**

### Key Findings
- **LSTM outperforms Prophet substantially** at the 24-hour horizon, confirming the hypothesis that short-term sequential patterns favor deep learning over trend/seasonality decomposition.
- **Chronos-2, despite zero training on PJM-specific data, performs competitively with — and in one fold outperforms — LSTM**, particularly during an extreme cold snap period (Jan–Mar 2025, avg temp −0.24°C) where both Prophet and LSTM struggled. This suggests the foundation model's broad pretraining exposure helps generalize to rare/extreme patterns underrepresented in local training data.
- **The ensemble wins in every single fold**, not just on average — evidence that LSTM and Chronos-2 have complementary error patterns.
- Consistency between walk-forward and final test results (<0.3 percentage point difference across all models) indicates the models generalize well rather than overfitting to the development period.

---

## 🔄 Automated Data Pipeline

A GitHub Actions workflow (`.github/workflows/update_data.yml`) runs daily:
1. Fetches the latest hourly demand + forecast data from the EIA API
2. Fetches corresponding weather data from Open-Meteo
3. Regenerates all derived features
4. Merges with existing historical data (deduplication-safe)
5. Auto-commits the updated dataset back to the repository

This keeps the "most recent 168 hours" context used for live forecasting always current, without manual intervention.

---

## 🚀 Deployment

| Component | Stack | Status |
|---|---|---|
| REST API | FastAPI + Docker, Pydantic schemas, 4 model endpoints | Built and tested locally (Docker) |
| Dashboard | Streamlit | **Deployed publicly** on Streamlit Community Cloud |

**Why isn't the FastAPI service deployed publicly?** Every free-tier cloud platform tested (Render, Hugging Face Spaces Docker) required credit card verification during this project — a policy that in some cases changed mid-project. Rather than blocking on this, the inference logic (`model_loader.py`, `features.py`, `inference.py`) was refactored to be directly importable, allowing the Streamlit app to run predictions in-process without a separate live API call. The FastAPI service remains fully functional and demonstrates production-style API design; it's documented here and runnable via Docker locally.

### Dashboard Modes
- **Historical Evaluation**: pick any past date to compare model predictions against actual recorded demand
- **Automatic Forecast**: one-click prediction for the next 24 hours from the most recent available data point

---

## ⚠️ Known Limitations & Future Work

- **24-hour horizon only**: models are trained and validated specifically for day-ahead forecasting (matching real grid-operator practice). Multi-day forecasting would require recursive forecasting, which introduces compounding error and has not been validated in this project.
- **No automated model retraining**: the pipeline updates data daily, but model weights are static. Continuous retraining with automated validation gates is a natural next step for a production system.
- **Single weather station proxy**: using Philadelphia alone is a simplification; a production system would likely aggregate weather across multiple PJM sub-regions.
- **Chronos-2 CPU inference is resource-intensive**: local testing on constrained hardware (8GB RAM) surfaced segmentation faults under memory pressure — a real-world reminder that foundation models, while training-free, are not necessarily lightweight at inference time.

---

## 🛠️ Tech Stack

**Data & Features**: Python, Pandas, NumPy, `holidays`
**Modeling**: Prophet, PyTorch (LSTM), Amazon Chronos-2, scikit-learn
**Experiment Tracking**: MLflow
**API**: FastAPI, Pydantic, Docker
**Dashboard**: Streamlit
**Automation**: GitHub Actions
**Data Sources**: EIA API v2, Open-Meteo API

---

## 📁 Repository Structure

```
├── api/                    # FastAPI service (Docker, local demo)
├── streamlit_app/          # Deployed dashboard
├── scripts/
│   └── update_data.py      # Daily automated data refresh
├── models/                 # Trained model artifacts (LSTM, Prophet, scaler, config)
├── data/                   # Feature-engineered historical dataset
├── .github/workflows/      # GitHub Actions automation
└── notebooks/              # Training & experimentation notebooks (Colab)
```

---

## 🏃 Running Locally

```bash
# Clone repo
git clone https://github.com/YuwandaAji/energy-forecasting-api.git
cd energy-forecasting-api

# Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run Streamlit dashboard
streamlit run streamlit_app/app.py

# OR run FastAPI (Docker)
docker build -t energy-forecasting-api .
docker run -p 8000:8000 energy-forecasting-api
# Visit http://localhost:8000/docs
```

---

## 👤 Author

**Yuwanda Aji** — Information Systems, Universitas Negeri Surabaya
Building a portfolio across Data Engineering, ML Engineering, and Data Science.

[GitHub](https://github.com/YuwandaAji) · [LinkedIn](www.linkedin.com/in/yuwanda-aji-pangestu-309675292)
