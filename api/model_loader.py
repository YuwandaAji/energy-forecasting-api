import torch
import torch.nn as nn
import joblib
import pickle
import json
import numpy as np

MODEL_DIR = "models"

# ============================================
# Definisi ulang arsitektur LSTM (HARUS identik dengan waktu training)
# ============================================
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_horizon=24, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_horizon)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out


def load_all_models():
    # 1. Load config
    with open(f"{MODEL_DIR}/config.json", "r") as f:
        config = json.load(f)
    
    # 2. Load scaler
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    
    # 3. Load LSTM
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(f"{MODEL_DIR}/lstm_final.pth", map_location=device)
    
    lstm_model = LSTMForecaster(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        output_horizon=checkpoint['output_horizon']
    ).to(device)
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    lstm_model.eval()
    
    # 4. Load Prophet
    with open(f"{MODEL_DIR}/prophet_final.pkl", "rb") as f:
        prophet_model = pickle.load(f)
    
    return {
        'config': config,
        'scaler': scaler,
        'lstm_model': lstm_model,
        'prophet_model': prophet_model,
        'device': device
    }


if __name__ == "__main__":
    print("Testing model loading...")
    models = load_all_models()
    print("\nSemua model berhasil di-load!")
    print("Config keys:", list(models['config'].keys()))
    print("Device:", models['device'])
    print("LSTM input_size:", models['lstm_model'].lstm.input_size)