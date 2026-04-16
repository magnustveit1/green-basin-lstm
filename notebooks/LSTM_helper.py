import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib


# Set random seeds for reproducibility
SEED = 42

# Define a function to add scaled columns to a DataFrame
def add_scaled_columns(path, feature_cols, target_col, frame: pd.DataFrame) -> pd.DataFrame:
    #load scalers
    feature_scaler = joblib.load(os.path.join(path, "feature_scaler.pkl"))
    target_scaler = joblib.load(os.path.join(path, "target_scaler.pkl"))
    out = frame.copy()
    out[feature_cols] = feature_scaler.transform(out[feature_cols])
    out[target_col] = target_scaler.transform(out[[target_col]])
    return out

# Define a function to create sequences for LSTM input
def make_sequences(DATE_COL, frame: pd.DataFrame, lookback: int, feature_cols, target_col):
    X, y, dates = [], [], []
    feature_values = frame[feature_cols].to_numpy(dtype=np.float32)
    target_values = frame[target_col].to_numpy(dtype=np.float32)
    date_values = frame[DATE_COL].to_numpy()

    for i in range(lookback, len(frame)):
        X.append(feature_values[i - lookback:i])
        y.append(target_values[i])
        dates.append(date_values[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(dates)

def evaluate(model, criterion, device, loader):
    model.eval()
    losses = []
    preds = []
    obs = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            losses.append(loss.item())
            preds.append(pred.detach().cpu().numpy())
            obs.append(yb.detach().cpu().numpy())
    return float(np.mean(losses)), np.concatenate(preds), np.concatenate(obs)

def save_model(model, LOOKBACK_DAYS, feature_cols, feature_scaler, target_scaler, MODEL_PATH):
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_cols': feature_cols,
        'lookback_days': LOOKBACK_DAYS,
        'feature_scaler_min': feature_scaler.data_min_,
        'feature_scaler_max': feature_scaler.data_max_,
        'target_scaler_min': target_scaler.data_min_,
        'target_scaler_max': target_scaler.data_max_,
    }, MODEL_PATH)
    print('Saved model to:', MODEL_PATH)


# Define a PyTorch Dataset class for the sequences
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)