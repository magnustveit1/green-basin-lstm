"""
lstm_model.py

Main script for the Green Basin LSTM streamflow prediction pipeline.
Trains a single LSTM on three unregulated stream sites and evaluates
on one unseen test site.

Run data_acquisition.py first to generate the HydroDF CSV files.

Usage
-----
    conda activate torch310env
    python lstm_model.py

Architecture
------------
    Input  : 30-day sliding window of [prcp_mm_day, tmax_degC, swe_cm]
    LSTM   : 1 layer, 64 hidden units
    Output : next-day flow_cms (single-step prediction)
    Loss   : MSE on MinMax-scaled values

Split
-----
    Train      : 1990–2014  (all three training sites)
    Validation : 2015–2018  (all three training sites)
    Test       : 2019–2023  (09251000 only — never seen during training)
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import joblib

# ── Local helpers ─────────────────────────────────────────────────────────────
# LSTM_helper: professor's module — LSTMRegressor, SequenceDataset,
#              make_sequences, add_scaled_columns, evaluate, save_model
from utils import LSTM_helper

# data_utils:  fetch, load, and split HydroDF data
from utils.data_utils import (
    SITES, TRAIN_IDS, TEST_ID,
    DATE_COL, TARGET_COL, FEATURE_COLS,
    load_hydrodf, split_by_year,
)

# train_utils: training loop, metrics, and figure generation
from utils.train_utils import (
    train_model, compute_metrics, print_metrics,
    plot_training_history, plot_observed_vs_predicted,
    plot_scatter, plot_performance_summary,
)

# ── Configuration ─────────────────────────────────────────────────────────────

# Set random seeds for reproducibility across all libraries
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device: GPU if available, otherwise CPU
# (GPU support is enabled — set CUDA_VISIBLE_DEVICES="" to force CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Paths
DATA_DIR  = os.path.join('data', 'HydroDF')
MODEL_DIR = 'model'
FIG_DIR   = 'figures'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR,   exist_ok=True)

# Hyperparameters — match professor's defaults
LOOKBACK_DAYS = 30    # days of antecedent conditions fed as input window
BATCH_SIZE    = 64    # mini-batch size for training
EPOCHS        = 50    # maximum training epochs
PATIENCE      = 8     # early stopping patience
LEARNING_RATE = 1e-3  # Adam learning rate

# Year splits — match professor's convention
TRAIN_START_YEAR = 1990
TRAIN_END_YEAR   = 2014
VAL_START_YEAR   = 2015
VAL_END_YEAR     = 2018
TEST_START_YEAR  = 2019
TEST_END_YEAR    = 2023


# ── Step 1: Prepare training site data loaders ────────────────────────────────

def prepare_train_site(site_id: str):
    """
    Load, split, scale, and build DataLoaders for one training site.
    Saves scalers to model/ for later use by add_scaled_columns.

    Returns
    -------
    train_loader, val_loader : DataLoaders for this site's train/val periods
    target_scaler            : fitted target scaler (needed for inverse-transform)
    """
    df = load_hydrodf(site_id, DATA_DIR)

    # Select only the columns needed
    df = df[[DATE_COL] + FEATURE_COLS + [TARGET_COL]].copy()

    # Fill gaps — matches professor's interpolation approach
    df[FEATURE_COLS + [TARGET_COL]] = (
        df[FEATURE_COLS + [TARGET_COL]]
        .interpolate(method='linear', limit_direction='both')
        .ffill().bfill()
    )

    # Split by year
    train_df, val_df, _ = split_by_year(
        df, TRAIN_END_YEAR, VAL_START_YEAR, VAL_END_YEAR,
        TEST_START_YEAR, TEST_END_YEAR
    )

    # Fit separate scalers on training data only (prevents data leakage)
    feature_scaler = MinMaxScaler()
    target_scaler  = MinMaxScaler()
    feature_scaler.fit(train_df[FEATURE_COLS])
    target_scaler.fit(train_df[[TARGET_COL]])

    # Save scalers so LSTM_helper.add_scaled_columns can load them by path
    joblib.dump(feature_scaler, os.path.join(MODEL_DIR, f'feature_scaler_{site_id}.pkl'))
    joblib.dump(target_scaler,  os.path.join(MODEL_DIR, f'target_scaler_{site_id}.pkl'))

    # Temporarily rename scaler files to the expected names for add_scaled_columns
    # (it looks for feature_scaler.pkl and target_scaler.pkl in the given directory)
    joblib.dump(feature_scaler, os.path.join(MODEL_DIR, 'feature_scaler.pkl'))
    joblib.dump(target_scaler,  os.path.join(MODEL_DIR, 'target_scaler.pkl'))

    # Apply scalers to train and val splits
    train_scaled = LSTM_helper.add_scaled_columns(MODEL_DIR, FEATURE_COLS, TARGET_COL, train_df)
    val_scaled   = LSTM_helper.add_scaled_columns(MODEL_DIR, FEATURE_COLS, TARGET_COL, val_df)

    # Build 30-day sliding window sequences: X (N, 30, 3), y (N,), dates (N,)
    X_train, y_train, _ = LSTM_helper.make_sequences(
        DATE_COL, train_scaled, LOOKBACK_DAYS, FEATURE_COLS, TARGET_COL)
    X_val, y_val, _     = LSTM_helper.make_sequences(
        DATE_COL, val_scaled, LOOKBACK_DAYS, FEATURE_COLS, TARGET_COL)

    print(f'  {site_id} — Train shape: {X_train.shape}  Val shape: {X_val.shape}')

    # Wrap in PyTorch Datasets and DataLoaders
    train_loader = DataLoader(
        LSTM_helper.SequenceDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        LSTM_helper.SequenceDataset(X_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, target_scaler


# ── Step 2: Prepare test site data loader ─────────────────────────────────────

def prepare_test_site(site_id: str):
    """
    Load, split, scale, and build a DataLoader for the test site.
    Uses a scaler fitted on the test site's own training-period data
    (1990–2014) to avoid leakage — the model never sees this site at all.

    Returns
    -------
    test_loader  : DataLoader for the test period
    d_test       : date array for the test period
    target_scaler: fitted target scaler for inverse-transform
    """
    df = load_hydrodf(site_id, DATA_DIR)
    df = df[[DATE_COL] + FEATURE_COLS + [TARGET_COL]].copy()
    df[FEATURE_COLS + [TARGET_COL]] = (
        df[FEATURE_COLS + [TARGET_COL]]
        .interpolate(method='linear', limit_direction='both')
        .ffill().bfill()
    )

    train_df, _, test_df = split_by_year(
        df, TRAIN_END_YEAR, VAL_START_YEAR, VAL_END_YEAR,
        TEST_START_YEAR, TEST_END_YEAR
    )

    # Fit scalers on the test site's own training period (1990–2014)
    feature_scaler = MinMaxScaler()
    target_scaler  = MinMaxScaler()
    feature_scaler.fit(train_df[FEATURE_COLS])
    target_scaler.fit(train_df[[TARGET_COL]])

    # Save with the test site ID so they don't overwrite training scalers
    joblib.dump(feature_scaler, os.path.join(MODEL_DIR, f'feature_scaler_{site_id}.pkl'))
    joblib.dump(target_scaler,  os.path.join(MODEL_DIR, f'target_scaler_{site_id}.pkl'))
    joblib.dump(feature_scaler, os.path.join(MODEL_DIR, 'feature_scaler.pkl'))
    joblib.dump(target_scaler,  os.path.join(MODEL_DIR, 'target_scaler.pkl'))

    test_scaled = LSTM_helper.add_scaled_columns(MODEL_DIR, FEATURE_COLS, TARGET_COL, test_df)

    X_test, y_test, d_test = LSTM_helper.make_sequences(
        DATE_COL, test_scaled, LOOKBACK_DAYS, FEATURE_COLS, TARGET_COL)

    print(f'  {site_id} — Test shape: {X_test.shape}')

    test_loader = DataLoader(
        LSTM_helper.SequenceDataset(X_test, y_test),
        batch_size=BATCH_SIZE, shuffle=False)

    return test_loader, d_test, target_scaler


# ── Step 3: Evaluate one site and inverse-transform predictions ───────────────

def evaluate_site(model, loader, d_test, target_scaler):
    """
    Run the trained model on a DataLoader, inverse-transform predictions
    from scaled space back to cms, and compute performance metrics.

    Returns
    -------
    obs_cms, pred_cms : np.ndarray of observed and predicted flow in cms
    metrics           : dict of RMSE, MAE, R², NSE
    """
    criterion = torch.nn.MSELoss()
    _, pred_scaled, obs_scaled = LSTM_helper.evaluate(model, criterion, device, loader)

    # Inverse-transform from [0,1] scaled space back to cms
    obs_cms  = target_scaler.inverse_transform(obs_scaled.reshape(-1, 1)).ravel()
    pred_cms = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    metrics = compute_metrics(obs_cms, pred_cms)
    return obs_cms, pred_cms, metrics


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('Green Basin LSTM — Streamflow Prediction')
    print(f'Training sites: {TRAIN_IDS}')
    print(f'Test site:      {TEST_ID}')
    print('=' * 60)

    # ── Build training loaders ─────────────────────────────────────────────
    print('\n[1/5] Preparing training site data...')
    all_train_loaders, all_val_loaders = [], []
    train_scalers = {}

    for sid in TRAIN_IDS:
        print(f'\n  {SITES[sid]["role"]}: {sid}')
        tl, vl, ts = prepare_train_site(sid)
        all_train_loaders.append(tl)
        all_val_loaders.append(vl)
        train_scalers[sid] = ts

    # Combine all training site loaders into flat lists for training loop
    # (the training loop cycles through all sites each epoch)
    combined_train = [loader for loader in all_train_loaders]
    combined_val   = [loader for loader in all_val_loaders]

    # ── Define model ───────────────────────────────────────────────────────
    print('\n[2/5] Defining LSTM model...')
    model = LSTM_helper.LSTMRegressor(
        input_size=len(FEATURE_COLS),
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
    ).to(device)
    print(model)

    # ── Train ──────────────────────────────────────────────────────────────
    print(f'\n[3/5] Training on {len(TRAIN_IDS)} sites '
          f'(max {EPOCHS} epochs, patience={PATIENCE})...')

    # We train one model across all three sites by merging loaders.
    # Flatten all batches from all training sites into one combined loader list.
    from torch.utils.data import ConcatDataset
    combined_train_dataset = ConcatDataset([tl.dataset for tl in combined_train])
    combined_val_dataset   = ConcatDataset([vl.dataset for vl in combined_val])

    combined_train_loader = DataLoader(combined_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    combined_val_loader   = DataLoader(combined_val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    model, history = train_model(
        model, combined_train_loader, combined_val_loader,
        device, EPOCHS, LEARNING_RATE, PATIENCE
    )

    plot_training_history(history, FIG_DIR)

    # ── Save model ─────────────────────────────────────────────────────────
    MODEL_PATH = os.path.join(MODEL_DIR, 'green_basin_lstm.pt')
    # Use the first training site's scalers as the reference for save_model
    ref_scaler_f = joblib.load(os.path.join(MODEL_DIR, f'feature_scaler_{TRAIN_IDS[0]}.pkl'))
    ref_scaler_t = joblib.load(os.path.join(MODEL_DIR, f'target_scaler_{TRAIN_IDS[0]}.pkl'))
    LSTM_helper.save_model(model, LOOKBACK_DAYS, FEATURE_COLS,
                           ref_scaler_f, ref_scaler_t, MODEL_PATH)

    # ── Evaluate ────────────────────────────────────────────────────────────
    print('\n[4/5] Evaluating all sites on test period (2019–2023)...')
    results = {}

    # Training sites — evaluate on their own test period
    for sid in TRAIN_IDS:
        # Re-prepare the test split for each training site using its own scaler
        df = load_hydrodf(sid, DATA_DIR)
        df = df[[DATE_COL] + FEATURE_COLS + [TARGET_COL]].copy()
        df[FEATURE_COLS + [TARGET_COL]] = (
            df[FEATURE_COLS + [TARGET_COL]]
            .interpolate(method='linear', limit_direction='both')
            .ffill().bfill()
        )
        _, _, test_df = split_by_year(
            df, TRAIN_END_YEAR, VAL_START_YEAR, VAL_END_YEAR,
            TEST_START_YEAR, TEST_END_YEAR
        )
        # Load this site's own scalers
        fs = joblib.load(os.path.join(MODEL_DIR, f'feature_scaler_{sid}.pkl'))
        ts = joblib.load(os.path.join(MODEL_DIR, f'target_scaler_{sid}.pkl'))
        joblib.dump(fs, os.path.join(MODEL_DIR, 'feature_scaler.pkl'))
        joblib.dump(ts, os.path.join(MODEL_DIR, 'target_scaler.pkl'))

        test_scaled = LSTM_helper.add_scaled_columns(MODEL_DIR, FEATURE_COLS, TARGET_COL, test_df)
        X_t, y_t, d_t = LSTM_helper.make_sequences(
            DATE_COL, test_scaled, LOOKBACK_DAYS, FEATURE_COLS, TARGET_COL)
        loader = DataLoader(LSTM_helper.SequenceDataset(X_t, y_t),
                            batch_size=BATCH_SIZE, shuffle=False)
        obs, pred, metrics = evaluate_site(model, loader, d_t, ts)
        results[sid] = {
            'dates': d_t, 'obs_cms': obs, 'pred_cms': pred,
            'metrics': metrics, 'role': SITES[sid]['role'],
            'label': f'{SITES[sid]["name"].replace("_"," ")} ({SITES[sid]["role"]})',
        }
        print_metrics(results[sid]['label'], metrics)

    # Test site — the site the model never saw during training
    print(f'\n  Preparing test site: {TEST_ID}')
    test_loader, d_test, test_target_scaler = prepare_test_site(TEST_ID)
    obs, pred, metrics = evaluate_site(model, test_loader, d_test, test_target_scaler)
    results[TEST_ID] = {
        'dates': d_test, 'obs_cms': obs, 'pred_cms': pred,
        'metrics': metrics, 'role': SITES[TEST_ID]['role'],
        'label': f'{SITES[TEST_ID]["name"].replace("_"," ")} ({SITES[TEST_ID]["role"]})',
    }
    print_metrics(results[TEST_ID]['label'], metrics)

    # ── Figures ────────────────────────────────────────────────────────────
    print('\n[5/5] Generating figures...')
    plot_observed_vs_predicted(results, FIG_DIR)
    plot_scatter(results, FIG_DIR)
    plot_performance_summary(results, FIG_DIR)

    print('\nPipeline complete.')
    print(f'Model saved to:  {MODEL_PATH}')
    print(f'Figures saved to: {FIG_DIR}/')
    print('=' * 60)


if __name__ == '__main__':
    main()
