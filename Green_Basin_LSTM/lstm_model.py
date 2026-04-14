"""
lstm_model.py
MAIN SCRIPT
==============================
Green Basin LSTM - Unregulated Streamflow Prediction
GEOG 6150 Hydroinformatics, University of Utah
Author: Magnus Tveit

Trains a single PyTorch LSTM on three unregulated stream sites and
evaluates on one unseen test site. Follows the workflow from the
course reference notebook (Hydro_LSTM.ipynb).

Prerequisites
-------------
    Run data_acquisition.py once first to fetch and cache the data.

Usage
-----
    conda activate torch310env
    python lstm_model.py

Training sites (all unregulated - no major dams upstream)
----------------------------------------------------------
    09217000  Green River nr Green River, WY       (Train 1 - snowmelt headwater)
    09306500  White River nr Watson, UT            (Train 2 - plateau tributary)
    09239500  Yampa River at Steamboat Springs, CO (Train 3 - upper free-flowing)

Test site (never seen during training)
09251000  Yampa River at Deerlodge Park, CO

Model
-----
    Input  : 30-day sliding window of [prcp_mm_day, tmax_degC, swe_cm]
    LSTM   : 1 layer, 64 hidden units, no dropout
    Output : next-day flow_cms (single-step prediction)
    Loss   : MSE on MinMax-scaled values
    Split  : Train 1990–2014 · Val 2015–2018 · Test 2019–2023

Outputs
-------
    model/green_basin_lstm.pt          trained model weights + metadata
    model/feature_scaler_{id}.pkl      per-site feature scalers
    model/target_scaler_{id}.pkl       per-site target scalers
    figures/fig_training_history.png
    figures/fig_observed_vs_predicted.png
    figures/fig_scatter.png
    figures/fig_performance_summary.png
"""

import os
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, ConcatDataset
import joblib

# Dr. Johnson helper)
from notebooks import LSTM_helper

# Other helpers
from notebooks.data_utils import (
    SITES, TRAIN_IDS, TEST_ID,
    DATE_COL, TARGET_COL, FEATURE_COLS,
    load_hydrodf, split_by_year,
)

# Training and plotting helpers
from notebooks.train_utils import (
    train_model, compute_metrics, print_metrics,
    plot_training_history, plot_observed_vs_predicted,
    plot_scatter, plot_performance_summary,
)


# Reproducibility Set Up

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# Paths
DATA_DIR  = os.path.join('data', 'HydroDF')
MODEL_DIR = 'model'
FIG_DIR   = 'figures'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR,   exist_ok=True)


# Hyperparameters

LOOKBACK_DAYS = 30     # days of antecedent conditions fed as input window
BATCH_SIZE    = 64     
EPOCHS        = 50     
PATIENCE      = 8      # early stopping
LEARNING_RATE = 1e-3  

TRAIN_END_YEAR  = 2014
VAL_START_YEAR  = 2015
VAL_END_YEAR    = 2018
TEST_START_YEAR = 2019
TEST_END_YEAR   = 2023


# Helper: prepare one site's DataLoaders Helper

def prepare_site_loaders(site_id: str):
    """
    Load HydroDF CSV for one site, split by year, fit scalers on training
    data only, apply scalers, build sequences, and return DataLoaders.

    Scalers are saved to model/ so they can be reloaded for evaluation.
    Follows the professor's workflow from Hydro_LSTM.ipynb step by step.

    Returns
    -------
    train_loader, val_loader : DataLoaders for this site
    target_scaler            : fitted target scaler (needed for inverse-transform)
    """
    df = load_hydrodf(site_id, DATA_DIR)

    # Keep only the columns needed, fill any gaps
    df = df[[DATE_COL] + FEATURE_COLS + [TARGET_COL]].copy()
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

    # Fit separate scalers on training data only — prevents data leakage
    feature_scaler = MinMaxScaler()
    target_scaler  = MinMaxScaler()
    feature_scaler.fit(train_df[FEATURE_COLS])
    target_scaler.fit(train_df[[TARGET_COL]])

    # Save scalers — LSTM_helper.add_scaled_columns loads them by path
    joblib.dump(feature_scaler, os.path.join(MODEL_DIR, 'feature_scaler.pkl'))
    joblib.dump(target_scaler,  os.path.join(MODEL_DIR, 'target_scaler.pkl'))

    # Also save with site ID so each site's scalers are preserved
    joblib.dump(feature_scaler, os.path.join(MODEL_DIR, f'feature_scaler_{site_id}.pkl'))
    joblib.dump(target_scaler,  os.path.join(MODEL_DIR, f'target_scaler_{site_id}.pkl'))

    # Apply scalers — matches professor's add_scaled_columns pattern
    train_scaled = LSTM_helper.add_scaled_columns(MODEL_DIR, FEATURE_COLS, TARGET_COL, train_df)
    val_scaled   = LSTM_helper.add_scaled_columns(MODEL_DIR, FEATURE_COLS, TARGET_COL, val_df)

    # Build 30-day sliding window sequences
    X_train, y_train, _ = LSTM_helper.make_sequences(
        DATE_COL, train_scaled, LOOKBACK_DAYS, FEATURE_COLS, TARGET_COL)
    X_val, y_val, _     = LSTM_helper.make_sequences(
        DATE_COL, val_scaled, LOOKBACK_DAYS, FEATURE_COLS, TARGET_COL)

    print(f'  {site_id} — train: {X_train.shape}  val: {X_val.shape}')

    train_loader = DataLoader(
        LSTM_helper.SequenceDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        LSTM_helper.SequenceDataset(X_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, target_scaler


# Helper: evaluate one site on its test period

def evaluate_site(site_id: str):
    """
    Load HydroDF for one site, prepare the test period using that site's
    own saved scalers, run the trained model, and return results.

    Returns dict with: dates, obs_cms, pred_cms, metrics, label, role
    """
    df = load_hydrodf(site_id, DATA_DIR)
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

    # Load this site's own scalers — fit on its 1990–2014 training period
    fs = joblib.load(os.path.join(MODEL_DIR, f'feature_scaler_{site_id}.pkl'))
    ts = joblib.load(os.path.join(MODEL_DIR, f'target_scaler_{site_id}.pkl'))

    # Write to the generic path so add_scaled_columns can find them
    joblib.dump(fs, os.path.join(MODEL_DIR, 'feature_scaler.pkl'))
    joblib.dump(ts, os.path.join(MODEL_DIR, 'target_scaler.pkl'))

    test_scaled = LSTM_helper.add_scaled_columns(MODEL_DIR, FEATURE_COLS, TARGET_COL, test_df)
    X_test, y_test, d_test = LSTM_helper.make_sequences(
        DATE_COL, test_scaled, LOOKBACK_DAYS, FEATURE_COLS, TARGET_COL)

    test_loader = DataLoader(
        LSTM_helper.SequenceDataset(X_test, y_test),
        batch_size=BATCH_SIZE, shuffle=False)

    # Run model in eval mode
    criterion = torch.nn.MSELoss()
    _, pred_scaled, obs_scaled = LSTM_helper.evaluate(model, criterion, device, test_loader)

    # Inverse-transform from scaled space back to cms
    obs_cms  = ts.inverse_transform(obs_scaled.reshape(-1, 1)).ravel()
    pred_cms = ts.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    metrics = compute_metrics(obs_cms, pred_cms)
    label   = f'{SITES[site_id]["name"].replace("_", " ")} ({SITES[site_id]["role"]})'

    return {
        'dates':    d_test,
        'obs_cms':  obs_cms,
        'pred_cms': pred_cms,
        'metrics':  metrics,
        'label':    label,
        'role':     SITES[site_id]['role'],
    }


# Main pipeline

if __name__ == '__main__':

    print('=' * 60)
    print('Green Basin LSTM — Main Pipeline')
    print(f'Training sites : {TRAIN_IDS}')
    print(f'Test site      : {TEST_ID}')
    print('=' * 60)

    # Step 1: Build DataLoaders for all training sites
    print('\n[1/5] Preparing training data...')
    train_loaders, val_loaders = [], []

    for sid in TRAIN_IDS:
        print(f'\n  {SITES[sid]["role"]}: {sid}')
        tl, vl, _ = prepare_site_loaders(sid)
        train_loaders.append(tl)
        val_loaders.append(vl)

    # Also prepare the test site's scalers now (fitted on its own train period)
    print(f'\n  Test site: {TEST_ID}')
    _, _, _ = prepare_site_loaders(TEST_ID)

    # Combine all three training sites into single loaders for the training loop
    combined_train = DataLoader(
        ConcatDataset([tl.dataset for tl in train_loaders]),
        batch_size=BATCH_SIZE, shuffle=True)
    combined_val = DataLoader(
        ConcatDataset([vl.dataset for vl in val_loaders]),
        batch_size=BATCH_SIZE, shuffle=False)

    # Step 2: Define model
    print('\n[2/5] Defining model...')
    model = LSTM_helper.LSTMRegressor(
        input_size=len(FEATURE_COLS),
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
    ).to(device)
    print(model)

    # Step 3: Train
    print(f'\n[3/5] Training (max {EPOCHS} epochs, patience={PATIENCE})...')
    model, history = train_model(
        model, combined_train, combined_val,
        device, EPOCHS, LEARNING_RATE, PATIENCE
    )
    plot_training_history(history, FIG_DIR)

    # Save model using professor's save_model function
    ref_fs = joblib.load(os.path.join(MODEL_DIR, f'feature_scaler_{TRAIN_IDS[0]}.pkl'))
    ref_ts = joblib.load(os.path.join(MODEL_DIR, f'target_scaler_{TRAIN_IDS[0]}.pkl'))
    MODEL_PATH = os.path.join(MODEL_DIR, 'green_basin_lstm.pt')
    LSTM_helper.save_model(model, LOOKBACK_DAYS, FEATURE_COLS, ref_fs, ref_ts, MODEL_PATH)

    # Step 4: Evaluate all four sites
    print('\n[4/5] Evaluating all sites on test period (2019–2023)...')
    results = {}
    for sid in TRAIN_IDS + [TEST_ID]:
        print(f'\n  {SITES[sid]["role"]}: {sid}')
        results[sid] = evaluate_site(sid)
        print_metrics(results[sid]['label'], results[sid]['metrics'])

    # ── Step 5: Generate figures ───────────────────────────────────────────
    print('\n[5/5] Generating figures...')
    plot_observed_vs_predicted(results, FIG_DIR)
    plot_scatter(results, FIG_DIR)
    plot_performance_summary(results, FIG_DIR)

    print('\n' + '=' * 60)
    print('Pipeline complete.')
    print(f'  Model  -> {MODEL_PATH}')
    print(f'  Figures -> {FIG_DIR}/')
    print('=' * 60)
