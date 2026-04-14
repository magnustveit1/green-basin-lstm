"""
utils/train_utils.py

Training loop, evaluation metrics, and figure generation helpers
for the Green Basin LSTM pipeline.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn

from utils import LSTM_helper


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, device,
                epochs: int, learning_rate: float, patience: int):
    """
    Train an LSTMRegressor with early stopping based on validation loss.
    Follows the professor's training loop from Hydro_LSTM.ipynb exactly.

    Parameters
    ----------
    model        : LSTMRegressor instance (already moved to device)
    train_loader : DataLoader for training sequences
    val_loader   : DataLoader for validation sequences
    device       : torch.device (cpu or cuda)
    epochs       : maximum number of training epochs
    learning_rate: Adam optimizer step size
    patience     : number of epochs without val improvement before stopping

    Returns
    -------
    model   : model loaded with best weights from training
    history : dict with 'train_loss' and 'val_loss' lists
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss  = np.inf          # tracks best validation loss seen
    best_state     = None            # stores best model weights
    patience_counter = 0             # counts epochs without improvement
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, epochs + 1):
        # ── Training pass ────────────────────────────────────────────────────
        model.train()                # enable dropout / batch norm training behaviour
        batch_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()    # clear gradients from previous step
            pred = model(xb)         # forward pass
            loss = criterion(pred, yb)
            loss.backward()          # backpropagate gradients
            optimizer.step()         # update weights
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))

        # ── Validation pass ───────────────────────────────────────────────────
        val_loss, _, _ = LSTM_helper.evaluate(model, criterion, device, val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f'Epoch {epoch:03d} | train loss = {train_loss:.5f} | val loss = {val_loss:.5f}')

        # ── Early stopping check ──────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            # Save a CPU copy of the best weights so we can restore later
            best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch}.')
                break

    # Restore the best model weights before returning
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    """
    Compute RMSE, MAE, R², and NSE between observed and predicted arrays.
    All values are in the original (unscaled) units passed in.

    NSE (Nash-Sutcliffe Efficiency):
        1.0 = perfect prediction
        0.0 = model no better than predicting the mean
        < 0  = model worse than the mean baseline
    """
    obs  = np.array(obs,  dtype=float)
    pred = np.array(pred, dtype=float)

    rmse = math.sqrt(mean_squared_error(obs, pred))
    mae  = mean_absolute_error(obs, pred)
    r2   = r2_score(obs, pred)
    nse  = 1.0 - np.sum((obs - pred) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

    return {'RMSE': round(rmse, 4), 'MAE': round(mae, 4),
            'R2': round(r2, 4), 'NSE': round(nse, 4)}


def print_metrics(site_label: str, metrics: dict):
    """Print a formatted one-line metrics summary for one site."""
    print(
        f'  {site_label:<50} '
        f'RMSE={metrics["RMSE"]:.4f}  '
        f'MAE={metrics["MAE"]:.4f}  '
        f'R²={metrics["R2"]:.4f}  '
        f'NSE={metrics["NSE"]:.4f}'
    )


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_training_history(history: dict, fig_dir: str):
    """
    Plot train vs. validation MSE loss by epoch.
    Marks the best epoch with a vertical dashed line.
    Matches the professor's training history figure style.
    """
    best_ep = int(np.argmin(history['val_loss'])) + 1

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history['train_loss'], label='Train loss',      color='steelblue', lw=1.5)
    ax.plot(history['val_loss'],   label='Validation loss', color='tomato',    lw=1.5)
    ax.axvline(best_ep - 1, color='gray', lw=1, ls='--', label=f'Best epoch ({best_ep})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training History', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = os.path.join(fig_dir, 'fig_training_history.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


def plot_observed_vs_predicted(results: dict, fig_dir: str):
    """
    Four-panel time series of observed vs. predicted streamflow for all sites.
    Test site panel is highlighted with a distinct background.
    Matches the professor's time-series figure style.

    Parameters
    ----------
    results : dict keyed by site_id, each value has:
              {'dates', 'obs_cms', 'pred_cms', 'label', 'role', 'metrics'}
    """
    site_ids = list(results.keys())
    fig, axes = plt.subplots(len(site_ids), 1, figsize=(13, 3 * len(site_ids)), sharex=False)
    if len(site_ids) == 1:
        axes = [axes]

    fig.suptitle('Observed vs. Predicted Streamflow — Test Period (2019–2023)',
                 fontsize=13, fontweight='bold', y=1.01)

    for ax, sid in zip(axes, site_ids):
        r = results[sid]
        m = r['metrics']

        ax.plot(r['dates'], r['obs_cms'],  color='#1f77b4', lw=0.8, alpha=0.9, label='Observed')
        ax.plot(r['dates'], r['pred_cms'], color='#ff7f0e', lw=0.8, alpha=0.9, label='Predicted')

        # Annotate metrics in top-right corner of each panel
        ax.text(0.99, 0.96,
                f'RMSE={m["RMSE"]:.3f} cms  MAE={m["MAE"]:.3f} cms  '
                f'R²={m["R2"]:.3f}  NSE={m["NSE"]:.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

        # Highlight test site with a distinct background
        if r['role'] == 'Test':
            ax.set_facecolor('#fff3f3')
            ax.set_title(f'{r["label"]}  ★ TEST SITE — unseen during training',
                         fontsize=9, loc='left', fontweight='bold')
        else:
            ax.set_title(r['label'], fontsize=9, loc='left')

        ax.set_ylabel('Streamflow (cms)', fontsize=9)
        ax.legend(loc='upper left', fontsize=8, framealpha=0.8)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, lw=0.5)

    axes[-1].set_xlabel('Date', fontsize=10)
    fig.subplots_adjust(hspace=0.65)

    out = os.path.join(fig_dir, 'fig_observed_vs_predicted.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


def plot_scatter(results: dict, fig_dir: str):
    """
    One scatter panel per site: observed vs. predicted streamflow.
    Includes 1:1 line. Matches the professor's scatter figure style.
    """
    site_ids = list(results.keys())
    fig, axes = plt.subplots(1, len(site_ids), figsize=(5 * len(site_ids), 5))
    if len(site_ids) == 1:
        axes = [axes]

    fig.suptitle('Observed vs. Predicted Streamflow — Scatter',
                 fontsize=12, fontweight='bold')

    for ax, sid in zip(axes, site_ids):
        r = results[sid]
        m = r['metrics']

        ax.scatter(r['obs_cms'], r['pred_cms'], alpha=0.35, s=6,
                   color='#d62728' if r['role'] == 'Test' else '#1f77b4')

        # 1:1 reference line
        lims = [min(r['obs_cms'].min(), r['pred_cms'].min()),
                max(r['obs_cms'].max(), r['pred_cms'].max())]
        ax.plot(lims, lims, 'k--', lw=1)

        ax.set_xlabel('Observed (cms)', fontsize=9)
        ax.set_ylabel('Predicted (cms)', fontsize=9)
        title = r['label'].split('(')[0].strip()
        ax.set_title(f'{title}\nNSE={m["NSE"]:.3f}  R²={m["R2"]:.3f}', fontsize=8.5)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    out = os.path.join(fig_dir, 'fig_scatter.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


def plot_performance_summary(results: dict, fig_dir: str):
    """
    Three-panel bar chart of RMSE, MAE, and NSE across all sites.
    Test site bar is highlighted in red.
    """
    site_ids = list(results.keys())
    labels   = [results[s]['label'].split('(')[0].strip().replace(' ', '\n') for s in site_ids]
    colors   = ['#d62728' if results[s]['role'] == 'Test' else '#4878cf' for s in site_ids]

    rmse_vals = [results[s]['metrics']['RMSE'] for s in site_ids]
    mae_vals  = [results[s]['metrics']['MAE']  for s in site_ids]
    nse_vals  = [results[s]['metrics']['NSE']  for s in site_ids]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Performance Summary — All Sites (Test Period 2019–2023)',
                 fontsize=12, fontweight='bold')

    for ax, vals, ylabel, title in zip(
            axes,
            [rmse_vals, mae_vals, nse_vals],
            ['RMSE (cms)', 'MAE (cms)', 'NSE'],
            ['Root Mean Squared Error', 'Mean Absolute Error', 'Nash-Sutcliffe Efficiency']):

        bars = ax.bar(labels, vals, color=colors, edgecolor='white')
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(labelsize=8)

        # Label each bar with its numeric value
        for bar, v in zip(bars, vals):
            ypos = bar.get_height() + max(np.abs(vals)) * 0.015
            ax.text(bar.get_x() + bar.get_width() / 2, ypos, f'{v:.3f}',
                    ha='center', fontsize=8)

    # Add NSE=0 reference line (mean-flow baseline)
    axes[2].axhline(0, color='black', lw=0.8, ls='--')

    fig.tight_layout()

    out = os.path.join(fig_dir, 'fig_performance_summary.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')
