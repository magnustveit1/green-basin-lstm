"""
data_acquisition.py

Fetches daily streamflow (USGS NWIS) and climate data (Daymet) for all four
study sites and saves one HydroDF CSV per site to data/HydroDF/.

Run this script first before lstm_model.py.

Usage
-----
    conda activate torch310env
    python data_acquisition.py

Sites
-----
    09217000 — Green River nr Green River, WY       (Train 1)
    09306500 — White River nr Watson, UT            (Train 2)
    09239500 — Yampa River at Steamboat Springs, CO (Train 3)
    09251000 — Yampa River at Deerlodge Park, CO    (Test — unseen during training)

Outputs
-------
    data/HydroDF/HydroDF_{SiteName}_{SiteID}.csv   (one per site)
    figures/fig_eda_timeseries.png
"""

import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from utils.data_utils import (
    SITES, TRAIN_IDS, TEST_ID,
    DATE_COL, TARGET_COL,
    build_hydrodf,
)

# ── Configuration ─────────────────────────────────────────────────────────────

START_DATE = '1990-01-01'
END_DATE   = '2023-12-31'

DATA_DIR = os.path.join('data', 'HydroDF')
FIG_DIR  = 'figures'

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)


# ── Fetch all sites ───────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('Green Basin LSTM — Data Acquisition')
    print(f'Date range: {START_DATE} → {END_DATE}')
    print('=' * 60)

    hydro_dfs = {}
    for site_id in SITES:
        print(f'\n[{SITES[site_id]["role"]}] {site_id}')
        hydro_dfs[site_id] = build_hydrodf(site_id, START_DATE, END_DATE, DATA_DIR)

    # ── EDA figure ────────────────────────────────────────────────────────────
    print('\nGenerating EDA time series figure...')

    short_labels = {
        '09217000': 'Green R. nr Green River, WY\n(Train 1 — snowmelt headwater)',
        '09306500': 'White R. nr Watson, UT\n(Train 2 — plateau tributary)',
        '09239500': 'Yampa R. at Steamboat Springs, CO\n(Train 3 — upper free-flowing)',
        '09251000': 'Yampa R. at Deerlodge Park, CO\n(Test — downstream, unseen)',
    }
    colors = ['#1f77b4', '#2ca02c', '#9467bd', '#d62728']

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Daily Mean Streamflow — Study Sites (1990–2023)',
                 fontsize=13, fontweight='bold', y=1.01)

    for ax, sid, color in zip(axes, SITES.keys(), colors):
        df = hydro_dfs[sid]
        ax.plot(df[DATE_COL], df[TARGET_COL], color=color, lw=0.6, alpha=0.85)
        ax.set_title(short_labels[sid], fontsize=9, loc='left', pad=3)
        ax.set_ylabel('Flow (cms)', fontsize=9)
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, lw=0.5)

    axes[-1].set_xlabel('Date', fontsize=10)
    fig.subplots_adjust(hspace=0.60)

    out = os.path.join(FIG_DIR, 'fig_eda_timeseries.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

    print('\nData acquisition complete. Run lstm_model.py next.')
    print('=' * 60)


if __name__ == '__main__':
    main()
