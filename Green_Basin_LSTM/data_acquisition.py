"""
data_acquisition.py  —  SETUP SCRIPT (run once before lstm_model.py)
=====================================================================
Fetches daily streamflow from USGS NWIS and climate data from Daymet
for all four study sites. Saves one CSV per site to data/HydroDF/.

After running this once, data is cached to disk. Re-running will load
from cache without making any API calls — making this fully reproducible.

Usage
-----
    conda activate torch310env
    python data_acquisition.py

    Then run the main pipeline:
    python lstm_model.py

Output files
------------
    data/HydroDF/HydroDF_Green_R_nr_Green_River_WY_09217000.csv
    data/HydroDF/HydroDF_White_R_nr_Watson_UT_09306500.csv
    data/HydroDF/HydroDF_Yampa_R_at_Steamboat_Springs_CO_09239500.csv
    data/HydroDF/HydroDF_Yampa_R_at_Deerlodge_Park_CO_09251000.csv
    figures/fig_eda_timeseries.png
"""

import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from notebooks.data_utils import (
    SITES, DATE_COL, TARGET_COL,
    build_hydrodf,
)

# ── Configuration ─────────────────────────────────────────────────────────────

START_DATE = '1990-01-01'
END_DATE   = '2023-12-31'

DATA_DIR = os.path.join('data', 'HydroDF')
FIG_DIR  = 'figures'

# Create output directories if they do not exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('Green Basin LSTM — Data Acquisition (setup)')
    print(f'Date range: {START_DATE} → {END_DATE}')
    print('=' * 60)

    # Fetch or load from cache for all four sites
    hydro_dfs = {}
    for site_id, info in SITES.items():
        print(f'\n[{info["role"]}] {site_id} — {info["name"].replace("_", " ")}')
        hydro_dfs[site_id] = build_hydrodf(site_id, START_DATE, END_DATE, DATA_DIR)

    # ── EDA figure — visual check of all four time series ─────────────────────
    print('\nGenerating EDA figure...')

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

    print('\nSetup complete. Now run: python lstm_model.py')
    print('=' * 60)


if __name__ == '__main__':
    main()
