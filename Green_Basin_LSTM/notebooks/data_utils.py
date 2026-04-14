"""
notebooks/data_utils.py

Data helpers for the Green Basin LSTM pipeline.
Handles USGS NWIS streamflow retrieval, Daymet climate retrieval,
merging into HydroDF CSVs, and train/val/test splitting.

All functions are called by data_acquisition.py and lstm_model.py.
Do not run this file directly.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from dataretrieval import nwis
import pydaymet as daymet


# ── Site definitions ──────────────────────────────────────────────────────────
# All four gauges are confirmed unregulated — no major dams upstream.
# The test site (09251000) is on the same river as Train 3 (Yampa) but
# downstream, adding the Little Snake River confluence. The model never
# sees this site during training, testing spatial generalization.

SITES = {
    '09217000': {'name': 'Green_R_nr_Green_River_WY',        'role': 'Train 1', 'lat': 41.527, 'lon': -109.466},
    '09306500': {'name': 'White_R_nr_Watson_UT',              'role': 'Train 2', 'lat': 39.979, 'lon': -109.179},
    '09239500': {'name': 'Yampa_R_at_Steamboat_Springs_CO',  'role': 'Train 3', 'lat': 40.484, 'lon': -106.831},
    '09251000': {'name': 'Yampa_R_at_Deerlodge_Park_CO',     'role': 'Test',    'lat': 40.468, 'lon': -108.534},
}

TRAIN_IDS    = ['09217000', '09306500', '09239500']
TEST_ID      = '09251000'
DAYMET_VARS  = ['prcp', 'tmax', 'swe']

# Column names — match the professor's HydroDF convention
DATE_COL     = 'Date'
TARGET_COL   = 'flow_cms'
FEATURE_COLS = ['prcp_mm_day', 'tmax_degC', 'swe_cm']


# ── Streamflow ────────────────────────────────────────────────────────────────

def fetch_streamflow(site_id: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily mean streamflow from USGS NWIS for one gauge.
    Parameter 00060 = discharge in cfs. Converts to cms for the professor's
    flow_cms convention (1 cfs = 0.0283168 cms).

    Returns DataFrame with columns: [Date, site_no, flow_cms]
    """
    print(f'  [NWIS] {site_id}', end=' ... ')

    raw, _ = nwis.get_dv(sites=site_id, parameterCd='00060', start=start, end=end)

    df = raw[['00060_Mean']].rename(columns={'00060_Mean': 'flow_cfs'})
    df.index.name = DATE_COL
    df = df.reset_index()

    # Remove timezone info so dates merge cleanly with Daymet dates
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.tz_localize(None)

    # Convert cfs → cms
    df['flow_cms'] = df['flow_cfs'] * 0.0283168
    df = df.drop(columns=['flow_cfs'])
    df.insert(1, 'site_no', site_id)
    df = df.dropna(subset=['flow_cms'])

    print(f'{len(df):,} records ({df[DATE_COL].min().date()} → {df[DATE_COL].max().date()})')
    return df


# ── Daymet climate ────────────────────────────────────────────────────────────

def fetch_daymet(site_id: str, lat: float, lon: float,
                 start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily Daymet climate variables for a single point location.
    Variables: prcp (mm/day), tmax (°C), swe (kg/m2 ≈ mm, treated as cm here).

    pydaymet.get_bycoords returns a pandas DataFrame with the date as the index
    and column names that include units e.g. 'prcp (mm/day)'.
    We rename to clean column names matching FEATURE_COLS.

    Returns DataFrame with columns: [Date, prcp_mm_day, tmax_degC, swe_cm]
    """
    print(f'  [Daymet] {site_id}', end=' ... ')

    # get_bycoords returns a DataFrame — date is the index
    df = daymet.get_bycoords(
        coords=(lon, lat),
        dates=(start, end),
        variables=DAYMET_VARS,
        crs='epsg:4326',
    )

    # Bring the date index out as a plain column
    df = df.reset_index()

    # pydaymet column names include units — find and rename them robustly
    # regardless of whether the column is named 'time', 'Date', or 'index'
    rename = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'time' in col_lower or col_lower == 'index':
            rename[col] = DATE_COL
        elif 'prcp' in col_lower:
            rename[col] = 'prcp_mm_day'
        elif 'tmax' in col_lower:
            rename[col] = 'tmax_degC'
        elif 'swe' in col_lower:
            rename[col] = 'swe_cm'
    df = df.rename(columns=rename)

    # Keep only the columns we need
    df = df[[DATE_COL, 'prcp_mm_day', 'tmax_degC', 'swe_cm']]

    # Ensure timezone-naive datetime so dates merge cleanly with NWIS dates
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.tz_localize(None)

    print(f'{len(df):,} records')
    return df


# ── Merge and save ────────────────────────────────────────────────────────────

def build_hydrodf(site_id: str, start: str, end: str, data_dir: str) -> pd.DataFrame:
    """
    Fetch streamflow + Daymet for one site, merge on Date, fill gaps,
    and save to data/HydroDF/ as a CSV.

    File is named HydroDF_{SiteName}_{SiteID}.csv to match the professor's
    HydroDF naming convention from Hydro_LSTM.ipynb.

    If the CSV already exists it is loaded from disk — no API calls are made.
    This makes re-runs fast and the pipeline reproducible.

    Returns the merged DataFrame.
    """
    info     = SITES[site_id]
    out_path = os.path.join(data_dir, f'HydroDF_{info["name"]}_{site_id}.csv')

    # Load from cache if already fetched
    if os.path.exists(out_path):
        print(f'  [cache] {site_id} — {os.path.basename(out_path)}')
        return pd.read_csv(out_path, parse_dates=[DATE_COL])

    # Fetch both sources
    flow_df    = fetch_streamflow(site_id, start, end)
    climate_df = fetch_daymet(site_id, info['lat'], info['lon'], start, end)

    # Inner join on Date — only keep days where both sources have data
    merged = pd.merge(flow_df, climate_df, on=DATE_COL, how='inner')

    # Fill any remaining gaps — matches the professor's approach in Hydro_LSTM.ipynb
    cols = FEATURE_COLS + [TARGET_COL]
    if merged[cols].isnull().sum().sum() > 0:
        merged[cols] = (
            merged[cols]
            .interpolate(method='linear', limit_direction='both')
            .ffill()
            .bfill()
        )

    merged.to_csv(out_path, index=False)
    print(f'  Saved → {os.path.basename(out_path)}  ({len(merged):,} rows)')
    return merged


def load_hydrodf(site_id: str, data_dir: str) -> pd.DataFrame:
    """
    Load a saved HydroDF CSV for one site.
    Finds the file by matching the site ID in the filename.
    Raises a clear error if data_acquisition.py has not been run yet.
    """
    matches = [f for f in os.listdir(data_dir) if site_id in f and f.endswith('.csv')]
    if not matches:
        raise FileNotFoundError(
            f'No HydroDF CSV found for {site_id} in {data_dir}.\n'
            f'Run data_acquisition.py first.'
        )
    return pd.read_csv(os.path.join(data_dir, matches[0]), parse_dates=[DATE_COL])


# ── Year-based split ──────────────────────────────────────────────────────────

def split_by_year(df: pd.DataFrame,
                  train_end: int, val_start: int, val_end: int,
                  test_start: int, test_end: int):
    """
    Split a HydroDF DataFrame into train, val, and test by calendar year.
    Mirrors the professor's split approach from Hydro_LSTM.ipynb exactly.

    Returns train_df, val_df, test_df.
    """
    df = df.copy()
    df['year'] = df[DATE_COL].dt.year

    train_df = df[df['year'] <= train_end].copy()
    val_df   = df[(df['year'] >= val_start)  & (df['year'] <= val_end)].copy()
    test_df  = df[(df['year'] >= test_start) & (df['year'] <= test_end)].copy()

    print(f'  Train: {len(train_df):,} rows  |  '
          f'Val: {len(val_df):,} rows  |  '
          f'Test: {len(test_df):,} rows')

    return train_df, val_df, test_df
