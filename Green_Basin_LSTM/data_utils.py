"""
utils/data_utils.py

Data acquisition and preparation helpers for the Green Basin LSTM pipeline.
Handles USGS NWIS streamflow retrieval, Daymet climate data retrieval,
merging into HydroDF CSVs, and train/val/test splitting.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from dataretrieval import nwis
import pydaymet as daymet


# ── Site metadata ─────────────────────────────────────────────────────────────

# All four gauges confirmed unregulated — no major dams upstream.
# Test site (09251000) is the same river as Train 3 but downstream,
# adding the Little Snake River confluence and testing spatial generalization.
SITES = {
    '09217000': {'name': 'Green_R_nr_Green_River_WY',       'role': 'Train 1', 'lat': 41.527, 'lon': -109.466},
    '09306500': {'name': 'White_R_nr_Watson_UT',             'role': 'Train 2', 'lat': 39.979, 'lon': -109.179},
    '09239500': {'name': 'Yampa_R_at_Steamboat_Springs_CO', 'role': 'Train 3', 'lat': 40.484, 'lon': -106.831},
    '09251000': {'name': 'Yampa_R_at_Deerlodge_Park_CO',    'role': 'Test',    'lat': 40.468, 'lon': -108.534},
}

TRAIN_IDS = ['09217000', '09306500', '09239500']
TEST_ID   = '09251000'

# Daymet variables to retrieve
DAYMET_VARS = ['prcp', 'tmax', 'swe']

# Column name constants — match the professor's HydroDF convention
DATE_COL   = 'Date'
TARGET_COL = 'flow_cms'
FEATURE_COLS = ['prcp_mm_day', 'tmax_degC', 'swe_cm']


# ── Streamflow fetch ──────────────────────────────────────────────────────────

def fetch_streamflow(site_id: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily mean streamflow (parameter 00060) from USGS NWIS for one gauge.
    Converts cfs → cms to match the professor's flow_cms convention.

    Returns
    -------
    DataFrame with columns [Date, site_no, flow_cms]
    """
    print(f'  [NWIS] {site_id}', end=' ... ')

    # Pull daily values: 00060 = mean discharge in cfs
    raw, _ = nwis.get_dv(sites=site_id, parameterCd='00060', start=start, end=end)

    df = raw[['00060_Mean']].rename(columns={'00060_Mean': 'flow_cfs'})
    df.index.name = DATE_COL
    df = df.reset_index()

    # Strip timezone info from datetime index
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.tz_localize(None)

    # Convert cfs → cms (1 cfs = 0.0283168 cms)
    df['flow_cms'] = df['flow_cfs'] * 0.0283168
    df = df.drop(columns=['flow_cfs'])

    df.insert(1, 'site_no', site_id)

    # Drop days with missing discharge
    df = df.dropna(subset=['flow_cms'])

    print(f'{len(df):,} records ({df[DATE_COL].min().date()} → {df[DATE_COL].max().date()})')
    return df


# ── Daymet fetch ──────────────────────────────────────────────────────────────

def fetch_daymet(site_id: str, lat: float, lon: float,
                 start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily Daymet climate variables (prcp, tmax, swe) for a single point.

    Returns
    -------
    DataFrame with columns [Date, prcp_mm_day, tmax_degC, swe_cm]
    """
    print(f'  [Daymet] {site_id}', end=' ... ')

    # pydaymet.get_bycoords returns an xarray Dataset for a single point
    ds = daymet.get_bycoords(
        coords=(lon, lat),
        dates=(start, end),
        variables=DAYMET_VARS,
        crs='epsg:4326',
    )

    df = ds.to_dataframe().reset_index()

    # Keep only the columns we need and rename to descriptive names
    df = df[['time'] + DAYMET_VARS].rename(columns={
        'time': DATE_COL,
        'prcp': 'prcp_mm_day',
        'tmax': 'tmax_degC',
        'swe':  'swe_cm',
    })

    # Ensure timezone-naive datetime
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.tz_localize(None)

    print(f'{len(df):,} records')
    return df


# ── Merge and save ────────────────────────────────────────────────────────────

def build_hydrodf(site_id: str, start: str, end: str, data_dir: str) -> pd.DataFrame:
    """
    Fetch streamflow + Daymet for one site, merge on Date, fill gaps,
    and save to data/HydroDF/ as a CSV.

    Follows the professor's HydroDF naming convention:
        HydroDF_{SiteName}_{SiteID}.csv

    Returns the merged DataFrame.
    """
    info = SITES[site_id]
    out_path = os.path.join(data_dir, f'HydroDF_{info["name"]}_{site_id}.csv')

    # Return cached file if it already exists — avoids unnecessary API calls
    if os.path.exists(out_path):
        print(f'  [cache] {site_id} — loading {os.path.basename(out_path)}')
        return pd.read_csv(out_path, parse_dates=[DATE_COL])

    # Fetch streamflow and Daymet
    flow_df    = fetch_streamflow(site_id, start, end)
    climate_df = fetch_daymet(site_id, info['lat'], info['lon'], start, end)

    # Inner join on Date — keep only days where both sources have data
    merged = pd.merge(flow_df, climate_df, on=DATE_COL, how='inner')

    # Fill any remaining gaps with linear interpolation + forward/backward fill
    # (matches the professor's missing-value approach in Hydro_LSTM.ipynb)
    cols_to_fill = FEATURE_COLS + [TARGET_COL]
    n_missing = merged[cols_to_fill].isnull().sum().sum()
    if n_missing > 0:
        print(f'  Filling {n_missing} missing values via interpolation')
        merged[cols_to_fill] = (
            merged[cols_to_fill]
            .interpolate(method='linear', limit_direction='both')
            .ffill()
            .bfill()
        )

    merged.to_csv(out_path, index=False)
    print(f'  Saved → {os.path.basename(out_path)}  ({len(merged):,} rows)')
    return merged


def load_hydrodf(site_id: str, data_dir: str) -> pd.DataFrame:
    """
    Load a saved HydroDF CSV for one site by matching the site ID in the filename.
    Raises a clear error if the file is not found.
    """
    matches = [f for f in os.listdir(data_dir) if site_id in f and f.endswith('.csv')]
    if not matches:
        raise FileNotFoundError(
            f'No HydroDF CSV found for {site_id} in {data_dir}. '
            f'Run data_acquisition.py first.'
        )
    return pd.read_csv(os.path.join(data_dir, matches[0]), parse_dates=[DATE_COL])


# ── Train / val / test split ──────────────────────────────────────────────────

def split_by_year(df: pd.DataFrame,
                  train_end: int, val_start: int, val_end: int,
                  test_start: int, test_end: int):
    """
    Split a HydroDF DataFrame into train, val, and test subsets by calendar year.
    Mirrors the professor's year-based split approach exactly.

    Returns
    -------
    train_df, val_df, test_df : DataFrames for each period
    """
    df = df.copy()
    df['year'] = df[DATE_COL].dt.year

    train_df = df[df['year'] <= train_end].copy()
    val_df   = df[(df['year'] >= val_start) & (df['year'] <= val_end)].copy()
    test_df  = df[(df['year'] >= test_start) & (df['year'] <= test_end)].copy()

    print(f'  Train rows: {len(train_df):,}  |  '
          f'Val rows: {len(val_df):,}  |  '
          f'Test rows: {len(test_df):,}')

    return train_df, val_df, test_df
