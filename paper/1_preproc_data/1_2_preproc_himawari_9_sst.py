import xarray as xr
import pandas as pd
import numpy as np
import os

from inversion_sst_gp import utils

# Configuration
# Define geographical and temporal boundaries for data processing
LON_LIMITS = (115, 118)
LAT_LIMITS = (-15.5, -12.5)  # Note: xarray sel uses [min, max], so flip for slicing
TIME_STEP_SECONDS = 3600  # Time step in seconds for derivative calculations
TIME_STR_LIST = ["2023-09-22T04:00:00", "2023-12-18T01:00:00"]
IGNORE_NAN = False  # Whether to ignore NaN values in calculations

# Define data directories
NON_PROCESSED_DIR = "1_preproc_data/non_proc_data/himawari"
PROCESSED_DIR = "1_preproc_data/proc_data"

# Himawari product name for file path construction
PRODUCT_NAME = "STAR-L3C_GHRSST-SSTsubskin-AHI_H09-ACSPO_V2.90-v02.0-fv01.0"


# Helper functions
def get_himawari_file_path(time_str):
    """Constructs the file path for a Himawari NetCDF file."""
    time_id = pd.to_datetime(time_str).strftime("%Y%m%d%H%M%S")
    return f"{NON_PROCESSED_DIR}/{time_id}-{PRODUCT_NAME}.nc"


def load_and_preprocess_sst(file_path, target_time):
    """Loads SST data, selects time/region, flips latitude, and converts to Celsius."""
    with xr.open_dataset(file_path, decode_timedelta=False) as ds:
        ds_sel = ds.sel(
            time=target_time,
            lon=slice(LON_LIMITS[0], LON_LIMITS[1]),
            lat=slice(LAT_LIMITS[1], LAT_LIMITS[0]),
        )

        lon = ds_sel.lon.values
        lat = ds_sel.lat.values
        sst = ds_sel.sea_surface_temperature.values

        lat = np.flip(lat)
        sst = np.flip(sst, axis=0)

        sst_celsius = sst - 273.15
    return lon, lat, sst_celsius


def get_sst_time_series_data(time_str):
    """Retrieves SST data for the current, previous, and next time steps."""
    current_time = np.datetime64(time_str)
    previous_time = current_time - np.timedelta64(TIME_STEP_SECONDS, "s")
    next_time = current_time + np.timedelta64(TIME_STEP_SECONDS, "s")

    file_curr = get_himawari_file_path(time_str)
    file_prev = get_himawari_file_path(str(previous_time))
    file_next = get_himawari_file_path(str(next_time))

    lon, lat, T_curr = load_and_preprocess_sst(file_curr, current_time)
    _, _, T_prev = load_and_preprocess_sst(file_prev, previous_time)
    _, _, T_next = load_and_preprocess_sst(file_next, next_time)

    return lon, lat, T_curr, T_prev, T_next, current_time


# Main Processing Function
def process_himawari_sst_data(time_str):
    """Processes Himawari SST data and calculates temporal/spatial gradients."""
    print(f"  Starting processing for {time_str}")
    lon, lat, T_curr, T_prev, T_next, current_time = get_sst_time_series_data(time_str)

    # Apply 3x3 window averaging to reduce resolution and noise
    lon_resampled = utils.calculate_mean_window_1d(lon, 3)
    lat_resampled = utils.calculate_mean_window_1d(lat, 3)
    T_curr_resampled = utils.calculate_mean_window_2d(T_curr, 3, 3, ignore_nan=IGNORE_NAN)
    T_prev_resampled = utils.calculate_mean_window_2d(T_prev, 3, 3, ignore_nan=IGNORE_NAN)
    T_next_resampled = utils.calculate_mean_window_2d(T_next, 3, 3, ignore_nan=IGNORE_NAN)

    # Calculate temporal derivative (dT/dt)
    dTdt = (T_next_resampled - T_prev_resampled) / (2 * TIME_STEP_SECONDS)

    # Calculate grid properties and spatial derivatives (dT/dx, dT/dy)
    lonc, latc, X, Y, LON, LAT = utils.calculate_grid_properties(
        lon_resampled, lat_resampled
    )
    dTdx, dTdy = utils.finite_difference_2d(X, Y, T_curr_resampled)
    
    # Extend time dimension
    T_curr_resampled = T_curr_resampled[np.newaxis, :, :]
    dTdt = dTdt[np.newaxis, :, :]
    dTdx = dTdx[np.newaxis, :, :]
    dTdy = dTdy[np.newaxis, :, :]

    # Create an xarray Dataset to store all processed variables
    ds = xr.Dataset(
        coords={
            "lon": (["lon"], lon_resampled),
            "lat": (["lat"], lat_resampled),
            "LON": (["lat", "lon"], LON),
            "LAT": (["lat", "lon"], LAT),
            "X": (["lat", "lon"], X),
            "Y": (["lat", "lon"], Y),
            "lonc": lonc,
            "latc": latc,
            "time": (["time"], [current_time]),
            "time_step": TIME_STEP_SECONDS,
        },
        data_vars={
            "T": (["time","lat", "lon"], T_curr_resampled),
            "dTdt": (["time","lat", "lon"], dTdt),
            "dTdx": (["time","lat", "lon"], dTdx),
            "dTdy": (["time","lat", "lon"], dTdy),
        },
    )
    print(f"  Finished processing for {time_str}")
    return ds


# Main Execution
if __name__ == "__main__":
    print("--- Starting preprocessing Himawari-9 SST ---")

    print("Loading Himawari SST data")
    ds_list = []
    for i, time_str in enumerate(TIME_STR_LIST):
        processed_ds = process_himawari_sst_data(time_str)
        ds_list.append(processed_ds)
    
    print("Merging processed datasets and saving")
    # Merge by time dimension
    if ds_list:
        ds_combined = xr.concat(ds_list, dim="time")
        output_file_path = f"{PROCESSED_DIR}/himawari.nc"
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        ds_combined.to_netcdf(output_file_path)
        print(f"Combined Himawari SST data saved to {output_file_path}")
    else:
        print("No datasets were processed to merge.")

    print("Data processing complete")