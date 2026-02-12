"""
Himawari SST download tools.

This module provides functionality for downloading and processing Himawari satellite
data, specifically Sea Surface Temperature (SST) data. It includes functions for
downloading data from the Himawari satellite, processing the data, and preparing
it for further analysis or modeling.


"""

import os
import xarray as xr
import pandas as pd
import numpy as np
import time

import earthaccess
from pyproj import Proj

# h5netcdf

from inversion_sst_gp import utils

# Configuration
# Define geographical and temporal boundaries for data processing
# LON_LIMITS = (115, 118)
# LAT_LIMITS = (-15.5, -12.5)  # Note: xarray sel uses [min, max], so flip for slicing
TIME_STEP_SECONDS = 3600  # Time step in seconds for derivative calculations
# TIME_STR_LIST = ["2023-09-22T04:00:00", "2023-12-18T01:00:00"]
IGNORE_NAN = False  # Whether to ignore NaN values in calculations

# # Define data directories
# NON_PROCESSED_DIR = "1_preproc_data/non_proc_data/himawari"
# PROCESSED_DIR = "1_preproc_data/proc_data"

# # Himawari product name for file path construction
# PRODUCT_NAME = "STAR-L3C_GHRSST-SSTsubskin-AHI_H09-ACSPO_V2.90-v02.0-fv01.0"


# Download functions

def get_timesteps(time_str):
    current_time = np.datetime64(time_str)
    previous_time = str(current_time - np.timedelta64(TIME_STEP_SECONDS, "s"))
    next_time = str(current_time + np.timedelta64(TIME_STEP_SECONDS, "s"))
    return [previous_time, time_str, next_time]


def get_str_timesteps(time_str):
    return [str(t) for t in get_timesteps(time_str)]


def calculate_latlon_xr(ds):
    """Calculate latitude and longitude from a geostationary dataset."""
    # ONLY NEEDED FOR L2P DATASET
    x = ds['ni'].values
    y = ds['nj'].values

    geo_var = ds['geostationary']

    lon_0 = float(geo_var.longitude_of_projection_origin)
    h     = float(geo_var.perspective_point_height)
    sweep = str(geo_var.sweep_angle_axis)  

    p = Proj(proj='geos', h=h, lon_0=lon_0, sweep=sweep, datum='WGS84')

    X, Y = np.meshgrid(x*h, y*h)
    lon, lat = p(X, Y, inverse=True)

    # fill space pixels with NANs
    lon[np.abs(lon) > 360.0] = np.nan
    lat[np.abs(lat) > 90.0]  = np.nan

    return X, Y, lat, lon, x, y


def create_coords_dataset_xr(ds):
    """
    Add latitude, longitude, and geostationary coordinates to an existing xarray dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset to which coordinates will be added.

    Returns
    -------
    xarray.Dataset
        The input dataset with added latitude, longitude, and geostationary coordinates.
    """
    X, Y, lat, lon, x, y = calculate_latlon_xr(ds)

    # Add coordinates to the existing dataset
    ds = ds.assign_coords({
        'ni': ('ni', x),
        'nj': ('nj', y),
        'lat': (('nj', 'ni'), lat),
        'lon': (('nj', 'ni'), lon),
        'X': (('nj', 'ni'), X),
        'Y': (('nj', 'ni'), Y),
    })

    # Assign attributes
    ds['lat'].attrs.update(long_name='latitude', units='degrees_north')
    ds['lon'].attrs.update(long_name='longitude', units='degrees_east')
    ds['X'].attrs.update(long_name='geostationary X', units='m')
    ds['Y'].attrs.update(long_name='geostationary Y', units='m')

    return ds


def get_version_filename(version=9, level='L3C'):
    """Returns the short and long names for the specified Himawari data version and level."""
    if version == 8:
        short_name = f"AHI_H08-STAR-{level}-v2.70"
        long_name = f"STAR-{level}_GHRSST-SSTsubskin-AHI_H08-ACSPO_V2.70-v02.0-fv01.0"
    elif version == 9:
        short_name = f"H09-AHI-{level}-ACSPO-v2.90"
        long_name = f"STAR-{level}_GHRSST-SSTsubskin-AHI_H09-ACSPO_V2.90-v02.0-fv01.0"
    else:
        raise ValueError("Unsupported version.")
    return short_name, long_name


def get_sst_scene_nasa(time_str, savedir, version=9, level='L3C', overwrite=False):
    """Downloads Himawari SST data from NASA EarthData using EarthAccess.
    """

    # Get the time steps to download
    time_dl = get_timesteps(time_str)

    # Set the version filenames
    short_name, long_name = get_version_filename(version=version, level=level)

    get_sst_nasa(time_dl, savedir, short_name, long_name, overwrite=overwrite)



def get_sst_series_nasa(time_limits, savedir, version=9, level='L3C', overwrite=False):
    """Downloads Himawari SST data from NASA EarthData using EarthAccess.
    """
    
    # Create and hourly time series within the defined time limits
    time_del = np.timedelta64(TIME_STEP_SECONDS, "s")
    time_steps = np.arange(np.datetime64(time_limits[0]) - time_del, np.datetime64(time_limits[1]) + time_del, np.timedelta64(1, 'h'))

    # Get the time steps to download
    time_dl = [str(t) for t in time_steps]

    # Set the version filenames
    short_name, long_name = get_version_filename(version=version, level=level)
    
    get_sst_nasa(time_dl, savedir, short_name, long_name, overwrite=overwrite)



def get_sst_nasa(time_series, savedir, short_name, long_name, overwrite=False):

    # Authenticate with EarthAccess
    # For more information on how to authenticate, see:
    # https://earthaccess.readthedocs.io/en/stable/howto/authenticate/
    auth = earthaccess.login()
    print(f"EarthAccess authenticated: {auth.authenticated}")

    retries = 1
    wait_seconds = 5

    # flag = False
    for dt in time_series:
        dt_pd = pd.Timestamp(dt)
        time_str = dt_pd.strftime("%Y%m%d%H%M%S")
        file_name = f"{time_str}-{long_name}.nc"
        link = f"https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/{short_name}/{file_name}"
        file_path = os.path.join(savedir, file_name)
        
        # Check if file already exists
        if os.path.exists(file_path) and not overwrite:
            print(f"Skipping! File already exists and overwrite is False: {file_name}")
            continue

        attempts = 0
        for attempts in range(retries):
            try:
                earthaccess.download(link, savedir)
                ds = xr.open_dataset(file_path, decode_timedelta=False)
                ds.close()

                print(f"Downloaded and verified: {file_name}")
                break
            except Exception as e:
                print(f"[Retry] Attempt {attempts + 1}/{retries} failed: {e}")
                time.sleep(wait_seconds)
        
    return None


def crop_sst_scene_nasa(savedir, time_step, ll_box, version=9, level='L3C', file_app='_cropped', overwrite=False):
    """Crops Himawari SST data to the specified lat/lon box over a time series."""
    
    _, long_name = get_version_filename(version, level)
    
    dt_pd = pd.Timestamp(time_step)
    time_str = dt_pd.strftime("%Y%m%d%H%M%S")
    file_name = f"{time_str}-{long_name}.nc"
    full_name = os.path.join(savedir, file_name)
    
    if file_app != '':
        crop_file_name = f"{time_str}-{long_name}-{file_app}.nc"
    else:
        crop_file_name = f"{time_str}-{long_name}-temp.nc"
    full_crop_name = os.path.join(savedir, crop_file_name)

    if (not overwrite) & (os.path.exists(full_crop_name)):
        print(f"Cropped file already exists: {crop_file_name}")
    else:
        if os.path.exists(full_name):
            print(f"Cropping file for time {time_str}")
            
            # try:
            ds = xr.open_dataset(full_name, decode_times=False)
            if ds['lat'][0] > ds['lat'][-1]:
                ds_cropped = ds.sel(lon=slice(ll_box[0][0], ll_box[0][1]), lat=slice(ll_box[1][1], ll_box[1][0]))
            else:
                ds_cropped = ds.sel(lon=slice(ll_box[0][0], ll_box[0][1]), lat=slice(ll_box[1][0], ll_box[1][1]))
            
            ds_cropped.to_netcdf(full_crop_name)
            ds.close()
            
            # Remove the original file to save space
            if overwrite:
                os.remove(full_name)
                if file_app == '':
                    os.rename(full_crop_name, full_name)
                
            # except Exception as e:
            #     print(f"Error processing file {full_name}: {e}")
        else:
            print(f"File does not exist: {full_name}")
            
    return None


def crop_sst_series_nasa(savedir, time_limits, ll_box, version=9, level='L3C', file_app='_cropped', overwrite=False):
    """Crops Himawari SST data to the specified lat/lon box over a time series."""
    if file_app == '':
        print("No file_app provided, original files will be replaced.")
    if (file_app != '') & overwrite:
        print('Overwrite set to True: original files will be deleted!')
        
    # Create and hourly time series within the defined time limits
    time_del = np.timedelta64(TIME_STEP_SECONDS, "s")
    time_steps = np.arange(np.datetime64(time_limits[0]) - time_del, np.datetime64(time_limits[1]) + time_del, np.timedelta64(1, 'h'))

    for dt in time_steps:
        crop_sst_scene_nasa(savedir, dt, ll_box, version=version, level=level, file_app=file_app, overwrite=overwrite)
            
    return None


# Helper functions
def get_himawari_file_path(savedir, time_str, version=9, level='L3C', file_app=''):
    """Constructs the file path for a Himawari NetCDF file."""
    
    _, long_name = get_version_filename(version=version, level=level)
    
    time_id = pd.to_datetime(time_str).strftime("%Y%m%d%H%M%S")
    
    if file_app != '':
        return os.path.join(savedir, f"{time_id}-{long_name}-{file_app}.nc")
    else:
        return os.path.join(savedir, f"{time_id}-{long_name}.nc")


def load_and_preprocess_sst(file_path, target_time, ll_box):
    """Loads SST data, selects time/region, flips latitude, and converts to Celsius."""
    
    with xr.open_dataset(file_path, decode_timedelta=False) as ds:
        ds_sel = ds.sel(
            time=target_time,
            lon=slice(ll_box[0][0], ll_box[0][1]),
            lat=slice(ll_box[1][1], ll_box[1][0]),
        )

        lon = ds_sel.lon.values
        lat = ds_sel.lat.values
        sst = ds_sel.sea_surface_temperature.values

        lat = np.flip(lat)
        sst = np.flip(sst, axis=0)

        sst_celsius = sst - 273.15
    return lon, lat, sst_celsius


def get_sst_time_series_data(savedir, time_str, ll_box, version=9, level='L3C', file_app=''):
    """Retrieves SST data for the current, previous, and next time steps."""
    
    previous_time, current_time, next_time = get_str_timesteps(time_str)

    file_curr = get_himawari_file_path(savedir, time_str, version=version, level=level, file_app=file_app)
    file_prev = get_himawari_file_path(savedir, str(previous_time), version=version, level=level, file_app=file_app)
    file_next = get_himawari_file_path(savedir, str(next_time), version=version, level=level, file_app=file_app)
    
    # Check all 3 files exist
    fail = False
    for f in [file_curr, file_prev, file_next]:
        if not os.path.exists(f):
            # print(f"File does not exist: {f}")
            fail = True

    if not fail:
        lon, lat, T_curr = load_and_preprocess_sst(file_curr, current_time, ll_box)
        _, _, T_prev = load_and_preprocess_sst(file_prev, previous_time, ll_box)
        _, _, T_next = load_and_preprocess_sst(file_next, next_time, ll_box)
    else:
        lon, lat, T_curr, T_prev, T_next = None, None, None, None, None

    return lon, lat, T_curr, T_prev, T_next, current_time, fail


def create_processed_dataset(lon, lat, LON, LAT, X, Y, lonc, latc, current_time,
                             T_curr_resampled, dTdt, dTdx, dTdy, t_step=TIME_STEP_SECONDS):
    """Creates an xarray Dataset with processed SST data and derivatives."""
    ds = xr.Dataset(
        coords={
            "lon": (["lon"], lon),
            "lat": (["lat"], lat),
            "LON": (["lat", "lon"], LON),
            "LAT": (["lat", "lon"], LAT),
            "X": (["lat", "lon"], X),
            "Y": (["lat", "lon"], Y),
            "lonc": lonc,
            "latc": latc,
            "time": (["time"], [np.datetime64(current_time)]),
            "time_step": t_step,
        },
        data_vars={
            "T": (["time","lat", "lon"], T_curr_resampled),
            "dTdt": (["time","lat", "lon"], dTdt),
            "dTdx": (["time","lat", "lon"], dTdx),
            "dTdy": (["time","lat", "lon"], dTdy),
        },
    )
    return ds
        

# Main Processing Function
def process_sst_scene(savedir, time_str, ll_box, version=9, level='L3C', file_app='', return_time=False, sst_reduce=3):
    """Processes Himawari SST data and calculates temporal/spatial gradients."""
    # print(f"  Starting processing for {time_str}")
    lon, lat, T_curr, T_prev, T_next, current_time, fail = get_sst_time_series_data(savedir, time_str, ll_box, version=version, level=level, file_app=file_app)

    # Don't process if any file is missing
    if not fail:
        
        # Apply 3x3 window averaging to reduce resolution and noise
        lon_resampled = utils.calculate_mean_window_1d(lon, sst_reduce)
        lat_resampled = utils.calculate_mean_window_1d(lat, sst_reduce)
        T_curr_resampled = utils.calculate_mean_window_2d(T_curr, sst_reduce, sst_reduce, ignore_nan=IGNORE_NAN)
        T_prev_resampled = utils.calculate_mean_window_2d(T_prev, sst_reduce, sst_reduce, ignore_nan=IGNORE_NAN)
        T_next_resampled = utils.calculate_mean_window_2d(T_next, sst_reduce, sst_reduce, ignore_nan=IGNORE_NAN)

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
        ds = create_processed_dataset(
            lon_resampled, lat_resampled, LON, LAT, X, Y, lonc, latc, current_time,
            T_curr_resampled, dTdt, dTdx, dTdy, t_step=TIME_STEP_SECONDS)
        
        print(f"  Finished processing for {time_str}")
        if return_time:
            return ds.isel(time=0)
        else:
            return ds
    else:
        print(f"  Skipping processing for {time_str} due to missing files.")
        return None



def process_sst_series(savedir, time_limits, ll_box, version=9, level='L3C', file_app='', sst_reduce=3):
    """Processes Himawari SST data and calculates temporal/spatial gradients."""
    
    # Create and hourly time series within the defined time limits
    time_del = np.timedelta64(TIME_STEP_SECONDS, "s")
    time_steps = np.arange(np.datetime64(time_limits[0]), np.datetime64(time_limits[1]), np.timedelta64(1, 'h'))

    # Get the time steps to download
    time_dl = [str(t) for t in time_steps]
    
    ## Loop through the time steps and process each scene
    ds_list = []
    
    for time_str in time_dl:
        # Create an xarray Dataset to store all processed variables
        ds = process_sst_scene(savedir, time_str, ll_box, version=version, level=level, file_app=file_app, return_time=True, sst_reduce=sst_reduce)
    
        # print(f"  Finished processing for {time_str}")
        ds_list.append(ds)

    ds_all = [ds for ds in ds_list if ds is not None]  
    return xr.concat(ds_all, dim='time')


def load_crop_process_series_nasa(savedir, time_limits, ll_box, version=9, level='L3C', file_app='_cropped', overwrite=True, sst_reduce=3):
    
    # Create and hourly time series within the defined time limits
    time_del = np.timedelta64(TIME_STEP_SECONDS, "s")
    time_steps = np.arange(np.datetime64(time_limits[0]), np.datetime64(time_limits[1]), np.timedelta64(1, 'h'))

    # Download and preprocess Himawari data for the defined time steps
    ds_list = []
    for t_step in time_steps:
        f_name = get_himawari_file_path(savedir, str(t_step), version=version, level=level, file_app=file_app)
        if not os.path.exists(f_name) or not overwrite:
            get_sst_scene_nasa(t_step, savedir, overwrite=False, version=version, level=level)
        
        if overwrite:
            crop_sst_scene_nasa(savedir, t_step - time_del, ll_box, file_app=file_app, overwrite=True, version=version, level=level)
        else:
            crop_sst_scene_nasa(savedir, t_step - time_del, ll_box, file_app=file_app, overwrite=False, version=version, level=level)

        crop_sst_scene_nasa(savedir, t_step, ll_box, file_app=file_app, overwrite=False, version=version, level=level)
        crop_sst_scene_nasa(savedir, t_step + time_del, ll_box, file_app=file_app, overwrite=False, version=version, level=level)
                    
        ds = process_sst_scene(savedir, str(t_step), ll_box, file_app=file_app, return_time=True, version=version, level=level, sst_reduce=sst_reduce)
        ds_list.append(ds)
        
    if overwrite:
        crop_sst_scene_nasa(savedir, t_step, ll_box, file_app=file_app, overwrite=True, version=version, level=level)
        crop_sst_scene_nasa(savedir, t_step + time_del, ll_box, file_app=file_app, overwrite=True, version=version, level=level)
    
    ds_list = [ds for ds in ds_list if ds is not None]
    ds_all = xr.concat(ds_list, dim='time')
    return ds_all