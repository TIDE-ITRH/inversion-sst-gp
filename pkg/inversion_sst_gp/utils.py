"""
Utility functions for geographic and numerical computations.

This module provides a collection of utility functions for tasks such as
coordinate transformations, finite difference calculations, vorticity
calculations, and JSON file handling.
"""

import numpy as np
import json
import pandas as pd
from geographiclib.geodesic import Geodesic

def find_centre(lon, lat):
    """
    Calculate the geographic center of a given set of longitude and latitude values.

    Parameters:
        lon (array-like): Array of longitude values.
        lat (array-like): Array of latitude values.

    Returns:
        tuple: The latitude and longitude of the center.
    """

    latlims = [np.min(lat), np.max(lat)]


    lonlims = [np.min(lon), np.max(lon)]
    latc = np.mean(latlims)
    lonc = np.mean(lonlims)
    return latc, lonc

def geo_to_car(lon, lat, lonc, latc):
    """
    Convert geographic coordinates to Cartesian coordinates with the center as origin.

    Parameters:
        lon (array-like): Array of longitude values.
        lat (array-like): Array of latitude values.
        lonc (float): Longitude of the center.
        latc (float): Latitude of the center.

    Returns:
        tuple: Cartesian coordinates (X, Y).
    """

    # convert geographic coordinate to cartesian coordinates with the centre as origin        
    Nx  = len(lon)


    Ny  = len(lat)
    X = np.empty([Ny,Nx])
    Y = np.empty([Ny,Nx])
    for i in range(Nx):
        for j in range(Ny): 
            X[j,i] = np.sign(lon[i]-lonc)*Geodesic.WGS84.Inverse(lat[j],lon[i],lat[j],lonc)['s12']
            Y[j,i] = np.sign(lat[j]-latc)*Geodesic.WGS84.Inverse(lat[j],lon[i],latc,lon[i])['s12']
    return X,Y

def calculate_grid_properties(lon, lat):
    """
    Calculate grid properties including Cartesian coordinates and meshgrid.

    Parameters:
        lon (array-like): Array of longitude values.
        lat (array-like): Array of latitude values.

    Returns:
        tuple: Center coordinates, Cartesian coordinates, and meshgrid.
    """

    latc, lonc = find_centre(lon, lat)


    X, Y = geo_to_car(lon, lat, lonc, latc)
    LON, LAT = np.meshgrid(lon, lat)
    return lonc, latc, X, Y, LON, LAT

def finite_difference_1d(s, x):
    """
    Compute the finite difference of a 1D array, handling NaN values.

    Parameters:
        s (array-like): Independent variable.
        x (array-like): Dependent variable.

    Returns:
        ndarray: Finite difference values.
    """

    inan = np.isnan(x) 


    if np.any(inan): 
        inanR = np.hstack([inan[1:],True]) 
        inanL = np.hstack([True,inan[:-1]]) 

        sp = np.roll(s,-1) 
        xp = np.roll(x,-1) 
        sm = np.roll(s,1) 
        xm = np.roll(x,1) 

        iforward = (~inan) & (~inanR) & inanL
        icentral = (~inanL) & (~inanR)
        ibackward = (~inan) & inanR & (~inanL)

        dxds = np.empty(len(x)) 
        dxds.fill(np.nan)
        dxds[iforward] = (xp[iforward]-x[iforward])/(sp[iforward]-s[iforward]) 
        dxds[icentral] = (xp[icentral]-xm[icentral])/(sp[icentral]-sm[icentral])
        dxds[ibackward] = (x[ibackward]-xm[ibackward])/(s[ibackward]-sm[ibackward]) 
    else: 
        forward_diff = (x[1]-x[0])/(s[1]-s[0]) 
        central_diff = (x[2:]-x[:-2])/(s[2:]-s[:-2])
        backward_diff = (x[-1]-x[-2])/(s[-1]-s[-2]) 

        dxds = np.hstack([forward_diff,central_diff,backward_diff])
    return dxds   

def finite_difference_2d(s1_2d, s2_2d, x_2d):
    """
    Compute the finite difference of a 2D array along two dimensions.

    Parameters:
        s1_2d (array-like): Independent variable for the first dimension.
        s2_2d (array-like): Independent variable for the second dimension.
        x_2d (array-like): Dependent variable.

    Returns:
        tuple: Finite differences along both dimensions.
    """

    N2,N1 = x_2d.shape


    dxds1 = np.stack([finite_difference_1d(s1_2d[i,:],x_2d[i,:]) for i in range(N2)])
    dxds2 = np.stack([finite_difference_1d(s2_2d[:,i],x_2d[:,i]) for i in range(N1)]).T
    return dxds1,dxds2

def calculate_vorticity(s1_2d, s2_2d, u, v):
    """
    Calculate the vorticity of a 2D velocity field.

    Parameters:
        s1_2d (array-like): Independent variable for the first dimension.
        s2_2d (array-like): Independent variable for the second dimension.
        u (array-like): Velocity component in the first dimension.
        v (array-like): Velocity component in the second dimension.

    Returns:
        ndarray: Vorticity values.
    """

    _, dudy = finite_difference_2d(s1_2d,s2_2d,u)


    dvdx, _ = finite_difference_2d(s1_2d,s2_2d,v)
    return dvdx - dudy

def calculate_coriolis_parameter(latitude_deg):
    """
    Calculate the Coriolis parameter for a given latitude.

    Parameters:
        latitude_deg (float): Latitude in degrees.

    Returns:
        float: Coriolis parameter.
    """

    omega = 7.2921e-5


    latitude_rad = np.deg2rad(latitude_deg)
    return 2 * omega * np.sin(latitude_rad)

def calculate_dynamic_rossby_number(s1_2d, s2_2d, u, v, latitude_deg):
    """
    Calculate the dynamic Rossby number for a 2D velocity field.

    Parameters:
        s1_2d (array-like): Independent variable for the first dimension.
        s2_2d (array-like): Independent variable for the second dimension.
        u (array-like): Velocity component in the first dimension.
        v (array-like): Velocity component in the second dimension.
        latitude_deg (float): Latitude in degrees.

    Returns:
        ndarray: Dynamic Rossby number values.
    """

    vorticity = calculate_vorticity(s1_2d,s2_2d,u,v)


    coriolis_parameter = calculate_coriolis_parameter(latitude_deg)
    return vorticity/coriolis_parameter

def map_val(x, mask):
    """
    Map values to a masked array.

    Parameters:
        x (array-like): Values to map.
        mask (array-like): Boolean mask array.

    Returns:
        ndarray: Mapped values.
    """

    # map value given mask
    xf = np.full(list(np.shape(mask)) + list(x.shape[1:]),np.nan) # allocate


    xf[mask] = x
    return xf

def map_mask(maski, mask):
    """
    Map a mask to another mask.

    Parameters:
        maski (array-like): Input mask.
        mask (array-like): Target mask.

    Returns:
        ndarray: Mapped mask.
    """

    # map maski given mask
    maskf = np.full(len(mask),False) # allocate


    maskf[mask] = maski
    return maskf

def extract_params(filename, name, value, type):
    """
    Extract parameters from a CSV file based on a specific value.

    Parameters:
        filename (str): Path to the CSV file.
        name (str): Column name to match.
        value: Value to match in the column.
        type (str): Type of parameters to extract.

    Returns:
        dict: Extracted parameters.
    """

    if type == "gp":


        keys=["sigma_u","l_u","tau_u","l_v","tau_v","sigma_v","sigma_S","l_S","tau_S","sigma_tau"]
    elif type == "num_est":
        keys=["sigma_u","l_u","tau_u","l_v","tau_v","sigma_v","sigma_S","l_S","tau_S"]
    elif type == "gos":
        keys=["n"]
    
    df = pd.read_csv(filename)
    row = df[df[name] == value].iloc[0]
    return {k: row[k] for k in keys} if keys else row.to_dict()

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for handling NumPy data types.
    """

    def default(self, obj):


        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)
    
def save_json(data, path):
    """
    Save data to a JSON file.

    Parameters:
        data (dict): Data to save.
        path (str): Path to the JSON file.
    """

    # save variable to json
    with open(path, 'w') as json_file:


        json.dump(data, json_file, indent=4, cls=NumpyEncoder)
    pass

def load_json(path):
    """
    Load data from a JSON file.

    Parameters:
        path (str): Path to the JSON file.

    Returns:
        dict: Loaded data.
    """

    # load variable from json
    with open(path, 'r') as json_file:


        data = json.load(json_file)
    return data

def calculate_mean_window_1d(val, n, ignore_nan=True):
    """
    Calculate the mean of every n elements in a 1D array.

    Parameters:
        val (array-like): Input array.
        n (int): Window size.
        ignore_nan (bool): Whether to ignore NaN values.

    Returns:
        ndarray: Array of mean values.
    """

    # calculate the mean of every n elements
    Nx = len(val)


    Nxm = Nx // n 
    valm = np.empty([Nxm])
    for i in range(Nxm): 
        val_window = val[i*n:(i+1)*n]
        mask = np.logical_not(np.isnan(val_window))
        if ignore_nan:
            if np.any(mask):
                valm[i] = np.mean(val_window[mask])
            else:
                valm[i] = np.nan
        else:
            if np.all(mask):
                valm[i] = np.mean(val_window)
            else:
                valm[i] = np.nan
    return valm

def calculate_mean_window_2d(val, nx, ny, ignore_nan=True):
    """
    Calculate the mean of nx by ny grids in a 2D array.

    Parameters:
        val (array-like): Input 2D array.
        nx (int): Grid size in the x-dimension.
        ny (int): Grid size in the y-dimension.
        ignore_nan (bool): Whether to ignore NaN values.

    Returns:
        ndarray: Array of mean values for each grid.
    """

    # calculate the mean of a nx by ny grid
    Ny, Nx = np.shape(val)


    Nym = Ny // nx
    Nxm = Nx // ny
    valm = np.empty([Nym, Nxm]) 
    for i in range(Nxm):
        for j in range(Nym):
            valb = val[j*ny:(j+1)*ny, i*nx:(i+1)*nx] 
            mask = np.logical_not(np.isnan(valb)) 
            if ignore_nan:
                if np.any(mask):
                    valm[j, i] = np.mean(valb[mask]) 
                else:
                    valm[j, i] = np.nan
            else:
                if np.all(mask):
                    valm[j, i] = np.mean(valb) 
                else:
                    valm[j, i] = np.nan
    return valm



# Assuming `ds` is your xarray dataset
def clean_dataset(ds):
    # Calculate the difference along the time dimension
    diff = ds.diff(dim='time')
    
    # Identify time steps where all variables are zero (repeated time steps)
    repeated_time_steps = (diff == 0).to_array().all(dim='variable').shift(time=1, fill_value=False)
    
    # Drop repeated time steps
    ds = ds.where(~repeated_time_steps, drop=True)
    return ds

def make_even_hourly(ds, time_dim='time'):
    """
    Make the time coordinate of an xarray dataset evenly spaced hourly and fill missing steps with NaN.

    Parameters:
    - ds: xarray.Dataset
        The input dataset with a time coordinate.
    - time_dim: str
        The name of the time dimension (default is 'time').

    Returns:
    - xarray.Dataset
        The dataset with an even hourly time coordinate and NaN for missing steps.
    """
    # Ensure the time coordinate is a pandas datetime index
    time_index = pd.to_datetime(ds[time_dim].values)
    
    # Create an evenly spaced hourly time index
    even_time_index = pd.date_range(start=time_index.min(), end=time_index.max(), freq='h')
    
    # Reindex the dataset to the new time index, filling missing steps with NaN
    ds_even = ds.reindex({time_dim: even_time_index})
    
    return ds_even

def compute_good_data_over_time(ds, variable_name):
    """
    Compute the amount of good data (non-NaN values) for a variable along all
    dimensions except 'time', output as a function of time.

    Parameters:
    - ds: xarray.Dataset
        The input dataset.
    - variable_name: str
        The name of the variable to compute the good data for.

    Returns:
    - xarray.DataArray
        A DataArray with the amount of good data as a function of time.
    """
    # Select the variable
    data = ds[variable_name]
    
    # Count non-NaN values along all dimensions except 'time'
    good_data_count = data.notnull().sum(dim=[dim for dim in data.dims if dim != 'time'])
    
    full_data_count = np.prod([data.sizes[dim] for dim in data.dims if dim != 'time'])
    
    return good_data_count / full_data_count