import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import os

from inversion_sst_gp import utils, simulate_obs

# Configuration
LON_LIMITS = (115, 118)
LAT_LIMITS = (-15.5, -12.5)
HIMAWARI_GRID_PATH = "1_preproc_data/proc_data/himawari.nc"
OCEAN_MODEL_DATA_PATH = '1_preproc_data/non_proc_data/suntans/suntans_surface_prop_north_australian_basin.nc'
OSSE_SNAPSHOT_TIME = np.datetime64('2014-02-19T18:00:00')
TIME_STEP = 3600  # seconds
PROCESSED_DIR = "1_preproc_data/proc_data"

# Test configurations
TEST_CONFIGS = [
    {
        'name': 'measurement_error',
        'val_range': np.round(np.arange(0, 0.015 + 0.0001, 0.001),3),
        'dataset_name': "suntans_measurement_error",
        'param_name': 'sigma_tau',
        'time_dependent': False,
    },
    {
        'name': 'sparse_cloud',
        'val_range': np.round(np.arange(0, 0.75 + 0.001, 0.03),2),
        'dataset_name': "suntans_sparse_cloud",
        'param_name': 'coverage_sparse',
        'time_dependent': False,
    },
    {
        'name': 'dense_cloud',
        'val_range': np.round(np.arange(0, 0.75 + 0.001, 0.03),2),
        'dataset_name': "suntans_dense_cloud",
        'param_name': 'coverage_dense',
        'time_dependent': False,
    },
    {
        'name': 'time_24h',
        'val_range': np.arange(0, 100) * 24 * TIME_STEP,
        'dataset_name': "suntans_24h",
        'param_name': 'time',
        'time_dependent': True,
    },
    {
        'name': 'time_1h',
        'val_range': np.arange(0, 49) * TIME_STEP,
        'dataset_name': "suntans_1h",
        'param_name': 'time',
        'time_dependent': True,
    },
]

# Helper functions
def load_himawari_grid(path_himawari_file):
    """Loads the Himawari grid coordinates from a NetCDF file."""
    ds = xr.open_dataset(path_himawari_file)
    ds_grid = ds.coords.to_dataset()
    ds_grid = ds_grid.drop_vars(['time', 'time_step'])
    return ds_grid


def interpolate_osse_to_grid(ds_o, ds_grid, target_datetime):
    """Interpolates non-gridded ocean model output to a rectangular grid."""
    idx = np.where(ds_o.time == target_datetime)[0][0]

    T_ug = ds_o.T.isel(time=idx).values
    T_ug_p = ds_o.T.isel(time=idx - 1).values
    T_ug_n = ds_o.T.isel(time=idx + 1).values
    u_ug = ds_o.u.isel(time=idx).values
    v_ug = ds_o.v.isel(time=idx).values
    time_p = ds_o.time.isel(time=idx - 1).values
    time = ds_o.time.isel(time=idx).values
    time_n = ds_o.time.isel(time=idx + 1).values
    lon_ug = ds_o.lon.values
    lat_ug = ds_o.lat.values

    tstep_calc = (time_n - time_p) / np.timedelta64(1, 's') / 2
    dT_ug = (T_ug_n - T_ug_p) / 2
    dTdt_ug = dT_ug / tstep_calc

    LON = ds_grid.LON.values
    LAT = ds_grid.LAT.values
    lon = ds_grid.lon.values
    lat = ds_grid.lat.values
    X = ds_grid.X.values
    Y = ds_grid.Y.values

    points = np.stack([lon_ug, lat_ug]).T
    T = griddata(points, T_ug, (LON, LAT), method='cubic')
    dTdt = griddata(points, dTdt_ug, (LON, LAT), method='cubic')
    u = griddata(points, u_ug, (LON, LAT), method='cubic')
    v = griddata(points, v_ug, (LON, LAT), method='cubic')

    dTdx, dTdy = utils.finite_difference_2d(X, Y, T)
    S = dTdt + u * dTdx + v * dTdy

    ds_data = xr.Dataset(
        data_vars=dict(
            T=(['lat', 'lon'], T),
            dTdt=(['lat', 'lon'], dTdt),
            dTdx=(['lat', 'lon'], dTdx),
            dTdy=(['lat', 'lon'], dTdy),
            u=(['lat', 'lon'], u),
            v=(['lat', 'lon'], v),
            S=(['lat', 'lon'], S),
        ),
        coords=dict(
            lon=(['lon'], lon),
            lat=(['lat'], lat),
            time=time,
            time_step=tstep_calc,
        )
    )
    return xr.merge([ds_grid, ds_data])


def save_dataset(data_arrays, coords, dataset_name):
    """Saves the processed xarray Dataset to a NetCDF file."""
    ds = xr.Dataset(data_arrays, coords=coords)
    output_path = f'{PROCESSED_DIR}/{dataset_name}.nc'
    ds.to_netcdf(output_path)
    print(f"Saved {dataset_name} to {output_path}")


# Combined test function
def run_osse_test(config, ds_osse_data, ds_grid, osse_snapshot_time):
    """
    Runs a specified OSSE test (time series or observation modification) based on provided config.
    `ds_osse_data` will be `ds_osse_full` for time-dependent tests and `ds_osse_snapshot` otherwise.
    """
    test_name = config['name']
    val_range = config['val_range']
    dataset_name = config['dataset_name']
    param_name = config['param_name']
    time_dependent = config['time_dependent']

    print(f"\nStarting '{test_name}' test")
    np.random.seed(0)  # Set seed for reproducibility

    Ny, Nx = len(ds_grid.lat), len(ds_grid.lon)
    Toc, uc, vc, Sc, dTdtoc = (
        np.empty((len(val_range), Ny, Nx)),
        np.empty((len(val_range), Ny, Nx)),
        np.empty((len(val_range), Ny, Nx)),
        np.empty((len(val_range), Ny, Nx)),
        np.empty((len(val_range), Ny, Nx)),
    )
    time_coords = []
    current_time_for_osse = None  # Ensure variable is always defined

    for i, val in enumerate(val_range):
        # Determine the source dataset for this iteration
        if time_dependent:
            current_time_for_osse = osse_snapshot_time + np.timedelta64(int(val), 's')
            ds_current_iteration_data = interpolate_osse_to_grid(ds_osse_data, ds_grid, current_time_for_osse)
        else:
            # ds_osse_data is already the snapshot for non-time-dependent tests
            ds_current_iteration_data = ds_osse_data

        Tt = ds_current_iteration_data.T.values
        dTdtt = ds_current_iteration_data.dTdt.values
        X = ds_current_iteration_data.X.values
        Y = ds_current_iteration_data.Y.values

        uc[i, :, :] = ds_current_iteration_data.u.values
        vc[i, :, :] = ds_current_iteration_data.v.values
        Sc[i, :, :] = ds_current_iteration_data.S.values

        if test_name == 'measurement_error':
            Toc[i, :, :], _, _, dTdtoc[i, :, :], _ = simulate_obs.ModifyData(Tt, dTdtt, TIME_STEP, X, Y).noise(val).convert_to_input()
        elif test_name == 'sparse_cloud':
            Toc[i, :, :], _, _, dTdtoc[i, :, :], _ = simulate_obs.ModifyData(Tt, dTdtt, TIME_STEP, X, Y).sparse_cloud(val).convert_to_input()
        elif test_name == 'dense_cloud':
            Toc[i, :, :], _, _, dTdtoc[i, :, :], _ = simulate_obs.ModifyData(Tt, dTdtt, TIME_STEP, X, Y).circ_cloud(val).convert_to_input()
        else:  # For time-dependent tests, T and dTdt are directly from ds_current_iteration_data
            Toc[i, :, :] = Tt
            dTdtoc[i, :, :] = dTdtt
        
        if time_dependent:
            time_coords.append(current_time_for_osse)

        if (i + 1) % 10 == 0 or i == len(val_range) - 1:
            print(f"Progress: {i + 1}/{len(val_range)} for '{test_name}'")
            
    data_arrays = {
        'T': ([param_name, 'lat', 'lon'], Toc),
        'dTdt': ([param_name, 'lat', 'lon'], dTdtoc),
        'u': ([param_name, 'lat', 'lon'], uc),
        'v': ([param_name, 'lat', 'lon'], vc),
        'S': ([param_name, 'lat', 'lon'], Sc),
    }

    coords = {
        param_name: time_coords if time_dependent else val_range,
        "time_step": TIME_STEP,
        "lat": ds_grid.lat.values,
        "lon": ds_grid.lon.values,
    }
    if not time_dependent:
        coords["time"] = osse_snapshot_time

    save_dataset(data_arrays, coords, dataset_name)
    print(f"Finished '{test_name}' test")


# Main execution
if __name__ == "__main__":
    print("--- Starting preprocessing SUNTANS data---")

    print("Loading Himawari grid")
    ds_himawari_grid = load_himawari_grid(HIMAWARI_GRID_PATH)

    print("Loading OSSE dataset")
    ds_osse_full = xr.open_mfdataset(OCEAN_MODEL_DATA_PATH)

    # Prepare single OSSE snapshot for non-time-dependent tests once
    print("Preparing single OSSE snapshot for observation modification tests")
    ds_osse_snapshot = interpolate_osse_to_grid(ds_osse_full, ds_himawari_grid, OSSE_SNAPSHOT_TIME)
    
    # Create processed directory if it doesn't exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Run all tests defined in TEST_CONFIGS
    for test_config in TEST_CONFIGS:
        if test_config['time_dependent']:
            # For time-dependent tests, pass the full dataset
            run_osse_test(test_config, ds_osse_full, ds_himawari_grid, OSSE_SNAPSHOT_TIME)
        else:
            # For non-time-dependent tests, pass the pre-computed snapshot
            run_osse_test(test_config, ds_osse_snapshot, ds_himawari_grid, OSSE_SNAPSHOT_TIME)

    print("\nData processing complete")