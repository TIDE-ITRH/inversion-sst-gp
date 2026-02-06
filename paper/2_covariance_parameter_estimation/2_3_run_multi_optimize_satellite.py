import os
import numpy as np
import xarray as xr

from inversion_sst_gp import gp_regression, utils

def run_satellite(
    time_str: str,
    id: str,
    save_results: bool = True,
):
    """
    Run satellite data processing and GP regression for hyperparameter estimation.

    Parameters
    ----------
    time_str : str
        Time string for the dataset.
    id : str
        Unique identifier for result output file.
    save_results : bool, optional
        If True, saves the results JSON to the intermediate directory.
        
    Returns
    -------
    dict
        Results dictionary with estimated parameters.
    """

    # Load Himawari data
    print(f"Loading Himawari data for time {time_str}")
    ds = xr.open_dataset("1_preproc_data/proc_data/himawari.nc").sel(time=np.datetime64(time_str))
    lon, lat, To, dTdto = (
        ds[var].values for var in ("lon", "lat", "T", "dTdt")
    )
    tstep = ds.time_step.item()
    _, _, X, Y, _, _ = utils.calculate_grid_properties(lon, lat)
    dTds1o, dTds2o = utils.finite_difference_2d(X, Y, To)

    # Set model hyperparameters
    initial_params = {
        "sigma_u": 9e-2,
        "l_u": 3e4,
        "tau_u": 1e-2,
        "sigma_S": 3e-7,
        "l_S": 2e4,
        "tau_S": 2e-7,
        "sigma_tau": 1e-2,
    }
    penalty_params = {
        "l_u": [3e4, 0.5e4],
        "sigma_u": [9e-2, 2e-2],
        "tau_u": [1e-2, 0.1e-2],
        "l_S": [2e4, 2e4],
        "sigma_S": [3e-7, 5e-6],
        "tau_S": [2e-7, 5e-6],
    }
    bounds_params = {
        "sigma_u": [1e-10, 10],
        "l_u": [1, 1e6],
        "tau_u": [1e-10, 1],
        "sigma_S": [1e-8, 1e-3],
        "l_S": [1, 1e6],
        "tau_S": [1e-15, 1e-3],
        "sigma_tau": [1e-15, 1],
    }
    prop_sat = {
        "initial_params": initial_params,
        "const_params": {},
        "penalty_params": penalty_params,
        "share_len": True,
        "share_tau": True,
        "share_sigma": True,
        "solve_log": True,
        "bounds_params": bounds_params,
    }

    # Metric functions
    def run_gprm_optim(time_str, dTds1, dTds2, dTdt, X, Y, tstep, prop, callback="off"):
        mask = np.ones_like(X, dtype=bool)
        gprm = gp_regression.GPRegressionJoint(dTds1, dTds2, dTdt, tstep, X, Y, mask)
        est_params = gprm.estimate_params(**prop, callback=callback)
        return {
            "step": time_str,
            "est_params": est_params,
        }

    # Run model
    print("Running GP optimization")
    results = run_gprm_optim(
        time_str, dTds1o, dTds2o, dTdto, X, Y, tstep, prop_sat
    )
    
    # Save results
    if save_results:
        intermediate_dir = "2_covariance_parameter_estimation/intermediate"
        os.makedirs(intermediate_dir, exist_ok=True)
        file_name = f"{intermediate_dir}/satellite_{id}.json"
        print(f"Saving results to {file_name}")
        utils.save_json(results, file_name)

    return results

def main():
    print("--- Starting multi optimize hyperparameter satellite ---")
    
    # List of time strings for processing
    time_str_list = ["2023-09-22T04:00:00","2023-12-18T01:00:00"]
    
    # Run satellite processing for each time string
    for id, time_str in enumerate(time_str_list):
        print(f"\nRunning task {id + 1}/{len(time_str_list)} for time {time_str}")
        run_satellite(time_str, str(id))
        
    print("\nAll tasks completed")

if __name__ == "__main__":
    main()