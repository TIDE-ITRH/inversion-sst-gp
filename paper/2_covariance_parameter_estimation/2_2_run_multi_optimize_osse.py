import os
import numpy as np
import xarray as xr
from inversion_sst_gp import gp_regression, metrics, other_methods, utils

def run_osse(
    step: float | int,
    model_type: str,
    test_type: str,
    run_id: str,
    time_base_str: str = "2014-02-19T18:00:00",
    save_results: bool = True,
):
    """
    Run OSSE for given configuration.

    Parameters
    ----------
    step : float | int
        Step size (seconds for time-based tests, sigma_tau for noise, etc.).
    model_type : str
        Model type: "gos", "gprm", "gprm_e", or "optimum".
    test_type : str
        Test type: "noise", "time_24h", "time_1h", "cloud_sparse", "cloud_dense".
    run_id : str
        Unique identifier for result output file.
    time_base_str : str, optional
        Base reference time string.
    save_results : bool, optional
        If True, saves the results JSON to the intermediate directory.

    Returns
    -------
    dict
        Results dictionary with metrics and parameters.
    """

    time_base = np.datetime64(time_base_str)

    # Set model hyperparameters
    if model_type == "gprm":
        theta = utils.extract_params(
            "2_covariance_parameter_estimation/outputs/num_model_estimated_t.csv",
            "time",
            time_base_str,
            type="num_est",
        )
        prop_osse = {
            "initial_params": {"sigma_tau": 0.01},
            "const_params": theta,
            "penalty_params": {},
            "share_len": False,
            "share_tau": False,
            "share_sigma": False,
            "solve_log": True,
            "bounds_params": {},
        }

    elif model_type == "gprm_e":
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
        prop_osse = {
            "initial_params": initial_params,
            "const_params": {},
            "penalty_params": penalty_params,
            "share_len": True,
            "share_tau": True,
            "share_sigma": True,
            "solve_log": True,
            "bounds_params": bounds_params,
        }

    elif model_type == "optimum":
        ds_base = xr.open_dataset("1_preproc_data/proc_data/suntans_1h.nc").sel(time=time_base)

        lon, lat, _, _, u, v, S = (
            ds_base[var].values for var in ("lon", "lat", "T", "dTdt", "u", "v", "S")
        )
        _, _, X, Y, _, _ = utils.calculate_grid_properties(lon, lat)

        print("Estimating parameters for optimum model")
        sigma_u, l_u, tau_u = gp_regression.estimate_params_process(u, X, Y, 1e-1, 4e4, 1e-3)
        sigma_v, l_v, tau_v = gp_regression.estimate_params_process(v, X, Y, 1e-1, 4e4, 1e-3)
        sigma_S, l_S, tau_S = gp_regression.estimate_params_process(S, X, Y, 3e-7, 3e4, 2e-7)

        theta = {
            "sigma_u": sigma_u, "l_u": l_u, "tau_u": tau_u,
            "sigma_v": sigma_v, "l_v": l_v, "tau_v": tau_v,
            "sigma_S": sigma_S, "l_S": l_S, "tau_S": tau_S,
        }

        prop_osse = {
            "initial_params": {"sigma_tau": 0.01},
            "const_params": theta,
            "penalty_params": {},
            "share_len": False,
            "share_tau": False,
            "share_sigma": False,
            "solve_log": True,
            "bounds_params": {},
        }

    else:
        prop_osse = None  # For "gos" case

    # Load test dataset
    print(f"Loading dataset for test_type={test_type}, step={step}")
    if test_type == "noise":
        ds = xr.open_dataset("1_preproc_data/proc_data/suntans_measurement_error.nc").sel(sigma_tau=step)
    elif test_type == "time_24h":
        ds = xr.open_dataset("1_preproc_data/proc_data/suntans_24h.nc").sel(
            time=time_base + np.timedelta64(int(step), "s")
        )
    elif test_type == "time_1h":
        ds = xr.open_dataset("1_preproc_data/proc_data/suntans_1h.nc").sel(
            time=time_base + np.timedelta64(int(step), "s")
        )
    elif test_type == "cloud_sparse":
        ds = xr.open_dataset("1_preproc_data/proc_data/suntans_sparse_cloud.nc").sel(coverage_sparse=step)
    elif test_type == "cloud_dense":
        ds = xr.open_dataset("1_preproc_data/proc_data/suntans_dense_cloud.nc").sel(coverage_dense=step)
    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    tstep = ds.time_step.item()
    lon, lat, To, dTdto, u, v, _ = (
        ds[var].values for var in ("lon", "lat", "T", "dTdt", "u", "v", "S")
    )
    _, _, X, Y, _, _ = utils.calculate_grid_properties(lon, lat)
    dTds1o, dTds2o = utils.finite_difference_2d(X, Y, To)

    # Helper functions
    def run_gos_optim(step, dTds1, dTds2, dTdt, u, v):
        gos = other_methods.GlobalOptimalSolution(dTds1, dTds2, dTdt)
        n = gos.optimize_n(u, v)
        ugos, vgos, _ = gos.compute(n)
        rmse = metrics.overview(u, v, ugos, vgos)["RMSE"]
        return {"step": step, "n": n, "RMSE": rmse}

    def run_gprm_optim(step, dTds1, dTds2, dTdt, u, v, X, Y, tstep, prop):
        mask = np.ones_like(X, dtype=bool)
        gprm = gp_regression.GPRegressionJoint(dTds1, dTds2, dTdt, tstep, X, Y, mask)
        est_params = gprm.estimate_params(**prop, callback="off")
        mu_u, mu_v, _, std_u, std_v, _, _ = gprm.predict(est_params)
        results = metrics.overview(u, v, mu_u, mu_v, std_u, std_v)
        return {
            "step": step,
            "est_params": est_params,
            "RMSE": results["RMSE"],
            "crps_norm": results["crps_norm"],
            "coverage90": results["coverage90"],
        }

    # Run model
    if model_type == "gos":
        print("Running GOS optimization")
        results = run_gos_optim(step, dTds1o, dTds2o, dTdto, u, v)
    else:
        print(f"Running {model_type.upper()} optimization")
        results = run_gprm_optim(step, dTds1o, dTds2o, dTdto, u, v, X, Y, tstep, prop_osse)

    # Save results
    if save_results:
        intermediate_dir = "2_covariance_parameter_estimation/intermediate"
        os.makedirs(intermediate_dir, exist_ok=True)
        file_name = f"{intermediate_dir}/osse_{model_type}_{test_type}_{run_id}.json"
        print(f"Saving results to {file_name}")
        utils.save_json(results, file_name)

    return results

def main():
    print("--- Starting multi optimize hyperparameter OSSE ---")
    
    # Mapping from test_type → steps list and models list
    step_map = {
        "time_24h": list(range(0, 8640001, 86400)),
        "time_1h": list(range(0, 172801, 3600)),
        "noise": np.arange(0, 0.016, 0.001).tolist(),
        "cloud_sparse": np.arange(0, 0.76, 0.03).tolist(),
        "cloud_dense": np.arange(0, 0.76, 0.03).tolist(),
    }

    model_map = {
        "time_24h": ["gos", "gprm_e", "gprm", "optimum"],
        "time_1h": ["gos", "gprm_e", "gprm", "optimum"],
        "noise": ["gos", "gprm_e", "gprm"],
        "cloud_sparse": ["gprm_e", "gprm"],
        "cloud_dense": ["gprm_e", "gprm"],
    }

    # Build all combinations
    combinations = []
    for test_type, steps in step_map.items():
        models = model_map[test_type]
        for step_index, step in enumerate(steps):
            for model in models:
                combinations.append((step, model, test_type, step_index))

    # Run OSSE for each combination
    for task_id in range(len(combinations)):
        step, model_type, test_type, step_index = combinations[task_id]
        print(f"\nRunning task {task_id + 1}/{len(combinations)} with: index={step_index}, step={step}, model_type={model_type}, test_type={test_type}")
        run_osse(step=step, model_type=model_type, test_type=test_type, run_id=str(step_index))
        
    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()
