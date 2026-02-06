import os
import numpy as np
import pandas as pd
import xarray as xr
from inversion_sst_gp import gp_regression, utils

def main():
    print("--- Starting optimization OSSE per process ---")

    # Configuration
    time_str = "2014-02-19T18:00:00"
    dataset_path = "1_preproc_data/proc_data/suntans_1h.nc"
    outputs_dir = "2_covariance_parameter_estimation/outputs"
    output_path = f"{outputs_dir}/num_model_estimated_t.csv"

    # Load dataset
    print(f"Loading dataset from {dataset_path} for time {time_str}")
    ds = xr.open_dataset(dataset_path).sel(time=np.datetime64(time_str))

    # Extract variables
    lon, lat, u, v, S = (
        ds[var].values for var in ("lon", "lat", "u", "v", "S")
    )

    # Compute grid properties
    _, _, X, Y, _, _ = utils.calculate_grid_properties(lon, lat)

    # Estimate GP regression hyperparameters
    theta = {
        "sigma_u": None,
        "l_u": None,
        "tau_u": None,
        "sigma_v": None,
        "l_v": None,
        "tau_v": None,
        "sigma_S": None,
        "l_S": None,
        "tau_S": None,
    }

    print("Estimating hyperparameters for 'u'")
    theta["sigma_u"], theta["l_u"], theta["tau_u"] = gp_regression.estimate_params_process(
        u, X, Y, 1e-1, 4e4, 1e-3
    )

    print("Estimating hyperparameters for 'v'")
    theta["sigma_v"], theta["l_v"], theta["tau_v"] = gp_regression.estimate_params_process(
        v, X, Y, 1e-1, 4e4, 1e-3
    )

    print("Estimating hyperparameters for 'S'")
    theta["sigma_S"], theta["l_S"], theta["tau_S"] = gp_regression.estimate_params_process(
        S, X, Y, 3e-7, 3e4, 2e-7
    )

    # Save results
    print(f"Saving hyperparameters to {output_path}")
    theta["time"] = time_str # type: ignore
    os.makedirs(outputs_dir, exist_ok=True)
    pd.DataFrame([theta]).to_csv(output_path, index=False)

    print("GP hyperparameter estimation complete")

if __name__ == "__main__":
    main()