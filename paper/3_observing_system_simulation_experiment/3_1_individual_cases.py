import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import rc

from inversion_sst_gp import (
    plot_helper,
    utils,
    other_methods,
    metrics,
    gp_regression,
)

# Matplotlib configuration
rc("font", family="serif", serif=["Computer Modern"])
rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{amsmath}")

# Plotting parameters
lonlims = (115, 118)
latlims = (-15.5, -12.5)


# Helper functions
def load_and_prepare_dataset(path, selection_key=None, selection_val=None):
    print(f"Loading dataset: {path}")
    ds = xr.open_dataset(path)
    if selection_key and selection_val is not None:
        ds = ds.sel({selection_key: selection_val})
        print(f"Selecting dataset with {selection_key}={selection_val}")
    time_step = ds.time_step.item()
    lon, lat, To, dTdto, u, v, S = (
        ds[var].values for var in ("lon", "lat", "T", "dTdt", "u", "v", "S")
    )
    lonc, latc, X, Y, LON, LAT = utils.calculate_grid_properties(lon, lat)
    dTds1o, dTds2o = utils.finite_difference_2d(X, Y, To)
    plot_helper.visualize_data(
        LON, LAT, To, dTdto, dTds1o, dTds2o, lonlims=lonlims, latlims=latlims
    )
    print("Dataset loaded and preprocessed")
    return {
        "ds": ds,
        "time_step": time_step,
        "lon": lon,
        "lat": lat,
        "To": To,
        "dTdto": dTdto,
        "u": u,
        "v": v,
        "S": S,
        "lonc": lonc,
        "latc": latc,
        "X": X,
        "Y": Y,
        "LON": LON,
        "LAT": LAT,
        "dTds1o": dTds1o,
        "dTds2o": dTds2o,
    }


def run_gp_regression_and_metrics(
    dTds1o,
    dTds2o,
    dTdto,
    params_path,
    param_key,
    param_val,
    param_type,
    X,
    Y,
    time_step,
    u,
    v,
):
    print(
        f"Extracting parameters from {params_path} for {param_key}={param_val} ({param_type})"
    )
    params = utils.extract_params(params_path, param_key, param_val, type=param_type)
    print("Calculating GP regression predictions")
    muustar, muvstar, muSstar, stdustar, stdvstar, stdSstar, Kxstar_vel = (
        gp_regression.calculate_prediction_gpregression(
            dTds1o, dTds2o, dTdto, params, X, Y, time_step
        )
    )
    print("Calculating GP metrics")
    metrics_gp = metrics.overview(
        u, v, muustar, muvstar, stdustar, stdvstar, print_bool=False
    )
    return (
        muustar,
        muvstar,
        muSstar,
        stdustar,
        stdvstar,
        stdSstar,
        Kxstar_vel,
        metrics_gp,
    )


def run_global_optimal_solution(
    dTds1o, dTds2o, dTdto, params_path, param_key, param_val, param_type, u, v
):
    print(
        f"Extracting GOS parameters from {params_path} for {param_key}={param_val} ({param_type})"
    )
    params = utils.extract_params(params_path, param_key, param_val, type=param_type)
    n_gos = int(params["n"])
    print(f"Calculating Global Optimal Solution with n={n_gos}")
    ugos, vgos, Sgos = other_methods.calculate_prediction_gos(
        dTds1o, dTds2o, dTdto, n_gos
    )
    print("Calculating GOS metrics")
    metrics_gos = metrics.overview(u, v, ugos, vgos, print_bool=False)
    return ugos, vgos, Sgos, metrics_gos


def plot_and_save_predictions(
    LON,
    LAT,
    To,
    dTds1o,
    dTds2o,
    dTdto,
    muSstar,
    Kxstar_vel,
    stdSstar,
    muustar,
    muvstar,
    lonlims,
    latlims,
    u=None,
    v=None,
    S=None,
    ugos=None,
    vgos=None,
    Sgos=None,
    filename=None,
):
    print(f"Plotting predictions and saving to {filename}")
    fig, ax = plot_helper.plot_predictions_osse(
        LON,
        LAT,
        To,
        dTds1o,
        dTds2o,
        dTdto,
        muSstar,
        Kxstar_vel,
        stdSstar,
        muustar,
        muvstar,
        lonlims,
        latlims,
        u=u,
        v=v,
        S=S,
        ugos=ugos,
        vgos=vgos,
        Sgos=Sgos,
        pscale=4,
        nx=17,
        ny=17,
        plimdTdt=[-2.4e-6, -0.4e-6],
        return_fig=True,
    )
    fig.savefig(filename, bbox_inches="tight", dpi=300)


def store_transect_data(lon, To, v, muvstar, stdvstar):
    print("Storing transect data")
    return {
        "lon": lon,
        "maskc": np.isnan(To[25, :]),
        "v": v[25, :],
        "muvstar": muvstar[25, :],
        "stdvstar": stdvstar[25, :],
    }


def experiment_fully_observed():
    print("\n--- Running Fully Observed Noiseless Experiment ---")
    data = load_and_prepare_dataset(
        "1_preproc_data/proc_data/suntans_1h.nc",
        "time",
        np.datetime64("2014-02-19T18:00:00"),
    )
    muustar, muvstar, muSstar, stdustar, stdvstar, stdSstar, Kxstar_vel, metrics_gp = (
        run_gp_regression_and_metrics(
            data["dTds1o"],
            data["dTds2o"],
            data["dTdto"],
            "2_covariance_parameter_estimation/outputs/noise_gp_obs_t.csv",
            "noise_sd",
            0,
            "gp",
            data["X"],
            data["Y"],
            data["time_step"],
            data["u"],
            data["v"],
        )
    )
    ugos, vgos, Sgos, metrics_gos = run_global_optimal_solution(
        data["dTds1o"],
        data["dTds2o"],
        data["dTdto"],
        "2_covariance_parameter_estimation/outputs/noise_gos_t.csv",
        "noise_sd",
        0,
        "gos",
        data["u"],
        data["v"],
    )
    plot_and_save_predictions(
        data["LON"],
        data["LAT"],
        data["To"],
        data["dTds1o"],
        data["dTds2o"],
        data["dTdto"],
        muSstar,
        Kxstar_vel,
        stdSstar,
        muustar,
        muvstar,
        lonlims,
        latlims,
        u=data["u"],
        v=data["v"],
        S=data["S"],
        ugos=ugos,
        vgos=vgos,
        Sgos=Sgos,
        filename="3_observing_system_simulation_experiment/outputs/osse_instance_fully_observed.png",
    )
    transect = store_transect_data(
        data["lon"], data["To"], data["v"], muvstar, stdvstar
    )
    return transect, metrics_gp, metrics_gos


def experiment_measurement_error(noise=0.005):
    print(f"\n--- Running Measurement Error Experiment (noise={noise}) ---")
    data = load_and_prepare_dataset(
        "1_preproc_data/proc_data/suntans_measurement_error.nc", "sigma_tau", noise
    )
    muustar, muvstar, muSstar, stdustar, stdvstar, stdSstar, Kxstar_vel, metrics_gp = (
        run_gp_regression_and_metrics(
            data["dTds1o"],
            data["dTds2o"],
            data["dTdto"],
            "2_covariance_parameter_estimation/outputs/noise_gp_obs_t.csv",
            "noise_sd",
            noise,
            "gp",
            data["X"],
            data["Y"],
            data["time_step"],
            data["u"],
            data["v"],
        )
    )
    ugos, vgos, Sgos, metrics_gos = run_global_optimal_solution(
        data["dTds1o"],
        data["dTds2o"],
        data["dTdto"],
        "2_covariance_parameter_estimation/outputs/noise_gos_t.csv",
        "noise_sd",
        noise,
        "gos",
        data["u"],
        data["v"],
    )
    plot_and_save_predictions(
        data["LON"],
        data["LAT"],
        data["To"],
        data["dTds1o"],
        data["dTds2o"],
        data["dTdto"],
        muSstar,
        Kxstar_vel,
        stdSstar,
        muustar,
        muvstar,
        lonlims,
        latlims,
        u=data["u"],
        v=data["v"],
        S=data["S"],
        ugos=ugos,
        vgos=vgos,
        Sgos=Sgos,
        filename="3_observing_system_simulation_experiment/outputs/osse_instance_noise.png",
    )
    transect = store_transect_data(
        data["lon"], data["To"], data["v"], muvstar, stdvstar
    )
    return transect, metrics_gp, metrics_gos


def experiment_dense_cloud(coverage_dense=0.3):
    print(f"\n--- Running Dense Cloud Experiment (coverage_dense={coverage_dense}) ---")
    data = load_and_prepare_dataset(
        "1_preproc_data/proc_data/suntans_dense_cloud.nc",
        "coverage_dense",
        coverage_dense,
    )
    muustar, muvstar, muSstar, stdustar, stdvstar, stdSstar, Kxstar_vel, metrics_gp = (
        run_gp_regression_and_metrics(
            data["dTds1o"],
            data["dTds2o"],
            data["dTdto"],
            "2_covariance_parameter_estimation/outputs/cloud_dense_gp_obs_t.csv",
            "coverage_dense",
            coverage_dense,
            "gp",
            data["X"],
            data["Y"],
            data["time_step"],
            data["u"],
            data["v"],
        )
    )
    plot_and_save_predictions(
        data["LON"],
        data["LAT"],
        data["To"],
        data["dTds1o"],
        data["dTds2o"],
        data["dTdto"],
        muSstar,
        Kxstar_vel,
        stdSstar,
        muustar,
        muvstar,
        lonlims,
        latlims,
        u=data["u"],
        v=data["v"],
        S=data["S"],
        filename="3_observing_system_simulation_experiment/outputs/osse_instance_dense_cloud.png",
    )
    transect = store_transect_data(
        data["lon"], data["To"], data["v"], muvstar, stdvstar
    )
    return transect, metrics_gp


def experiment_sparse_cloud(coverage_sparse=0.3):
    print(
        f"\n--- Running Sparse Cloud Experiment (coverage_sparse={coverage_sparse}) ---"
    )
    data = load_and_prepare_dataset(
        "1_preproc_data/proc_data/suntans_sparse_cloud.nc",
        "coverage_sparse",
        coverage_sparse,
    )
    muustar, muvstar, muSstar, stdustar, stdvstar, stdSstar, Kxstar_vel, metrics_gp = (
        run_gp_regression_and_metrics(
            data["dTds1o"],
            data["dTds2o"],
            data["dTdto"],
            "2_covariance_parameter_estimation/outputs/cloud_sparse_gp_obs_t.csv",
            "coverage_sparse",
            coverage_sparse,
            "gp",
            data["X"],
            data["Y"],
            data["time_step"],
            data["u"],
            data["v"],
        )
    )
    plot_and_save_predictions(
        data["LON"],
        data["LAT"],
        data["To"],
        data["dTds1o"],
        data["dTds2o"],
        data["dTdto"],
        muSstar,
        Kxstar_vel,
        stdSstar,
        muustar,
        muvstar,
        lonlims,
        latlims,
        u=data["u"],
        v=data["v"],
        S=data["S"],
        filename="3_observing_system_simulation_experiment/outputs/osse_instance_sparse_cloud.png",
    )
    transect = store_transect_data(
        data["lon"], data["To"], data["v"], muvstar, stdvstar
    )
    return transect, metrics_gp


def plot_and_save_transects(
    transect_fully_observed,
    transect_measurement_error,
    transect_dense_cloud,
    transect_sparse_cloud,
):
    print("\n--- Plotting transects overview ---")
    fig, ax = plot_helper.plot_transects(
        transect_fully_observed,
        transect_measurement_error,
        transect_dense_cloud,
        transect_sparse_cloud,
        lonlims,
        [-0.35, 0.25],
        return_fig=True,
    )
    file_name = "3_observing_system_simulation_experiment/outputs/osse_instance_overview_transect.png"
    print(f"Saving transects overview to {file_name}")
    fig.savefig(
        file_name,
        bbox_inches="tight",
        dpi=300,
    )


def make_metric_overview_gp(
    metrics_gp_fully_observed,
    metrics_gos_fully_observed,
    metrics_gp_measurement_error,
    metrics_gos_measurement_error,
    metrics_gp_dense_cloud,
    metrics_gp_sparse_cloud,
):
    print("\n--- Creating metric overview ---")
    overview = []
    overview.append(["Experiment", "GP RMSE (m/s)", "GP coverage90 (%)", "GOS RMSE (m/s)"])
    overview.append(
        [
            "Fully observed",
            "{:.4e}".format(metrics_gp_fully_observed["RMSE"]),
            "{:.4f}".format(metrics_gp_fully_observed["coverage90"]*100),
            "{:.4e}".format(metrics_gos_fully_observed["RMSE"]),
        ]
    )
    overview.append(
        [
            "Measurement error",
            "{:.4e}".format(metrics_gp_measurement_error["RMSE"]),
            "{:.4f}".format(metrics_gp_measurement_error["coverage90"]*100),
            "{:.4e}".format(metrics_gos_measurement_error["RMSE"]),
        ]
    )
    overview.append(
        [
            "Dense cloud",
            "{:.4e}".format(metrics_gp_dense_cloud["RMSE"]),
            "{:.4f}".format(metrics_gp_dense_cloud["coverage90"]*100),
            "-",
        ]
    )
    overview.append(
        [
            "Sparse cloud",
            "{:.4e}".format(metrics_gp_sparse_cloud["RMSE"]),
            "{:.4f}".format(metrics_gp_sparse_cloud["coverage90"]*100),
            "-",
        ]
    )

    df_overview = pd.DataFrame(overview[1:], columns=overview[0])
    file_name = "3_observing_system_simulation_experiment/outputs/osse_instance_overview_metrics.csv"
    print(f"Saving metric overview to {file_name}")
    df_overview.to_csv(file_name, index=False)

def main():
    print("--- Running individual cases for OSSE ---")
    transect_fully_observed, metrics_gp_fully_observed, metrics_gos_fully_observed = (
        experiment_fully_observed()
    )
    (
        transect_measurement_error,
        metrics_gp_measurement_error,
        metrics_gos_measurement_error,
    ) = experiment_measurement_error(noise=0.005)
    transect_dense_cloud, metrics_gp_dense_cloud = experiment_dense_cloud(
        coverage_dense=0.3
    )
    transect_sparse_cloud, metrics_gp_sparse_cloud = experiment_sparse_cloud(
        coverage_sparse=0.3
    )

    plot_and_save_transects(
        transect_fully_observed,
        transect_measurement_error,
        transect_dense_cloud,
        transect_sparse_cloud,
    )

    make_metric_overview_gp(
        metrics_gp_fully_observed,
        metrics_gos_fully_observed,
        metrics_gp_measurement_error,
        metrics_gos_measurement_error,
        metrics_gp_dense_cloud,
        metrics_gp_sparse_cloud,
    )

    print("\nAll experiments completed and figures saved.")


if __name__ == "__main__":
    main()
