import pandas as pd
from inversion_sst_gp import plot_helper
from matplotlib import rc

# Matplotlib configuration
rc("font", family="serif", serif=["Computer Modern"])
rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{amsmath}")

# Helper functions
def load_time_series_data():
    print("Loading time series scoring data")
    dfs = {}
    dfs['1h_gp_obs_t'] = pd.read_csv("2_covariance_parameter_estimation/outputs/time_1h_gp_obs_t.csv")
    dfs['1h_gp_num_t'] = pd.read_csv("2_covariance_parameter_estimation/outputs/time_1h_gp_num_t.csv")
    dfs['1h_gp_num_t1'] = pd.read_csv("2_covariance_parameter_estimation/outputs/time_1h_gp_num_t1.csv")
    dfs['1h_gos_t'] = pd.read_csv("2_covariance_parameter_estimation/outputs/time_1h_gos_t.csv")

    dfs['24h_gp_obs_t'] = pd.read_csv("2_covariance_parameter_estimation/outputs/time_24h_gp_obs_t.csv")
    dfs['24h_gp_num_t'] = pd.read_csv("2_covariance_parameter_estimation/outputs/time_24h_gp_num_t.csv")
    dfs['24h_gp_num_t1'] = pd.read_csv("2_covariance_parameter_estimation/outputs/time_24h_gp_num_t1.csv")
    dfs['24h_gos_t'] = pd.read_csv("2_covariance_parameter_estimation/outputs/time_24h_gos_t.csv")
    return dfs


def plot_time_series_metrics(dfs):
    print("Plotting time series metrics figure")
    fig, ax = plot_helper.plot_time_metrics(
        dfs['1h_gp_obs_t'],
        dfs['1h_gp_num_t'],
        dfs['1h_gp_num_t1'],
        dfs['1h_gos_t'],
        dfs['24h_gp_obs_t'],
        dfs['24h_gp_num_t'],
        dfs['24h_gp_num_t1'],
        dfs['24h_gos_t'],
        return_fig=True,
    )
    file_name = "3_observing_system_simulation_experiment/outputs/osse_metrics_time_48h.png"
    print(f"Saving time series metrics figure to {file_name}\n")
    fig.savefig(
        file_name,
        bbox_inches="tight",
        dpi=300,
    )


def load_noise_experiment_data():
    print("Loading measurement error noise experiment data")
    dfs = {}
    dfs['noise_gp_obs_t'] = pd.read_csv("2_covariance_parameter_estimation/outputs/noise_gp_obs_t.csv")
    dfs['noise_gp_num_t'] = pd.read_csv("2_covariance_parameter_estimation/outputs/noise_gp_num_t.csv")
    dfs['noise_gos_t'] = pd.read_csv("2_covariance_parameter_estimation/outputs/noise_gos_t.csv")
    return dfs


def plot_noise_metrics(dfs):
    print("Plotting noise experiment metrics figure")
    fig, ax = plot_helper.plot_noise_metrics(
        dfs['noise_gp_obs_t'],
        dfs['noise_gp_num_t'],
        dfs['noise_gos_t'],
        return_fig=True,
    )
    file_name = "3_observing_system_simulation_experiment/outputs/osse_metrics_time_100d.png"
    print(f"Saving noise metrics figure to {file_name}\n")
    fig.savefig(
        file_name,
        bbox_inches="tight",
        dpi=300,
    )


def load_cloud_experiment_data():
    print("Loading cloud experiment data")
    dfs = {}
    dfs['cloud_dense_gp_obs_t'] = pd.read_csv("2_covariance_parameter_estimation/outputs/cloud_dense_gp_obs_t.csv")
    dfs['cloud_dense_gp_num_t'] = pd.read_csv("2_covariance_parameter_estimation/outputs/cloud_dense_gp_num_t.csv")
    dfs['cloud_sparse_gp_obs_t'] = pd.read_csv("2_covariance_parameter_estimation/outputs/cloud_sparse_gp_obs_t.csv")
    dfs['cloud_sparse_gp_num_t'] = pd.read_csv("2_covariance_parameter_estimation/outputs/cloud_sparse_gp_num_t.csv")
    return dfs


def plot_cloud_metrics(dfs):
    print("Plotting cloud experiment metrics figure")
    fig, ax = plot_helper.plot_cloud_metrics(
        dfs['cloud_dense_gp_num_t'],
        dfs['cloud_dense_gp_obs_t'],
        dfs['cloud_sparse_gp_num_t'],
        dfs['cloud_sparse_gp_obs_t'],
        return_fig=True,
    )
    file_name = "3_observing_system_simulation_experiment/outputs/osse_metrics_clouds.png"
    print(f"Saving cloud metrics figure to {file_name}\n")
    fig.savefig(
        file_name,
        bbox_inches="tight",
        dpi=300,
    )

if __name__ == "__main__":
    print('--- Generating figures for OSSE evaluation metrics ---')
    time_series_dfs = load_time_series_data()
    plot_time_series_metrics(time_series_dfs)

    noise_dfs = load_noise_experiment_data()
    plot_noise_metrics(noise_dfs)

    cloud_dfs = load_cloud_experiment_data()
    plot_cloud_metrics(cloud_dfs)
    
    print("All figures generated successfully")
