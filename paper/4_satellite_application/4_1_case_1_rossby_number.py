import xarray as xr
import numpy as np
from matplotlib import rc

from inversion_sst_gp import (
    plot_helper,
    utils,
    gp_regression,
)

# Matplotlib configuration
rc("font", family="serif", serif=["Computer Modern"])
rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{amsmath}")

def main():
    print("--- Starting case 1: Rossby number ---")
    lonlims = (115, 118)
    latlims = (-15.5, -12.5)

    print("Loading Himawari dataset")
    time_himawari_str = "2023-09-22T04:00:00"
    ds = xr.open_dataset("1_preproc_data/proc_data/himawari.nc").sel(
        time=np.datetime64(time_himawari_str)
    )
    time_step = ds.time_step.item()
    lon, lat, To, dTdto = (ds[var].values for var in ("lon", "lat", "T", "dTdt"))
    lonc, latc, X, Y, LON, LAT = utils.calculate_grid_properties(lon, lat)
    dTds1o, dTds2o = utils.finite_difference_2d(X, Y, To)

    print("Loading altimetry current data")
    time_altimetry_str = "2023-09-22T00:00:00"
    ds_altimetry = xr.open_dataset("1_preproc_data/proc_data/altimeter_currents.nc").sel(
        time=time_altimetry_str
    )
    lonr, latr, ugos, vgos = (
        ds_altimetry[var].values for var in ("lon", "lat", "ugos", "vgos")
    )
    _, _, Xr, Yr, LONr, LATr = utils.calculate_grid_properties(lonr, latr)

    print("Extracting GP regression parameters")
    params_fully_obs_gp = utils.extract_params(
        "2_covariance_parameter_estimation/outputs/satellite_gp_obs_t.csv",
        "time",
        time_himawari_str,
        type="gp",
    )
    print("Calculating GP regression prediction")
    muustar, muvstar, muSstar, stdustar, stdvstar, stdSstar, Kxstar_vel = (
        gp_regression.calculate_prediction_gpregression(
            dTds1o, dTds2o, dTdto, params_fully_obs_gp, X, Y, time_step
        )
    )

    print("Plotting predictions")
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
        LONr=LONr,
        LATr=LATr,
        ur=ugos,
        vr=vgos,
        pscale=7,
        nx=16,
        ny=16,
        nxr=16,
        nyr=16,
        return_fig=True,
    )
    
    file_name = "4_satellite_application/outputs/satellite_case_1_prediction.png"
    print(f"Saving figure to {file_name}")
    fig.savefig(
        file_name,
        bbox_inches="tight",
        dpi=300,
    )
    
    print("Calculating dynamic Rossby number")
    Ro = utils.calculate_dynamic_rossby_number(X, Y, muustar, muvstar, LAT)
    Ror = utils.calculate_dynamic_rossby_number(Xr, Yr, ugos, vgos, LATr)

    print("Plotting dynamic Rossby number")
    fig, ax = plot_helper.plot_dynamic_rossby(
        LON,
        LAT,
        Ro,
        lonlims=lonlims,
        latlims=latlims,
        Ro_max=1,
        LONr=LONr,
        LATr=LATr,
        Ror=Ror,
        return_fig=True,
    )
    
    file_name = "4_satellite_application/outputs/satellite_case_1_rossby_number.png"
    print(f"Saving Rossby number figure to {file_name}")
    fig.savefig(
        file_name,
        bbox_inches="tight",
        dpi=300,
    )
    
    print('Finished processing and saving figures')

if __name__ == "__main__":
    main()
