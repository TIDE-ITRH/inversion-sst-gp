"""
Gaussian Process Regression Module.

This module provides functionality for performing Gaussian Process (GP) regression
to jointly estimate velocity and scalar fields. It includes the `GPRegressionJoint`
class, which implements a GP model for predicting spatial and temporal derivatives
of temperature fields based on observed data.

Key Features:
- Joint estimation of velocity and scalar fields.
- Support for polynomial basis functions for velocity and scalar fields.
- Efficient computation using distance matrices and block diagonal structures.

Dependencies:
- NumPy for numerical operations.
- SciPy for linear algebra and optimization routines.
- Utilities from the `inversion_sst_gp` package for data mapping.

Typical Use Case:
This module is designed for applications in geophysical and environmental modeling,
where Gaussian Process regression is used to infer velocity and scalar fields from
observational data.
"""

import numpy as np
import xarray as xr
import pandas as pd

from numpy.linalg import multi_dot as mdot

from scipy.linalg import block_diag
# from scipy.sparse import block_diag

from scipy.linalg.lapack import dpotrf, dpotri
from scipy.optimize import minimize, shgo
from scipy.spatial import distance_matrix

from inversion_sst_gp.utils import map_val


class GPRegressionJoint(object):
    """
    Gaussian Process Regression Model for joint estimation of velocity and scalar fields.

    Attributes
    ----------
    dTds1o : ndarray
        Observed spatial derivative of temperature in the first direction.
    dTds2o : ndarray
        Observed spatial derivative of temperature in the second direction.
    dTdto : ndarray
        Observed temporal derivative of temperature.
    tstep : float
        Time step for the simulation.
    X : ndarray
        Grid of x-coordinates.
    Y : ndarray
        Grid of y-coordinates.
    maskp : ndarray
        Mask for prediction locations.
    degreeu, degreev, degreeS : int, optional
        Degrees for polynomial basis functions for velocity and scalar fields.
    """

    def __init__(self, dTds1o, dTds2o, dTdto, tstep, X, Y, maskp, degreeu=2, degreev=2, degreeS=2
    ):  
        """
        Initializes the GPRegressionJoint instance.
        
        Parameters
        ----------
        dTds1o : ndarray
            Observed spatial derivative of temperature in the first direction.
        dTds2o : ndarray
            Observed spatial derivative of temperature in the second direction.
        dTdto : ndarray
            Observed temporal derivative of temperature.
        tstep : float
            Time step for the simulation.
        X : ndarray
            Grid of x-coordinates.
        Y : ndarray
            Grid of y-coordinates.
        maskp : ndarray
            Mask for prediction locations.
        degreeu, degreev, degreeS : int, optional
            Degrees for polynomial basis functions for velocity and scalar fields.
            
        Returns
        -------
        None
        """      
        self.dTds1o = dTds1o
        self.dTds2o = dTds2o
        self.dTdto = dTdto
        self.maskp = maskp
        self.tstep = tstep

        # mask
        self.masko = np.logical_not(
            np.isnan(self.dTds1o) | np.isnan(self.dTds2o) | np.isnan(self.dTdto)
        )

        # dimensions
        self.m = np.sum(self.masko)
        self.n = np.sum(self.maskp)

        # grid
        s = np.stack([X, Y],2)
        sp = s[maskp]
        phiu = phi(sp[:, 0], sp[:, 1], degreeu)
        phiv = phi(sp[:, 0], sp[:, 1], degreev)
        phiS = phi(sp[:, 0], sp[:, 1], degreeS)
        self.phix = block_diag(phiu, phiv, phiS)
        self.d = distance_matrix(sp, sp)
        self.match_mask = self.d == 0

        # construct HA matrix
        self.HA = self.construct_HA()

        # data vector
        self.z = self.dTdto[self.masko]

    def construct_HA(self):
        """
        Constructs the HA matrix, which relates the observed derivatives to the prediction locations.

        Returns
        -------
        ndarray
            The constructed HA matrix.
        """
        # construct HA matrix
        maskp_flat = self.maskp.flatten()
        masko_flat = self.masko.flatten()
        dTds1o_flat = self.dTds1o.flatten()
        dTds2o_flat = self.dTds2o.flatten()
        HA = np.zeros([self.m, 3 * self.n])  # allocate
        idxp = np.where(maskp_flat)[0]  # indices of prediction locations

        j = 0  # iteration index over observation
        for i, idx in enumerate(idxp):  # loop over prediction locations
            if masko_flat[idx]:  # when prediction location is a observation location
                # set HA components
                HA[j, i] = -dTds1o_flat[idx]
                HA[j, i + self.n] = -dTds2o_flat[idx]
                HA[j, i + 2 * self.n] = 1
                j += 1  # next observation
        return HA

    def construct_Kx(self, params):
        """
        Constructs the covariance matrix Kx for the velocity and scalar fields.

        Parameters
        ----------
        params : dict
            Dictionary of parameters for the covariance kernel.

        Returns
        -------
        ndarray
            The block diagonal covariance matrix Kx.
        """
        # construct Kx
        Ku = kernel_matern_3_2_var(
            self.d, self.match_mask, params["sigma_u"], params["l_u"], params["tau_u"]
        )
        Kv = kernel_matern_3_2_var(
            self.d, self.match_mask, params["sigma_v"], params["l_v"], params["tau_v"]
        )
        KS = kernel_matern_3_2_var(
            self.d, self.match_mask, params["sigma_S"], params["l_S"], params["tau_S"]
        )
        return block_diag(Ku, Kv, KS)

    def construct_Kz(self, params, HA, Kx):
        """
        Constructs the covariance matrix Kz for the observed data.

        Parameters
        ----------
        params : dict
            Dictionary of parameters for the covariance kernel.
        HA : ndarray
            The HA matrix.
        Kx : ndarray
            The covariance matrix Kx.

        Returns
        -------
        ndarray
            The covariance matrix Kz.
        """
        # construct Kz
        Ktildetau = params["sigma_tau"] ** 2 / (2 * self.tstep**2) * np.eye(self.m)
        return mdot([HA, Kx, HA.T]) + Ktildetau

    @staticmethod
    def calculate_rlml_z(
        params_val,
        params_key,
        gprm,
        const_params,
        penalty_params,
        share_len,
        share_sigma,
        share_tau,
        solve_log,
    ):
        """
        Calculates the restricted log marginal likelihood of the observed data.

        Parameters
        ----------
        params_val : list
            Values of the parameters to optimize.
        params_key : list
            Keys corresponding to the parameters to optimize.
        gprm : GPRegressionJoint
            The Gaussian Process Regression Model instance.
        const_params : dict
            Dictionary of constant parameters.
        penalty_params : dict
            Dictionary of penalty parameters.
        share_len : bool
            Whether to share length scale parameters between velocity components.
        share_sigma : bool
            Whether to share signal variance parameters between velocity components.
        share_tau : bool
            Whether to share noise variance parameters between velocity components.
        solve_log : bool
            Whether to solve in the log space.

        Returns
        -------
        float
            The restricted log marginal likelihood value.
        """

        # couple variable
        z = gprm.z
        phix = gprm.phix
        HA = gprm.HA

        # make dictionary of parameters
        if solve_log:
            params_val_i = np.exp(params_val)
        else:
            params_val_i = params_val
        params = dict(zip(params_key, params_val_i)) | const_params
        if share_len:
            params["l_v"] = params["l_u"]
        if share_sigma:
            params["sigma_v"] = params["sigma_u"]
        if share_tau:
            params["tau_v"] = params["tau_u"]

        # covariance matrix
        Kx = gprm.construct_Kx(params)  # construct Kx
        Kz = gprm.construct_Kz(params, HA, Kx)  # construct Kz
        Lz = chol(Kz)        
        Qz = chol2inv(Lz)

        # universal kriging
        betacov = cholinv(mdot([phix.T, HA.T, Qz, HA, phix]))
        beta = mdot([betacov, phix.T, HA.T, Qz, z])
        mux = mdot([phix, beta])

        # penalty
        penalty = 0
        for param_name in penalty_params:
            mu, sigma = penalty_params[param_name]
            penalty += penalty_gauss_centre(params[param_name], mu, sigma)

        # compute rlml (withouth constant)
        return (
            -1 / 2 * mdot([(z - mdot([HA, mux])).T, Qz, z - mdot([HA, mux])])
            - np.sum(np.log(np.diag(Lz)))
            - 1 / 2 * np.log(np.linalg.det(mdot([phix.T, HA.T, Qz, HA, phix])))
            + penalty
        )

    @staticmethod
    def calculate_negative_rlml_z(
        params_val,
        params_key,
        gprm,
        const_params,
        penalty_params,
        share_len,
        share_sigma,
        share_tau,
        solve_log,
        callback,
    ):
        """
        calculate negative restricted log marginal likelihood of z

        Parameters
        ----------
        params_val : list
            Values of the parameters to optimize.
        params_key : list
            Keys corresponding to the parameters to optimize.
        gprm : GPRegressionJoint
            The Gaussian Process Regression Model instance.
        const_params : dict
            Dictionary of constant parameters.
        penalty_params : dict
            Dictionary of penalty parameters.
        share_len : bool
            Whether to share length scale parameters between velocity components.
        share_sigma : bool
            Whether to share signal variance parameters between velocity components.
        share_tau : bool
            Whether to share noise variance parameters between velocity components.
        solve_log : bool
            Whether to solve in the log space.
        callback : str
            Callback option for printing progress.

        Returns
        -------
        float
            The negative restricted log marginal likelihood value.
        """

        lml = GPRegressionJoint.calculate_rlml_z(
            params_val,
            params_key,
            gprm,
            const_params,
            penalty_params,
            share_len,
            share_sigma,
            share_tau,
            solve_log,
        )

        if callback != "off":
            if solve_log:
                pval = np.append(np.exp(params_val), lml)  # values to print
            else:
                pval = np.append(params_val, lml)  # values to print

            pstring = "   ".join(
                ["{:<11.5}".format(xi) for xi in pval]
            )  # string to print

            if callback == "compact":
                print(pstring, end="\r")
            elif callback == "on":
                print(pstring)  # print

        return -lml

    def estimate_params(
        self,
        initial_params,
        const_params,
        penalty_params={},
        bounds_params={},
        share_len=False,
        share_sigma=False,
        share_tau=False,
        solve_log=True,
        shgo_bool=False,
        callback="off",
    ):
        """
        Estimate the parameters of the Gaussian Process Regression Model.

        Parameters
        ----------
        initial_params : dict
            Initial guess for the parameters.
        const_params : dict
            Dictionary of constant parameters.
        penalty_params : dict, optional
            Dictionary of penalty parameters for regularization (default is empty).
        bounds_params : dict, optional
            Dictionary of bounds for the parameters (default is empty).
        share_len : bool, optional
            Whether to share length scale parameters between velocity components (default is False).
        share_sigma : bool, optional
            Whether to share signal variance parameters between velocity components (default is False).
        share_tau : bool, optional
            Whether to share noise variance parameters between velocity components (default is False).
        solve_log : bool, optional
            Whether to solve in the log space (default is True).
        shgo_bool : bool, optional
            Whether to use SHGO optimization (default is False).
        callback : str, optional
            Callback option for printing progress (default is "off").

        Returns
        -------
        dict
            The estimated parameters.
        """

        if shgo_bool & (len(bounds_params) == 0):
            print("shgo requires bounds")

        # optimising rlml
        params_key = list(initial_params.keys())
        args = (
            params_key,
            self,
            const_params,
            penalty_params,
            share_len,
            share_sigma,
            share_tau,
            solve_log,
            callback,
        )

        if callback != "off":  # when callback is requested
            pkey = np.append(params_key, "lml")  # header keys
            pkey_comb = "   ".join(
                ["{0: <11}".format(xi) for xi in pkey]
            )  # combine keys into single line
            print(pkey_comb)  # create header

        if solve_log:
            initial_params_val = np.log(np.array(list(initial_params.values())))
        else:
            initial_params_val = np.array(list(initial_params.values()))

        if len(bounds_params) > 0:
            bounds = []
            for param_name in params_key:
                if param_name in bounds_params:
                    boundl, boundu = bounds_params[param_name]
                    if boundl is not None:
                        if solve_log:
                            boundl = np.log(boundl)
                    if boundu is not None:
                        if solve_log:
                            boundu = np.log(boundu)
                    bounds += [[boundl, boundu]]
                else:
                    bounds += [[None, None]]

            if shgo_bool:
                result = shgo(
                    GPRegressionJoint.calculate_negative_rlml_z, bounds, args=args, n=1e3
                )
            else:
                result = minimize(
                    GPRegressionJoint.calculate_negative_rlml_z,
                    initial_params_val,
                    args=args,
                    bounds=bounds,
                )
        else:
            result = minimize(
                GPRegressionJoint.calculate_negative_rlml_z, initial_params_val, args=args
            )

        if solve_log:
            params_val = list(np.exp(result.x))
        else:
            params_val = list(result.x)

        params = dict(zip(params_key, params_val)) | const_params
        if share_len:
            params["l_v"] = params["l_u"]
        if share_sigma:
            params["sigma_v"] = params["sigma_u"]
        if share_tau:
            params["tau_v"] = params["tau_u"]
        return params

    def format_output(self, mux, Kx):
        """
        Format the output of the Gaussian Process regression.

        Parameters:
            mux (ndarray): Mean predictions for the velocity and scalar fields.
            Kx (ndarray): Covariance matrix for the velocity and scalar fields.

        Returns:
            tuple: Formatted outputs including mean predictions, standard deviations,
                   and velocity covariances mapped to the prediction grid.
        """

        # format output

        # standard deviation
        stdx = np.sqrt(np.diag(Kx))

        # outputs
        muu_flat = mux[: self.n]
        muv_flat = mux[self.n : 2 * self.n]
        muS_flat = mux[2 * self.n : 3 * self.n]
        stdu_flat = stdx[: self.n]
        stdv_flat = stdx[self.n : 2 * self.n]
        stdS_flat = stdx[2 * self.n : 3 * self.n]

        # covariance velocity
        Kx_uu = np.diag(Kx[: self.n, : self.n])
        Kx_uv = np.diag(Kx[: self.n, self.n : 2 * self.n])
        Kx_vv = np.diag(Kx[self.n : 2 * self.n, self.n : 2 * self.n])
        Kx_vel_flat = np.stack([[Kx_uu, Kx_uv], [Kx_uv, Kx_vv]])  # stack components
        Kx_vel_flat = np.swapaxes(Kx_vel_flat, 0, 2)  # switch axes order

        # convert to 2d
        muu = map_val(muu_flat, self.maskp)
        muv = map_val(muv_flat, self.maskp)
        muS = map_val(muS_flat, self.maskp)
        stdu = map_val(stdu_flat, self.maskp)
        stdv = map_val(stdv_flat, self.maskp)
        stdS = map_val(stdS_flat, self.maskp)
        Kx_vel = map_val(Kx_vel_flat, self.maskp)
        return muu, muv, muS, stdu, stdv, stdS, Kx_vel

    def predict(self, params, return_prior=False, return_Kxstar=False):
        """
        Perform predictions using the Gaussian Process Regression model.

        Parameters:
            params (dict): Parameters for the covariance kernel.
            return_prior (bool, optional): Whether to return prior predictions (default is False).
            return_Kxstar (bool, optional): Whether to return the posterior covariance matrix (default is False).

        Returns:
            tuple: Predicted mean and standard deviation for velocity and scalar fields,
                   and optionally the prior predictions and posterior covariance matrix.
        """

        # predict using GPRegressionJoint
        Kx = self.construct_Kx(params)  # construct Kx
        Kz = self.construct_Kz(params, self.HA, Kx)  # construct Kz
        Lz = chol(Kz)
        Qz = chol2inv(Lz)

        # universal kriging
        betacov = cholinv(mdot([self.phix.T, self.HA.T, Qz, self.HA, self.phix]))

        # prediction covariance
        kappa_term = self.phix.T - mdot([self.phix.T, self.HA.T, Qz, self.HA, Kx])
        kappa = mdot([kappa_term.T, betacov, kappa_term])
        Kxstar = Kx - mdot([Kx, self.HA.T, Qz, self.HA, Kx]) + kappa

        # prediction mean
        beta = mdot([betacov, self.phix.T, self.HA.T, Qz, self.z])
        mux = mdot([self.phix, beta])
        muxstar = mux - mdot([Kx, self.HA.T, Qz, mdot([self.HA, mux]) - self.z])

        # format output
        muustar, muvstar, muSstar, stdustar, stdvstar, stdSstar, Kxstar_vel = (
            self.format_output(muxstar, Kxstar)
        )

        output = [muustar, muvstar, muSstar, stdustar, stdvstar, stdSstar, Kxstar_vel]
        if return_prior:  # return prior and posterior
            muu, muv, muS, stdu, stdv, stdS, Kx_vel = self.format_output(mux, Kx)
            output += [muu, muv, muS, stdu, stdv, stdS, Kx_vel]
        if return_Kxstar:
            output += [Kxstar]
        return tuple(output)


def calculate_prediction_gpregression(dTds1, dTds2, dTdt, params, X, Y, tstep, maskp = None, degreeu=2, degreev=2, degreeS=2, return_Kxstar=False):

    """
    Calculate predictions using Gaussian Process regression.

    Parameters:
        dTds1 (ndarray): Observed spatial derivative of temperature in the first direction.
        dTds2 (ndarray): Observed spatial derivative of temperature in the second direction.
        dTdt (ndarray): Observed temporal derivative of temperature.
        params (dict): Parameters for the covariance kernel.
        X (ndarray): Grid of x-coordinates.
        Y (ndarray): Grid of y-coordinates.
        tstep (float): Time step for the simulation.
        maskp (ndarray, optional): Mask for prediction locations (default is None).
        degreeu (int, optional): Degree for polynomial basis functions for the u-component (default is 2).
        degreev (int, optional): Degree for polynomial basis functions for the v-component (default is 2).
        degreeS (int, optional): Degree for polynomial basis functions for the scalar field (default is 2).
        return_Kxstar (bool, optional): Whether to return the posterior covariance matrix (default is False).

    Returns:
        tuple: Predicted mean and standard deviation for velocity and scalar fields,
               and optionally the posterior covariance matrix.
    """

    if maskp is None:


        maskp = np.ones_like(dTds1, dtype=bool)
        
    # GP regression
    gprm = GPRegressionJoint(
        dTds1,
        dTds2,
        dTdt,
        tstep,
        X, 
        Y,
        maskp,
        degreeu=degreeu,
        degreev=degreev,
        degreeS=degreeS,
    )
    
    if not return_Kxstar:
        muustar, muvstar, muSstar, stdustar, stdvstar, stdSstar, Kxstar_vel = gprm.predict(
            params, return_prior=False, return_Kxstar=False
        )
        return muustar, muvstar, muSstar, stdustar, stdvstar, stdSstar, Kxstar_vel
    else:
        muustar, muvstar, muSstar, stdustar, stdvstar, stdSstar, Kxstar_vel, Kxstar = gprm.predict(
            params, return_prior=False, return_Kxstar=True
        )
        return muustar, muvstar, muSstar, stdustar, stdvstar, stdSstar, Kxstar_vel, Kxstar
    
    
def predict_scene(ds, params, mask=None, full_cov=False):
    """Predict currents using GP regression for an xarray dataset.

    Args:
        ds (xarray.Dataset): Input dataset containing SST gradients.
        params (dict): Dictionary of GP regression parameters.
    Returns:
        xarray.Dataset: Dataset containing predicted currents and uncertainties.
    """
    
    muustar, muvstar, muSstar, stdustar, stdvstar, stdSstar, Kxstar_vel = (calculate_prediction_gpregression(
            ds['dTdx'].values, ds['dTdy'].values, ds['dTdt'].values, params, ds['X'].values, ds['Y'].values, ds['time_step'].values, maskp=mask))

    ds['mu_u'] = (('lat', 'lon'), muustar)
    ds['mu_v'] = (('lat', 'lon'), muvstar)
    ds['mu_S'] = (('lat', 'lon'), muSstar)
    ds['std_u'] = (('lat', 'lon'), stdustar)
    ds['std_v'] = (('lat', 'lon'), stdvstar)
    ds['std_S'] = (('lat', 'lon'), stdSstar)
    ds['K_uv'] = (('lat', 'lon'), Kxstar_vel[:, :, 0, 1])

    if full_cov:
        return ds, Kxstar_vel
    else:
        return ds


# Metric functions
def run_gprm_optim(time_str, dTds1, dTds2, dTdt, X, Y, tstep, prop, mask=None, callback="on"):
    if mask is None:
        mask = np.ones_like(X, dtype=bool)
    gpr_m = GPRegressionJoint(dTds1, dTds2, dTdt, tstep, X, Y, mask)
    est_params = gpr_m.estimate_params(**prop, callback=callback)
    return {"step": time_str,
            "est_params": est_params}
    
    
def run_gprm_optim_xr(ds, rlml_params, mask=None, callback="on"):
    results = run_gprm_optim(
    str(ds.time.values), ds['dTdx'].values, ds['dTdy'].values, ds['dTdt'].values, ds['X'].values, ds['Y'].values, ds['time_step'].values, rlml_params, mask=mask, callback=callback)
    return results


def fit_series(ds, init_params, refit_interval, coverage=0.8, save_steps=False, save_name=None, mask=None, callback="on"):
    
    if save_steps:
        assert save_name is not None
        
    refit_ticker = ds['time'][0].values - refit_interval
    
    results_step = []
    for t in range(len(ds['time'].values)):
        if ds['time'][t].values - refit_interval >= refit_ticker:
            ds_t = ds.isel(time=t)

            # Proceed if all coverage is good
            if check_coverage(ds_t, coverage=coverage):
                
                print("Refitting hyperparameters")
                refit_ticker = ds['time'][t].values
                res_t = run_gprm_optim_xr(ds.isel(time=t), init_params, mask=mask, callback=callback) 
            else:
                res_t = {"step": str(ds['time'][t].values),
                         "est_params": dict(zip(list(init_params['initial_params'].keys()), np.full(len(init_params['initial_params'].keys()), np.nan)))}   
                res_t['est_params']['l_v'] = res_t['est_params']['l_u']
                res_t['est_params']["sigma_v"] = res_t['est_params']["sigma_u"]
                res_t['est_params']["tau_v"] = res_t['est_params']["tau_u"]   
                print("Not enough data to fit")
        else:
            res_t['step'] = str(ds['time'][t].values)
        results_step.append(res_t)

        if save_steps:
            df = pd.DataFrame(data=res_t['est_params'], index=pd.Series(res_t['step'], name='time'))
            df.index = pd.to_datetime(df.index)        
            if t==0:
                ds_save = xr.Dataset.from_dataframe(df)
                ds_save.to_netcdf(path=save_name)
            else:
                ds_loop = xr.Dataset.from_dataframe(df)
                ds_save = xr.concat([ds_save, ds_loop], dim='time')
                ds_save.to_netcdf(path=save_name)
            print(f'Saved results for time step {ds['time'].values[t]}')
            
    return results_step



def predict_series(ds, params, mask=None, coverage=0.8):
    """Predict currents using GP regression for an xarray dataset time series.

    Args:
        ds (xarray.Dataset): Input dataset containing SST gradients.
        params (dict): Dictionary of GP regression parameters.
    Returns:
        xarray.Dataset: Dataset containing predicted currents and uncertainties.
    """
    ds_list = []

    ## Loop through time 
    for ii, t in enumerate(ds['time']):
        ds_t = ds.sel(time=t)

        # Proceed if all coverage is good
        if check_coverage(ds_t, coverage=coverage):
                            
            # Check if params is dict or list of dicts
            if isinstance(params, list):
                if not np.isnan(params[ii]['est_params']['sigma_u']):
                    ds_pred = predict_scene(ds_t, params[ii]['est_params'], mask=mask)
                    ds_list.append(ds_pred)        
            
            elif isinstance(params, dict):
                ds_pred = predict_scene(ds_t, params, mask=mask)
                ds_list.append(ds_pred)
                    
            elif isinstance(params, xr.Dataset):
                if not np.isnan(params['sigma_u'].sel(time=t).values):
                    ds_p = params.sel(time=t)
                    ds_p_dict = {var: ds_p[var].item() for var in ds_p.data_vars}
                    ds_pred = predict_scene(ds_t, ds_p_dict, mask=mask)
                    ds_list.append(ds_pred)
            else:
                raise ValueError("params must be a dict or list of dicts")
    
    return xr.concat(ds_list, dim='time')



def check_coverage(ds_step, coverage=0.8):
    """Check coverage of data in an xarray dataset time series.

    Args:
        ds_step (xarray.Dataset): Input dataset containing SST gradients.
        coverage (float): Minimum required coverage ratio.
    Returns:
        list: List of booleans indicating coverage status for each time step.
    """
    cov_vars = ['T', 'dTdt', 'dTdx', 'dTdy']

    # Check for at least coverage ratio (needs to be satisfied for T, dTdt, dTdx, dTdy)
    step_coverage = np.full(len(cov_vars), True)
    for iivar, var in enumerate(cov_vars):
        nans_frac = np.sum(np.isnan(ds_step[var].values)) / len(ds_step[var].values.flatten())
        if nans_frac >= (1 - coverage):
            step_coverage[iivar] = False
    
    return np.all(step_coverage)
    

def chol(M):
    """
    Perform Cholesky decomposition of a matrix.

    Parameters
    ----------
    M : ndarray
        Symmetric positive-definite matrix to decompose.

    Returns
    -------
    ndarray
        Lower triangular matrix resulting from the decomposition.
    """
    return dpotrf(M, 1)[0]

def chol2inv(L):
    """
    Compute the inverse of a matrix using its Cholesky decomposition.

    Parameters
    ----------
    L : ndarray
        Lower triangular matrix from Cholesky decomposition.

    Returns
    -------
    ndarray
        Inverse of the original matrix.
    """
    inv = dpotri(L, 1)[0]
    inv += np.tril(inv, k=-1).T
    return inv

def cholinv(M):
    """
    Compute the inverse of a matrix directly through Cholesky decomposition.

    Parameters
    ----------
    M : ndarray
        Symmetric positive-definite matrix to invert.

    Returns
    -------
    ndarray
        Inverse of the matrix.
    """
    L = chol(M)
    return chol2inv(L)

def kernel_matern_3_2_var(d, match_mask, sigma, ls, tau):
    """
    Compute the Matern covariance function (nu=3/2) with additional variance.

    Parameters
    ----------
    d : ndarray
        Distance matrix between points.
    match_mask : ndarray
        Mask for matching observation locations.
    sigma : float
        Signal variance parameter.
    ls : float
        Length scale parameter.
    tau : float
        Noise variance parameter.

    Returns
    -------
    ndarray
        Covariance matrix with Matern kernel and additional variance.
    """
    matern = sigma**2 * (1 + np.sqrt(3) * d / ls) * np.exp(-np.sqrt(3) * d / ls)
    var = tau**2 * match_mask
    return matern + var

def penalty_gauss_centre(theta, mu, sigma):
    """
    Compute Gaussian penalty centered at a given mean.

    Parameters
    ----------
    theta : float
        Parameter value.
    mu : float
        Mean value for the penalty.
    sigma : float
        Standard deviation for the penalty.

    Returns
    -------
    float
        Penalty value.
    """
    return -np.log(sigma) - 0.5 * (theta - mu) ** 2 / sigma**2

# def linear_mean(s, M, R1, R2):
#     """
#     Compute a linear mean function.

#     Parameters
#     ----------
#     s : ndarray
#         Spatial coordinates (X, Y).
#     M : float
#         Intercept term.
#     R1 : float
#         Coefficient for the first spatial dimension.
#     R2 : float
#         Coefficient for the second spatial dimension.

#     Returns
#     -------
#     ndarray
#         Linear mean values at the given coordinates.
#     """
#     s1 = s[:, 0]
#     s2 = s[:, 1]
#     return M + R1 * s1 + R2 * s2

def phi(s1, s2, degree):
    """
    Compute polynomial basis functions up to a given degree.

    Parameters
    ----------
    s1 : ndarray
        First spatial coordinate.
    s2 : ndarray
        Second spatial coordinate.
    degree : int
        Degree of the polynomial basis functions.

    Returns
    -------
    ndarray
        Polynomial basis functions evaluated at the given coordinates.
    """
    N = len(s1)
    constant = [np.ones(N)]
    component1 = [(s1**i) for i in np.arange(1, degree)]
    component2 = [(s2**i) for i in np.arange(1, degree)]
    return np.stack(constant + component1 + component2).T


def get_default_params():
    """
    Get default parameters for the Gaussian Process regression.

    Returns
    -------
    dict
        Default parameters for the covariance kernel.
    """

    initial_params = {
        "sigma_u": 9e-2,
        "l_u": 3e4,
        "tau_u": 1e-2,
        "sigma_v": 9e-2,
        "l_v": 3e4,
        "tau_v": 1e-2,
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
    
    return prop_sat