"""
Metrics module for model evaluation.

This module provides functions to compute various scoring metrics for evaluating
model performance, including RMSE, correlation coefficients, and probabilistic metrics.
"""

from scipy.stats import norm
import numpy as np

# functions to compute various scoring metrics for model evaluation

def overview(u0, v0, muu, muv, stdu=None, stdv=None, print_bool=False):
    """
    Computes an overview of various metrics for model evaluation.

    Parameters
    ----------
    u0 : ndarray
        True values of the u-component of velocity.
    v0 : ndarray
        True values of the v-component of velocity.
    muu : ndarray
        Predicted mean values of the u-component of velocity.
    muv : ndarray
        Predicted mean values of the v-component of velocity.
    stdu : ndarray, optional
        Standard deviation of the predicted u-component (if probabilistic).
    stdv : ndarray, optional
        Standard deviation of the predicted v-component (if probabilistic).
    print_bool : bool, optional
        Whether to print the computed metrics.

    Returns
    -------
    dict
        Dictionary containing computed metrics such as RMSE, rho_mag, phi, crps_norm, and coverage90.
    """

    # compute metric overview
    metric = {} # allocate


    # flatten and drop nan values
    idx = np.logical_not(np.isnan(u0) | np.isnan(v0) | np.isnan(muu) | np.isnan(muv))
    u0 = u0[idx]
    v0 = v0[idx]
    muu = muu[idx]
    muv = muv[idx]
    
    metric['RMSE'] = RMSE(u0,v0,muu,muv)
    metric['rho_mag'] = rho_mag(u0,v0,muu,muv)
    metric['phi'] = phi(u0,v0,muu,muv)
    if (stdu is not None) and (stdv is not None): # when probablistic
        stdu = stdu[idx]
        stdv = stdv[idx]
        x = np.hstack([u0,v0])
        mu = np.hstack([muu,muv])
        sigma = np.hstack([stdu,stdv])
        metric['crps_norm'] = crps_norm(x,mu,sigma)
        metric['coverage90'] = coverage90(x,mu,sigma)
    if print_bool:
        for i in metric:
            print(f'{i:11}: {metric[i]:.4g}')
    return metric


def crps_norm(x, mu, sigma):
    """
    Computes the Continuous Ranked Probability Score (CRPS) for normally distributed forecasts.

    Parameters
    ----------
    x : ndarray
        Observed values.
    mu : ndarray
        Predicted mean values.
    sigma : ndarray
        Predicted standard deviations.

    Returns
    -------
    float
        The computed CRPS value.
    """
    
    # compute Continuous Ranked Probability Score (CRPS) for normally distributed forecasts
    sx = (x - mu) / sigma
    return np.mean(sigma * (sx * (2 * norm.cdf(sx) - 1) + 2 * norm.pdf(sx) - 1 / np.sqrt(np.pi)))

def RMSE(u0, v0, up, vp):
    """
    Computes the Root Mean Square Error (RMSE) for velocity vectors.

    Parameters
    ----------
    u0 : ndarray
        True values of the u-component of velocity.
    v0 : ndarray
        True values of the v-component of velocity.
    up : ndarray
        Predicted values of the u-component of velocity.
    vp : ndarray
        Predicted values of the v-component of velocity.

    Returns
    -------
    float
        The computed RMSE value.
    """

    # compute Root Mean Square Error (RMSE) for velocity vectors
    return np.sqrt(0.5 * np.nanmean((u0 - up)**2 + (v0 - vp)**2))



def coverage90(x, mu, sigma):
    """
    Computes the coverage probability of a 90% confidence interval.

    Parameters
    ----------
    x : ndarray
        Observed values.
    mu : ndarray
        Predicted mean values.
    sigma : ndarray
        Predicted standard deviations.

    Returns
    -------
    float
        The proportion of observed values within the 90% confidence interval.
    """

    # compute the coverage probability of a 90% confidence interval
    lower = mu - 1.64 * sigma


    upper = mu + 1.64 * sigma
    return np.sum((x < upper) & (x > lower)) / len(x)

def rho_vec(u0, v0, up, vp):
    """
    Computes the complex correlation coefficient between true and predicted velocity vectors.

    Parameters
    ----------
    u0 : ndarray
        True values of the u-component of velocity.
    v0 : ndarray
        True values of the v-component of velocity.
    up : ndarray
        Predicted values of the u-component of velocity.
    vp : ndarray
        Predicted values of the v-component of velocity.

    Returns
    -------
    complex
        The complex correlation coefficient.
    """

    omega = u0 + v0*1j


    omega_c = u0 - v0*1j
    omega_pred = up + vp*1j
    omega_c_pred = up - vp*1j
    return np.mean(omega_pred*omega_c)/np.sqrt(np.real(np.mean(omega_pred*omega_c_pred)*np.mean(omega*omega_c)))

def rho_mag(u0, v0, up, vp):
    """
    Computes the magnitude of the complex correlation coefficient between true and predicted velocity vectors.

    Parameters
    ----------
    u0 : ndarray
        True values of the u-component of velocity.
    v0 : ndarray
        True values of the v-component of velocity.
    up : ndarray
        Predicted values of the u-component of velocity.
    vp : ndarray
        Predicted values of the v-component of velocity.

    Returns
    -------
    float
        The magnitude of the complex correlation coefficient.
    """

    rho = rho_vec(u0, v0, up, vp)


    re = np.real(rho)
    im = np.imag(rho)
    return np.sqrt(re**2+im**2)

def phi(u0, v0, up, vp):
    """
    Computes the angle (in degrees) of the complex correlation coefficient between true and predicted velocity vectors.

    Parameters
    ----------
    u0 : ndarray
        True values of the u-component of velocity.
    v0 : ndarray
        True values of the v-component of velocity.
    up : ndarray
        Predicted values of the u-component of velocity.
    vp : ndarray
        Predicted values of the v-component of velocity.

    Returns
    -------
    float
        The angle (in degrees) of the complex correlation coefficient.
    """

    rho = rho_vec(u0, v0, up, vp)


    re = np.real(rho)
    im = np.imag(rho)
    return np.rad2deg(np.arctan2(im,re))