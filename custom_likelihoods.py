#!/usr/bin/python3


import numpy as np


def MS_DEB_likelihood_function(theta, active_class, error_lists, uncertainty_method='parameterisation'):
    """
    Compute the log-likelihood for a detached eclipsing binary (DEB) stellar model
    using an ML regressor, combining observational and ML uncertainties.

    For each regression output, retrieves the ML-predicted value and its residual
    uncertainty, then evaluates a Gaussian log-likelihood against the observed data.

    Parameters
    ----------
    theta : array-like of shape (n_parameters,)
        Normalised stellar parameters [mass, Z, alpha_MLT, age].
    active_class : ActiveLearning
        An ActiveLearning instance providing the ML model, residual interpolators,
        and observational data.
    error_lists : dict
        Dictionary mapping error column names to lists; ML residual errors are
        appended in-place for each regression column.
    uncertainty_method : str, optional
        Method used to characterise ML uncertainty (default 'parameterisation').

    Returns
    -------
    float
        The log-likelihood value (sum over all regression outputs), or None if no
        regressor is attached to active_class.
    """
    mass, Z, alpha_MLT, age = theta

    likelihood = 0

    theta = np.reshape(theta, (1, len(active_class.ML_Data.parameter_columns)))
    # output_values = np.zeros((len(ml_data.regression_columns)))
    # ml_errors = np.zeros((len(ml_data.regression_columns)))
    
    if active_class.ML_Data.regressor is not None:
        for i in range(len(active_class.ML_Data.regression_columns)):
            output_value = active_class.ML_Data.regressor(theta)
            output_value = active_class.ML_Data.denormalise_regression_predictions(output_value)[0]
            ml_error = active_class.ML_Data.residual_interpolator[active_class.ML_Data.regression_columns[i]](np.reshape(active_class.ML_Data.denormalise_point(theta, parameters=True), (1, len(active_class.ML_Data.parameter_columns))))[0]
            error_lists[active_class.obs_err_cols[i]].append(ml_error[i])
            obs_val = float(active_class.observational_data[active_class.obs_data_cols[i]])
            obs_err = float(active_class.observational_data[active_class.obs_err_cols[i]])
            sigma2 = obs_err**2 + ml_error**2
            likelihood += ((output_value - obs_val)**2) / (sigma2) + np.log(2*np.pi*sigma2)
        return - likelihood / 2


def TRGB_SM_likelihood_function(theta, ml_outputs, observational_dataframe, observational_data_cols, observational_err_cols, scaling_factor=1):
    """
    Compute the scaled log-likelihood for a Tip of the Red Giant Branch (TRGB)
    single-mass model using the I-band absolute magnitude (M_I).

    Combines observational and ML uncertainties in quadrature and evaluates a
    single-term Gaussian log-likelihood.

    Parameters
    ----------
    theta : array-like of shape (3,)
        Stellar parameters [mass, Y, Z].
    ml_outputs : np.ndarray of shape (1, 2)
        ML-predicted [M_I value, M_I uncertainty].
    observational_dataframe : pd.DataFrame
        DataFrame containing 'M_I' and 'M_I_err' columns.
    observational_data_cols : list of str
        Column names for observed quantities (unused directly; kept for API consistency).
    observational_err_cols : list of str
        Column names for observed uncertainties (unused directly; kept for API consistency).
    scaling_factor : float, optional
        Multiplicative scale applied to the returned log-likelihood (default 1).

    Returns
    -------
    float
        The scaled log-likelihood value.
    """
    mass, Y, Z = theta
    likelihood = 0
    error_2 = (observational_dataframe['M_I_err'].to_numpy()**2 + ml_outputs[0,1]**2)
    likelihood += (ml_outputs[0, 0] - observational_dataframe['M_I'].to_numpy())**2 / error_2 + np.log(2*np.pi*error_2)
    return -likelihood / 2 * scaling_factor


def TRGB_DM_likelihood_function(theta, ml_outputs, observational_dataframe, observational_data_cols, observational_err_cols, scaling_factor=1):
    """
    Compute the scaled log-likelihood for a TRGB differential-mass model using
    the I-band absolute magnitude (M_I).

    Identical in form to TRGB_SM_likelihood_function but expects a four-parameter
    theta vector that includes an additional age parameter A.

    Parameters
    ----------
    theta : array-like of shape (4,)
        Stellar parameters [mass, Y, Z, A].
    ml_outputs : np.ndarray of shape (1, 2)
        ML-predicted [M_I value, M_I uncertainty].
    observational_dataframe : pd.DataFrame
        DataFrame containing 'M_I' and 'M_I_err' columns.
    observational_data_cols : list of str
        Column names for observed quantities (kept for API consistency).
    observational_err_cols : list of str
        Column names for observed uncertainties (kept for API consistency).
    scaling_factor : float, optional
        Multiplicative scale applied to the returned log-likelihood (default 1).

    Returns
    -------
    float
        The scaled log-likelihood value.
    """
    mass, Y, Z, A = theta
    likelihood = 0
    error_2 = (observational_dataframe['M_I_err'].to_numpy()**2 + ml_outputs[0,1]**2)
    likelihood += (ml_outputs[0, 0] - observational_dataframe['M_I'].to_numpy())**2 / error_2 + np.log(2*np.pi*error_2)
    return -likelihood / 2 * scaling_factor
