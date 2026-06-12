"""
This file defines functions to evaluate the fitted models.

Author: Dorian Verdel [d.verdel@imperial.ac.uk]
Last modified: 06/2026
"""

## Imports
# General
import os
import numpy as np
import pickle
from neuralop.models import FNO
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def compute_eval_params(model, eval_data, model_params, nb_joints = 2):
    """
    Function computing RMS and absolute mean errors for a given model for each joint.

    Args:
      - model        : fittedModel; Coefficients of the fitted model
      - eval_data    : NxM array  ; Array containing one ablated fold of evaluation data
      - model_params : dict       ; Dictionnary containg all relevant information
    """
    ## Initialize
    list_dictErrors = []
    model_name      = model_params.get('model')
    ablation        = model_params.get('ablation')
    fold_name       = model_params.get('fold')
    fold_id         = model_params.get('fold_id')

    ## Get degree of polynomial fitting
    if model_name == "MVPR":
        degree       = model_params.get('degree')
        polyFeatures = PolynomialFeatures(degree = degree)
    else:
        degree = 0

    ## Extract information from input data matrix
    subjList    = np.unique(eval_data[:,-1])

    ## Loop over subjects
    for subject in subjList:
        # Get subject data
        eval_data_subj   = eval_data[eval_data[:,-1] == subject]
        input_data_subj  = eval_data_subj[:,:-6]
        output_data_subj = eval_data_subj[:,-6:-2]

        if model_name == "MVPR":
            input_data_subj = polyFeatures.fit_transform(input_data_subj)

        pred_data_subj = model.predict(input_data_subj)
        # Position RMS
        qs_rms = compute_RMSerror(pred_data_subj[:,0], output_data_subj[:,0])
        qe_rms = compute_RMSerror(pred_data_subj[:,1], output_data_subj[:,1])
        # Velocity RMS
        dqs_rms = compute_RMSerror(pred_data_subj[:,2], output_data_subj[:,2])
        dqe_rms = compute_RMSerror(pred_data_subj[:,3], output_data_subj[:,3])
        # Position AAE
        qs_aae = compute_AAE(pred_data_subj[:,0], output_data_subj[:,0])
        qe_aae = compute_AAE(pred_data_subj[:,1], output_data_subj[:,1])
        # Velocity AAE
        dqs_aae = compute_AAE(pred_data_subj[:,2], output_data_subj[:,2])
        dqe_aae = compute_AAE(pred_data_subj[:,3], output_data_subj[:,3])
        # Position MAX absolute error
        qs_max = compute_MaxAbsError(pred_data_subj[:,0], output_data_subj[:,0])
        qe_max = compute_MaxAbsError(pred_data_subj[:,1], output_data_subj[:,1])
        # Velocity MAX error
        dqs_max = compute_MaxAbsError(pred_data_subj[:,2], output_data_subj[:,2])
        dqe_max = compute_MaxAbsError(pred_data_subj[:,3], output_data_subj[:,3])
        # Position STD error
        qs_std = compute_STD_error(pred_data_subj[:,0], output_data_subj[:,0])
        qe_std = compute_STD_error(pred_data_subj[:,1], output_data_subj[:,1])
        # Velocity STD error
        dqs_std = compute_STD_error(pred_data_subj[:,2], output_data_subj[:,2])
        dqe_std = compute_STD_error(pred_data_subj[:,3], output_data_subj[:,3])

        # Compute torques
        tau_s_pred, tau_e_pred, tau_s, tau_e = compute_Torques(input_data_subj, output_data_subj)

        # Torque RMS
        ts_rms = compute_RMSerror(tau_s_pred, tau_s)
        te_rms = compute_RMSerror(tau_e_pred, tau_e)
        # Torque AAE
        ts_aae = compute_AAE(tau_s_pred, tau_s)
        te_aae = compute_AAE(tau_e_pred, tau_e)
        # Torque MAX error
        ts_max = compute_MaxAbsError(tau_s_pred, tau_s)
        te_max = compute_MaxAbsError(tau_e_pred, tau_e)
        # Torque STD error
        ts_std = compute_STD_error(tau_s_pred, tau_s)
        te_std = compute_STD_error(tau_e_pred, tau_e)

        # Store computed errors
        dict_errors = {'model'    : [model_name]*nb_joints, 'ablation' : [ablation]*nb_joints,
                       'fold'     : [fold_name]*nb_joints , 'fold_id'  : [fold_id]*nb_joints ,
                       'subject'  : [subject]*nb_joints   , 'joint'    : ['shoulder', 'elbow'],
                       'posRMSE'  : [qs_rms, qe_rms]      , 'velRMSE'  : [dqs_rms, dqe_rms],
                       'posAAE'   : [qs_aae, qe_aae]      , 'velAAE'   : [dqs_aae, dqe_aae],
                       'tauRMSE'  : [ts_rms, te_rms]      , 'tauAAE'   : [ts_aae, te_aae],
                       'pErrMax'  : [qs_max, qe_max]      , 'vErrMax'  : [dqs_max, dqe_max],
                       'pErrStd'  : [qs_std, qe_std]      , 'vErrStd'  : [dqs_std, dqe_std],
                       'tauErrMax': [ts_max, te_max]      , 'tauErrStd': [ts_std, te_std]}
        
        list_dictErrors.append(dict_errors)
    
    ## Return list of dicts of errors
    return list_dictErrors

def compute_RMSerror(pred, data):
    """
    Compute the RMS error between two column vectors.

    Args:
      - pred : nSamplesx1; Column vector of model predictions
      - data : nSamplesx1; Column vector of recorded data
    """
    squared_err = (pred - data)**2
    return np.sqrt(np.mean(squared_err))


def compute_AAE(pred, data):
    """
    Compute the Average Absolute Error between two column vectors.

    Args:
      - pred : nSamplesx1; Column vector of model predictions
      - data : nSamplesx1; Column vector of recorded data
    """
    abs_err = np.abs(pred - data)
    return np.mean(abs_err)


def compute_MaxAbsError(pred, data):
    """
    Compute the maximum absolute error between two column vectors.

    Args:
      - pred : nSamplesx1; Column vector of model predictions
      - data : nSamplesx1; Column vector of recorded data
    """
    abs_err = np.abs(pred - data)
    return np.max(abs_err)


def compute_STD_error(pred, data):
    """
    Compute the standard deviation of the error between two column vectors.

    Args:
      - pred : nSamplesx1; Column vector of model predictions
      - data : nSamplesx1; Column vector of recorded data
    """
    err = pred - data
    return np.std(err)


def compute_Torques(pred, data, ablation):
    """
    Compute the RMS error between two column vectors.

    Args:
      - pred : nSamplesx1; Column vector of model predictions
      - data : nSamplesx1; Column vector of recorded data
    """
    # TODO