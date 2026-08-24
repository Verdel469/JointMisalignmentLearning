"""
This file defines functions to evaluate the fitted models.

Author: Dorian Verdel [d.verdel@imperial.ac.uk]
Created: 06/2026
Last modified: 07/2026
"""

## Imports
# General
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from neuralop.models import FNO
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
# Local
import motor_control_tools.signal as mct_sig
import inverse_dynamics as invDyn

def get_all_eval_params_dict(all_params, all_anthropo, saveDF_path = './all_params/'):
    """
    Function to compute evaluation metrics for all data folds and all model types.

    Args:
      - all_params   : list of dict; List of dictionnary containing all relevant data for the evaluation
      - all_anthropo : dict        ; Dictionnary with anthropometrics for each subject
      - saveDF_path  : string      ; Folder to save the full dataframe of metrics
    """
    ## Initialize
    all_metrics = []

    ## Loop over models
    for dict_model in all_params:
        # Get fitted models
        pathModel = dict_model.get('saveFittedDir')
        print('Working on model file: ' + pathModel)
        try:
            with open(pathModel, 'rb') as file:
                models = pickle.load(file)
            models_all_folds = models.get('fitted_models')

            # Get folded ablated data
            pathData = dict_model.get('dataPath')
            with open(pathData, 'rb') as file:
                all_data = pickle.load(file)
            
            # Get path to model evaluation saving
            path_eval = dict_model.get('savePath_eval')

            if os.path.isfile(path_eval):
                print('Model already evaluated, results available at: ' + path_eval)
            else:
                # Loop over folds
                nb_folds = dict_model.get('nb_folds')
                all_metrics = []
                for i in range(1, nb_folds + 1):
                    # Get model and fit time
                    fold_i  = 'fold' + str(i)
                    time_fold_i = 'fitTime' + fold_i
                    model_i = models_all_folds.get(fold_i)
                    time_i  = models_all_folds.get(time_fold_i)
                    dict_model.update({'fold_id': fold_i})

                    # Get training and evaluation data
                    train_data = all_data.get(fold_i).get('train_data')
                    eval_data  = all_data.get(fold_i).get('eval_data')

                    train_metrics = compute_eval_params(model_i, time_i, train_data, dict_model, all_anthropo, nb_joints = 2, type = 'train')
                    eval_metrics  = compute_eval_params(model_i, time_i, eval_data, dict_model, all_anthropo, nb_joints = 2, type = 'eval')

                    all_metrics.append(train_metrics)
                    all_metrics.append(eval_metrics)

                ## Get and save DataFrame of metrics
                all_metrics_df = pd.concat(all_metrics, axis = 0)
                all_metrics_df.to_csv(path_eval, index = False)
        except:
            print("Model not fitted: " + pathModel)

    return 1


def compute_eval_params(model, fitTime, eval_data, model_params, all_anthropo, nb_joints = 2, type = 'eval'):
    """
    Function computing RMS and absolute mean errors for a given model for each joint.

    Args:
      - model        : fittedModel; Coefficients of the fitted model
      - fitTime      : float      ; Elapsed time during model fitting [s]
      - eval_data    : NxM array  ; Array containing one ablated fold of evaluation data
      - model_params : dict       ; Dictionnary containg all relevant information regarding the fitted models
      - all_anthropo : dict       ; Dictionnary containg all the participants' anthropometric information
      - type         : string     ; Whether to evaluate performance on training or evaluation data
    Output:
      - list_dfErrors : list of DataFrames; Contains a list of computed errors stored in dataframes
    """
    ## Initialize
    list_dfErrors = []
    model_name      = model_params.get('model')
    ablation        = model_params.get('ablation')
    data_type       = model_params.get('dataType')
    out_type        = model_params.get('out_type')
    fold_name       = model_params.get('fold')
    fold_id         = model_params.get('fold_id')

    ## Get degree of polynomial fitting
    if model_name == "MVPR":
        degree       = model_params.get('degree')
        polyFeatures = PolynomialFeatures(degree = degree)
    else:
        degree = None

    ## Get FNO parameters
    if model_name == "FNO":
        batchSize = model_params.get('batchSize')
        nbModes   = model_params.get('nbModes')
        nbHC      = model_params.get('nbHC')
        maxIt     = model_params.get('maxIt')
        shuffled  = str(model_params.get('shuffle'))
        # minLoss   = model_params.get('minLoss')
    else:
        batchSize = None
        nbModes   = None
        nbHC      = None
        maxIt     = None
        shuffled  = None
        # minLoss   = None

    ## Extract information from input data matrix
    subjList = np.unique(eval_data[:,-1])

    ## Loop over subjects
    for subject in subjList:
        # Get subject recorded data
        eval_data_subj = eval_data[eval_data[:,-1] == subject]
        if out_type == 'posvel':
            input_data_subj  = eval_data_subj[:,:-6].astype(float)
            output_data_subj = eval_data_subj[:,-6:-2].astype(float)
        else:
            input_data_subj  = eval_data_subj[:,:-4].astype(float)
            output_data_subj = eval_data_subj[:,-4:-2].astype(float)
            
        
        anthropo_data_subj = all_anthropo.get(subject)

        if model_name == "MVPR":
            input_data_subj = polyFeatures.fit_transform(input_data_subj)

        if model_name == "FNO":
            input_data_subj_torch = torch.from_numpy(input_data_subj).float()
            input_data_subj_3D    = input_data_subj_torch.unsqueeze(0)
            input_data_subj_FNO   = input_data_subj_3D.permute(0, 2, 1)
            pred_data_subj_FNO    = model(input_data_subj_FNO)
            pred_data_subj_3D     = pred_data_subj_FNO.permute(0, 2, 1)
            pred_data_subj_2D     = pred_data_subj_3D.squeeze(0)
            pred_data_subj        = pred_data_subj_2D.detach().numpy()
        else:
            pred_data_subj = model.predict(input_data_subj)
        
        # Get velocities by numerical differentiation if pos is the only predicted
        if out_type == 'pos':
            # Motion capture baseline
            j_pos_filt       = mct_sig.filter(output_data_subj, 100, low_pass = 5, order = 5)
            j_vel            = mct_sig.diff_keep_length(j_pos_filt, 100)
            output_data_subj = np.concatenate((output_data_subj, j_vel), axis = 1)
            # Prediction
            j_pred_filt    = mct_sig.filter(pred_data_subj, 100, low_pass = 5, order = 5)
            j_vel_pred     = mct_sig.diff_keep_length(j_pred_filt, 100)
            pred_data_subj = np.concatenate((pred_data_subj, j_vel_pred), axis = 1)
            
        if out_type == 'pos' or out_type == 'posvel':
            # Position RMS
            qs_rms = compute_RMSerror(pred_data_subj[:,0], output_data_subj[:,0])
            qe_rms = compute_RMSerror(pred_data_subj[:,1], output_data_subj[:,1])
            # Position AAE
            qs_aae = compute_AAE(pred_data_subj[:,0], output_data_subj[:,0])
            qe_aae = compute_AAE(pred_data_subj[:,1], output_data_subj[:,1])
            # Position MAX absolute error
            qs_max = compute_MaxAbsError(pred_data_subj[:,0], output_data_subj[:,0])
            qe_max = compute_MaxAbsError(pred_data_subj[:,1], output_data_subj[:,1])
            # Position STD error
            qs_std = compute_STD_error(pred_data_subj[:,0], output_data_subj[:,0])
            qe_std = compute_STD_error(pred_data_subj[:,1], output_data_subj[:,1])

            # Velocity RMS
            dqs_rms = compute_RMSerror(pred_data_subj[:,2], output_data_subj[:,2])
            dqe_rms = compute_RMSerror(pred_data_subj[:,3], output_data_subj[:,3])
            # Velocity AAE
            dqs_aae = compute_AAE(pred_data_subj[:,2], output_data_subj[:,2])
            dqe_aae = compute_AAE(pred_data_subj[:,3], output_data_subj[:,3])
            # Velocity MAX error
            dqs_max = compute_MaxAbsError(pred_data_subj[:,2], output_data_subj[:,2])
            dqe_max = compute_MaxAbsError(pred_data_subj[:,3], output_data_subj[:,3])
            # Velocity STD error
            dqs_std = compute_STD_error(pred_data_subj[:,2], output_data_subj[:,2])
            dqe_std = compute_STD_error(pred_data_subj[:,3], output_data_subj[:,3])

            # Compute torques
            tau_s_pred, tau_e_pred, tau_s, tau_e = compute_Torques(pred_data_subj, output_data_subj, anthropo_data_subj)
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
        
        else:
            # Position errors to None
            qs_rms, qe_rms, qs_aae, qe_aae, qs_max, qe_max, qs_std, qe_std = None, None, None, None, None, None, None, None

            # Velocity RMS
            dqs_rms = compute_RMSerror(pred_data_subj[:,0], output_data_subj[:,0])
            dqe_rms = compute_RMSerror(pred_data_subj[:,1], output_data_subj[:,1])
            # Velocity AAE
            dqs_aae = compute_AAE(pred_data_subj[:,0], output_data_subj[:,0])
            dqe_aae = compute_AAE(pred_data_subj[:,1], output_data_subj[:,1])
            # Velocity MAX error
            dqs_max = compute_MaxAbsError(pred_data_subj[:,0], output_data_subj[:,0])
            dqe_max = compute_MaxAbsError(pred_data_subj[:,1], output_data_subj[:,1])
            # Velocity STD error
            dqs_std = compute_STD_error(pred_data_subj[:,0], output_data_subj[:,0])
            dqe_std = compute_STD_error(pred_data_subj[:,1], output_data_subj[:,1])

            # Torque errors to None
            ts_rms, te_rms, ts_aae, te_aae, ts_max, te_max, ts_std, te_std = None, None, None, None, None, None, None, None
        

        # Store computed errors
        dict_errors = {'model'    : [model_name]*nb_joints, 'ablation' : [ablation]*nb_joints,
                       'predicted': [out_type]*nb_joints  , 'fold'     : [fold_name]*nb_joints,
                       'fold_id'  : [fold_id]*nb_joints   , 'type'     : [type]*nb_joints,
                       'dataType' : [data_type]*nb_joints , 'subject'  : [subject]*nb_joints,
                       'joint'    : ['shoulder', 'elbow'] , 'polDegree': [degree]*nb_joints,
                       'FNO_shuff': [shuffled]*nb_joints  , 'FNO_batch': [batchSize]*nb_joints,
                       'FNO_nbMod': [nbModes]*nb_joints   , 'FNO_nbHC' : [nbHC]*nb_joints,
                       'FNO_maxIt': [maxIt]*nb_joints     , 'fitTime'  : [fitTime]*nb_joints,
                       'posRMSE'  : [qs_rms, qe_rms]      , 'velRMSE'  : [dqs_rms, dqe_rms],
                       'posAAE'   : [qs_aae, qe_aae]      , 'velAAE'   : [dqs_aae, dqe_aae],
                       'tauRMSE'  : [ts_rms, te_rms]      , 'tauAAE'   : [ts_aae, te_aae],
                       'pErrMax'  : [qs_max, qe_max]      , 'vErrMax'  : [dqs_max, dqe_max],
                       'pErrStd'  : [qs_std, qe_std]      , 'vErrStd'  : [dqs_std, dqe_std],
                       'tauErrMax': [ts_max, te_max]      , 'tauErrStd': [ts_std, te_std],
                       }
        
        df_errors = pd.DataFrame(dict_errors)
        
        list_dfErrors.append(df_errors)
    
    ## Return list of dicts of errors
    all_subjs_dfErrors = pd.concat(list_dfErrors, axis = 0)
    return all_subjs_dfErrors


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


def compute_Torques(pred, data, anthropo_subj):
    """
    Compute torques corresponding to predicted and recorded human joints trajectories.

    Args:
      - pred          : nSamplesx4; Column vector of model predictions
      - data          : nSamplesx4; Column vector of recorded data
      - anthropo_subj : dict      ; Dictionnary of anthropometric parameters for the considered subject
    Output:
      - tau_s_pred : nSamplesx1; Vector of predicted shoulder torques
      - tau_e_pred : nSamplesx1; Vector of predicted elbow torques
      - tau_s      : nSamplesx1; Vector of estimated shoulder torques
      - tau_e      : nSamplesx1; Vector of estimated elbow torques
    """
    ## Initialize
    j_pos_est  = data[:,:2]
    j_vel_est  = data[:,2:]
    j_pos_pred = pred[:,:2]
    j_vel_pred = pred[:,2:]

    ## Get torques
    tau_s, tau_e           = invDyn.inverse_dynamics(anthropo_subj, j_pos_est, j_vel = j_vel_est, sRate = 100)
    tau_s_pred, tau_e_pred = invDyn.inverse_dynamics(anthropo_subj, j_pos_pred, j_vel = j_vel_pred, sRate = 100)

    ## Return torques
    return tau_s_pred, tau_e_pred, tau_s, tau_e