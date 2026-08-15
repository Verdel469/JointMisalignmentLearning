"""
This file defines functions to train the different types of models: MVLR, MVPR, FNO, LSTM.

Author: Dorian Verdel [d.verdel@imperial.ac.uk]
Created: 06/2026
Last modified: 07/2026
"""

## Imports
# General
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import time
from neuralop.models import FNO
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# Local
import mapLearningUtils as mlu

def run_training(sharedList, listParamsDict, iter):
    """
    Function running the training of the model for each given set of parameters.

    Args:
      - sharedList: Manager.list(); Shared list for parallel solution saving.
    """
    ## Get values from dict
    params_i    = listParamsDict[iter]
    model       = params_i.get('model')
    dataPath    = params_i.get('dataPath')
    ablation    = params_i.get('ablation')
    data_type   = params_i.get('dataType')
    out_type    = params_i.get('out_type')
    foldName    = params_i.get('fold')
    saveDirFile = params_i.get('saveFittedDir')

    if os.path.isfile(saveDirFile):
        print('Model already fitted and available at: ' + saveDirFile)
    else:
        ## Build params dictionnary for each model
        with open(dataPath, 'rb') as file:
            data = pickle.load(file)

        ## Inform current step
        print(model + ': ablated ' + ablation + '; data ' + data_type + '; predict ' + out_type + '; ' + foldName)

        ## Get trained models (k models for each type of fitting)
        fitted_models = train_MappingJM(data, model, params = params_i)
        params_i.update({'fitted_models': fitted_models})
        mlu.save_fitted_model(params_i)

    return sharedList.append(params_i)

def train_MappingJM(folded_data, type, params):
    """
    Function calling model training functions with given input parameters.

    Args:
      - folded_data : dict   ; Nested dictionnary with the different folds
      - type        : string ; Model type within "MVLR": MultiVar LinReg, "MVPR": MultiVar PolyReg, "FNO": Fourier Neural Operators, "LSTM": LSTM
      - params      : dict   ; Model and training parameters
    Output:
      - models : dict ; Dictionnary of trained robot-to-human mappings for the different folds
    """

    if type == "MVLR":
        models = train_MVLR_Mappings(folded_data, params)
    elif type == "MVPR":
        models = train_MVPR_Mappings(folded_data, params)
    elif type == "FNO":
        models = train_FNO_Mappings(folded_data, params)
    elif type == "LSTM":
        models = train_LSTM_Mappings(folded_data, params)
    else:
        models = None
    
    return models

############################################### MVLR Functions

def train_MVLR_Mappings(folded_data, params):
    """
    Train multivariate linear regressions to estimate human joints states from robot data.

    Args:
      - folded_data : dict   ; Nested dictionnary with the different folds
      - params      : dict   ; Model and training parameters
    Output:
      - models : dict ; Dictionnary of trained robot-to-human linear mappings for the different folds
    """
    ## Initialize
    models_MVLR = {}
    out_type    = params.get('out_type')
    data_type   = params.get('dataType')
    nb_folds    = params.get('nb_folds')

    ## Loop over folds
    for i in range(1, nb_folds + 1):
        # Get fold i
        fold_i = "fold" + str(i)
        train_data_fold_i = folded_data.get(fold_i).get("train_data")
        
        # Keep only relevant training data
        if data_type == 'CALIB':
            last_col = train_data_fold_i[:, -1].astype(str)
            mask_calib = np.char.find(last_col, '_c') != -1
            train_data_fold_i = train_data_fold_i[mask_calib, :]
        elif data_type == 'ASSIST':
            last_col = train_data_fold_i[:, -1].astype(str)
            mask_assist = np.char.find(last_col, '_a') != -1
            train_data_fold_i = train_data_fold_i[mask_assist, :]
        
        # Get inputs and outputs
        if out_type == 'posvel':
            input_data  = np.array(train_data_fold_i[:,:-6], dtype = np.float64)
            output_data = np.array(train_data_fold_i[:,-6:-2], dtype = np.float64)
        else:
            input_data  = np.array(train_data_fold_i[:,:-4], dtype = np.float64)
            output_data = np.array(train_data_fold_i[:,-4:-2], dtype = np.float64)
        # Time before preprocessing and training
        timebef = time.time()
        # Fit MVLR model
        modelFold_i = LinearRegression()
        modelFold_i.fit(input_data, output_data)
        # Get elapsed time during model creation and fitting
        elapsed = time.time() - timebef
        time_fold_i = 'fitTime' + fold_i
        # Save model
        models_MVLR.update({fold_i: modelFold_i, time_fold_i: elapsed})

    return models_MVLR

############################################### MVPR Functions

def train_MVPR_Mappings(folded_data, params):
    """
    Train multivariate polynomial regressions to estimate human joints states from robot data.

    Args:
      - folded_data : dict   ; Nested dictionnary with the different folds
      - params      : dict   ; Model and training parameters
    Output:
      - models : dict ; Dictionnary of trained robot-to-human polynomial mappings for the different folds
    """
    ## Initialize
    models_MVPR = {}
    out_type    = params.get('out_type')
    data_type   = params.get('dataType')
    nb_folds    = params.get('nb_folds')
    degree      = params.get('degree')

    ## Loop over folds
    for i in range(1, nb_folds + 1):
        # Get fold i
        fold_i = "fold" + str(i)
        train_data_fold_i = folded_data.get(fold_i).get("train_data")

        # Keep only relevant training data
        if data_type == 'CALIB':
            last_col = train_data_fold_i[:, -1].astype(str)
            mask_calib = np.char.find(last_col, '_c') != -1
            train_data_fold_i = train_data_fold_i[mask_calib, :]
        elif data_type == 'ASSIST':
            last_col = train_data_fold_i[:, -1].astype(str)
            mask_assist = np.char.find(last_col, '_a') != -1
            train_data_fold_i = train_data_fold_i[mask_assist, :]

        # Get inputs and outputs
        if out_type == 'posvel':
            input_data  = np.array(train_data_fold_i[:,:-6], dtype = np.float64)
            output_data = np.array(train_data_fold_i[:,-6:-2], dtype = np.float64)
        else:
            input_data  = np.array(train_data_fold_i[:,:-4], dtype = np.float64)
            output_data = np.array(train_data_fold_i[:,-4:-2], dtype = np.float64)
        # Time before preprocessing and training
        timebef = time.time()
        # Preprocess and fit MVPR model
        polyFeatures = PolynomialFeatures(degree = degree)
        input_polyData = polyFeatures.fit_transform(input_data)
        modelFold_i = LinearRegression()
        modelFold_i.fit(input_polyData, output_data)
        # Get elapsed time during model creation and fitting
        elapsed = time.time() - timebef
        time_fold_i = 'fitTime' + fold_i
        # Save model
        models_MVPR.update({fold_i: modelFold_i, time_fold_i: elapsed})

    return models_MVPR

############################################### FNO Functions

def train_FNO_Mappings(folded_data, params):
    """
    Train Fourier neural operators to estimate human joints states from robot data.

    Args:
      - folded_data : dict; Nested dictionnary with the different folds
      - params      : dict; Model and training parameters
    Output:
      - models : dict; Dictionnary of trained robot-to-human FNO mappings for the different folds
    """
    ## Initialize
    models_FNO = {}
    nb_folds   = params.get('nb_folds')
    data_type   = params.get('dataType')
    nb_modes   = params.get('nbModes')
    nb_hChan   = params.get('nbHC')

    ## Loop over folds
    for i in range(1, nb_folds + 1):
        # Get fold i
        fold_i = "fold" + str(i)
        train_data_fold_i = folded_data.get(fold_i).get("train_data")

        # Keep only relevant training data
        if data_type == 'CALIB':
            last_col = train_data_fold_i[:, -1].astype(str)
            mask_calib = np.char.find(last_col, '_c') != -1
            train_data_fold_i = train_data_fold_i[mask_calib, :]
        elif data_type == 'ASSIST':
            last_col = train_data_fold_i[:, -1].astype(str)
            mask_assist = np.char.find(last_col, '_a') != -1
            train_data_fold_i = train_data_fold_i[mask_assist, :]

        # Time before preprocessing and training
        timebef = time.time()
        # Get inputs and outputs
        X, Y, d_in, d_out = get_tensorsFromMat(train_data_fold_i, params)
        # Format data
        # Create model
        modelFold_i = FNO(
            n_modes         = (nb_modes,),
            hidden_channels = nb_hChan,
            in_channels     = d_in,
            out_channels    = d_out
        )
        optimizer = torch.optim.Adam(modelFold_i.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        # Run training loop
        nbIt_i = 'nbItConv' + fold_i
        loss_i = 'losses' + fold_i
        modelFold_i, losses_i, nb_it_i = train_FNO_oneFold(X, Y, modelFold_i, optimizer, loss_fn, params)
        # Get elapsed time during model creation and fitting
        elapsed = time.time() - timebef
        time_fold_i = 'fitTime' + fold_i
        # Save model
        models_FNO.update({fold_i: modelFold_i, loss_i: losses_i, time_fold_i: elapsed, nbIt_i: nb_it_i})

    return models_FNO

def get_tensorsFromMat(train_data_fold_i, params):
    """
    Transforms input 2D numpy arrays containing all training trials to torch tensors

    Args:
      - train_data_fold_i : dict; Dictionnary of fold i
      - params            : dict; Model and training parameters
    Output:
      - X     : torch tensor; Input torch tensor
      - Y     : torch tensor; Output torch tensor
      - d_in  : int         ; Number of input features
      - d_out : int         ; Number of output features
    """
    ## Initialize
    out_type  = params.get('out_type')
    shuffle   = params.get('shuffle')
    batchSize = params.get('batchSize')

    ## Get shuffled or not shuffled input and output data
    if shuffle:
        shuffled_data = np.random.permutation(train_data_fold_i)
    else:
        shuffled_data = train_data_fold_i

    if out_type == 'posvel':
        input_data      = shuffled_data[:,:-6]
        input_data_num  = input_data.astype(np.float64)
        output_data     = shuffled_data[:,-6:-2]
        output_data_num = output_data.astype(np.float64)
    else:
        input_data      = shuffled_data[:,:-4]
        input_data_num  = input_data.astype(np.float64)
        output_data     = shuffled_data[:,-4:-2]
        output_data_num = output_data.astype(np.float64)

    ## Discard samples to match batch size
    nbBatch   = input_data_num.shape[0] // batchSize
    nbSamples = nbBatch * batchSize
    input_data_cut  = input_data_num[:nbSamples,:]
    output_data_cut = output_data_num[:nbSamples,:]

    ## Get input and output dimmension
    d_in  = input_data_cut.shape[1]
    d_out = output_data_cut.shape[1]

    ## Reshape to 3d arrays
    split_input  = np.array_split(input_data_cut, nbBatch, axis = 0)
    input_3d     = np.stack(split_input)
    split_output = np.array_split(output_data_cut, nbBatch, axis = 0)
    output_3d    = np.stack(split_output)

    ## Transform input and output matrices to torch tensors
    X_np = torch.from_numpy(input_3d).float()   # (batch, batchSize, d_in)
    Y_np = torch.from_numpy(output_3d).float()  # (batch, batchSize, d_out)

    ## Permute for FNO with [seq_len = batchSize]
    X = X_np.permute(0, 2, 1)  # (batch, d_in, seq_len)
    Y = Y_np.permute(0, 2, 1)  # (batch, d_out, seq_len)

    return X, Y, d_in, d_out

def train_FNO_oneFold(X, Y, modelFold_i, optimizer, loss_fn, params):
    """
    Run training loop of FNO model.

    Args:
      - X           : torch tensor; Input torch tensor
      - Y           : torch tensor; Output torch tensor
      - modelFold_i : FNO         ; Untrained FNO model
      - optimizer   : torch Adam  ; Adam optimizer from torch
      - loss_fn     : nn.MSELoss  ; Mean square error loss for training
      - params      : dict        ; Model and training parameters
    Output:
      - modelFold_i : FNO; Trained FNO model
      - epoch       : int; Number of iterations for convergence 
    """
    ## Initialize
    prev_loss = 1
    max_It    = params.get('maxIt')
    min_loss  = params.get('minLoss')
    losses    = []

    ## Run training
    for epoch in range(max_It):
        optimizer.zero_grad()
        Y_pred = modelFold_i(X)
        loss = loss_fn(Y_pred, Y)
        loss.backward()
        optimizer.step()
        if np.abs(loss.item()-prev_loss) < min_loss:
            print("No change at epoch: " + str(epoch))
        prev_loss = loss.item()
        losses.append(prev_loss)
        
        # Print current state
        print(f"Data {params.get('dataType')}; out {params.get('out_type')}; ablated {params.get('ablation')}; {params.get('fold')}; batch {params.get('batchSize')}; shuffle {params.get('shuffle')}; Epoch {epoch}; Loss = {loss.item():.6f}")
    
    return modelFold_i, losses, epoch

############################################### LSTM Functions

def train_LSTM_Mappings(folded_data, params):
    """
    Train long short term memory networks to estimate human joints states from robot data.

    Args:
      - folded_data : dict   ; Nested dictionnary with the different folds
      - params      : dict   ; Model and training parameters
    Output:
      - models : dict ; Dictionnary of trained robot-to-human LSTM mappings for the different folds
    """

    models_LSTM = {}

    return models_LSTM