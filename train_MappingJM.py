## Imports
# General
import numpy as np
import torch
import torch.nn as nn
from neuralop.models import FNO
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

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
    if type == "MVPR":
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
    """
    ## Initialize
    models_MVLR = {}
    nb_folds = params.get('nb_folds')

    ## Loop over folds
    for i in range(1, nb_folds + 1):
        # Get fold i
        fold_i = "fold" + str(i)
        train_data_fold_i = folded_data.get(fold_i).get("train_data")
        # Get inputs and outputs
        input_data  = train_data_fold_i[:,:-6]
        output_data = train_data_fold_i[:,-6:-2]
        # Fit MVLR model
        modelFold_i = LinearRegression()
        modelFold_i.fit(input_data, output_data)
        # Save model
        models_MVLR.update({fold_i: modelFold_i})

    return models_MVLR

############################################### MVPR Functions

def train_MVPR_Mappings(folded_data, params):
    """
    Train multivariate polynomial regressions to estimate human joints states from robot data.

    Args:
      - folded_data : dict   ; Nested dictionnary with the different folds
      - params      : dict   ; Model and training parameters
    """
    ## Initialize
    models_MVPR = {}
    nb_folds    = params.get('nb_folds')
    degree      = params.get('degree')

    ## Loop over folds
    for i in range(1, nb_folds + 1):
        # Get fold i
        fold_i = "fold" + str(i)
        train_data_fold_i = folded_data.get(fold_i).get("train_data")
        # Get inputs and outputs
        input_data  = train_data_fold_i[:,:-6]
        output_data = train_data_fold_i[:,-6:-2]
        # Preprocess and fit MVPR model
        polyFeatures = PolynomialFeatures(degree = degree)
        input_polyData = polyFeatures.fit_transform(input_data)
        modelFold_i = LinearRegression()
        modelFold_i.fit(input_polyData, output_data)
        # Save model
        models_MVPR.update({fold_i: modelFold_i})

    return models_MVPR

############################################### FNO Functions

def train_FNO_Mappings(folded_data, params):
    """
    Train Fourier neural operators to estimate human joints states from robot data.

    Args:
      - folded_data : dict   ; Nested dictionnary with the different folds
      - params      : dict   ; Model and training parameters
    """
    ## Initialize
    models_FNO = {}
    nb_folds   = params.get('nb_folds')
    nb_modes   = params.get('nb_modes')
    nb_hChan   = params.get('nb_hiddenChannels')

    ## Loop over folds
    for i in range(1, nb_folds + 1):
        # Get fold i
        fold_i = "fold" + str(i)
        train_data_fold_i = folded_data.get(fold_i).get("train_data")
        # Get inputs and outputs
        X, Y, d_in, d_out = get_tensorsFromMat(train_data_fold_i)
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
        modelFold_i = train_FNO_oneFold(X, Y, modelFold_i, optimizer, loss_fn, params)
        # Save model
        models_FNO.update({fold_i: modelFold_i})

    return models_FNO

def get_tensorsFromMat(train_data_fold_i):
    """
    Transforms input 2D numpy arrays containing all training trials to torch tensors

    Args:
      - train_data_fold_i : dict ; Dictionnary of fold i
    Output:
      - X : torch tensor ; Input torch tensor
      - Y : torch tensor ; Output torch tensor
    """
    ## Initialize
    input_data = []
    output_data = []

    ## Get 3-dimensional np arrays
    for subject in train_data_fold_i[:,-1].unique():
        subj_data_fold_i = train_data_fold_i[train_data_fold_i[:,-1] == subject]
        for mvt in subj_data_fold_i[:,-2].unique():
            subj_mvt_data_fold_i = subj_data_fold_i[subj_data_fold_i[:,-2] == mvt]
            # Store movement data
            input_data.append(subj_mvt_data_fold_i[:,:-6])
            output_data.append(subj_mvt_data_fold_i[:,-6:-2])

    ## Get input and output dimmension
    d_in  = input_data.shape[-1]
    d_out = output_data.shape[-1]

    ## Transform input and output matrices to torch tensors
    X_np = torch.from_numpy(input_data).float()   # (mvt, samples_per_mvt, d_in)
    Y_np = torch.from_numpy(output_data).float()  # (mvt, samples_per_mvt, d_out)

    ## Permute for FNO with [batch = mvt, seq_len = samples_per_mvt]
    X = X_np.permute(0, 2, 1)  # (batch, d_in, seq_len)
    Y = Y_np.permute(0, 2, 1)  # (batch, d_out, seq_len)

    return X, Y, d_in, d_out

def train_FNO_oneFold(X, Y, modelFold_i, optimizer, loss_fn, params):
    """
    Run training loop of FNO model.

    Args:
      - X           : torch tensor ; Input torch tensor
      - Y           : torch tensor ; Output torch tensor
      - modelFold_i : FNO          ; Untrained FNO model
      - optimizer   : torch Adam   ; Adam optimizer from torch
      - loss_fn     : nn.MSELoss   ; Mean square error loss for training
      - params      : dict   ; Model and training parameters
    Output:
      - modelFold_i : FNO ; Trained FNO model
    """
    ## Initialize
    prev_loss = 1
    nb_maxIter = params.get('nb_maxTrainingIterations')
    limLoss    = params.get('limitConvergenceLoss')

    ## Run training
    for epoch in range(nb_maxIter):
        optimizer.zero_grad()
        Y_pred = modelFold_i(X)
        loss = loss_fn(Y_pred, Y)
        loss.backward()
        optimizer.step()
        if np.abs(loss.item()-prev_loss) < limLoss:
            print("Converged in")
            break
        prev_loss = loss.item()
    
    return modelFold_i

############################################### LSTM Functions

def train_LSTM_Mappings(folded_data, params):
    """
    Train long short term memory networks to estimate human joints states from robot data.

    Args:
      - folded_data : dict   ; Nested dictionnary with the different folds
      - params      : dict   ; Model and training parameters
    """

    models_LSTM = {}

    return models_LSTM