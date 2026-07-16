"""
This file defines functions to prepare the folded and ablated data, and save the fitted models to pickle files.

Author: Dorian Verdel [d.verdel@imperial.ac.uk]
Created: 05/2026
Last modified: 07/2026
"""


## Imports
# General
import os
import numpy as np
import pickle
import copy


def get_all_folded_ablated_data(data_calib, data_assist, params_prepro):
    """
    Function returning a complete dictionnary containing all the tested ablations and
    foldings of data, with combined CALIB and ASSIST datasets. Data are still stored as
    2D np.array at this stage.

    Args:
      - data_calib  : 493000x18 array; Full calibration experiment dataset
      - data_assist : 227322x18 array; Full assistance experiment dataset
    Output:
      - all_folded_ablated_data : dict; Dictionnary containing all the combinations of folding and ablations
    """
    ## Initialization
    ablationsList = params_prepro.get('ablations')
    outTypesList  = params_prepro.get('outTypes')
    foldsList     = params_prepro.get('folds')
    forceLoc      = params_prepro.get('forceLoc')
    savePath      = params_prepro.get('savePathPrepro')

    savePathFile = savePath + 'velocity_pos_2-fold.pkl'
    if os.path.isfile(savePathFile):
        print('Folded and ablated data already computed.')
    else:
        ## Get the folds common to all ablations
        print('Computing folded and ablated datasets...')
        dict_folds = {}
        for fold in foldsList:
            # Get folded data
            folded_data = get_folds(data_calib, data_assist, fold, subj_col = -1)      
            fold_name = fold + '-fold'
            # Update folds dictionnary
            dict_folds.update({fold_name: folded_data})

        ## Get ablated matrices with always the same folds
        # Loop over ablations of inputs
        all_folded_ablated_data = {}
        for ablation in ablationsList:
            # Loop over ablations of outputs
            inout_ablated_data = {}
            for out_type in outTypesList:
                # Loop over possible numbers of folds
                dict_foldsonly = {}
                for fold in foldsList:
                    folds_to_ablate_copy = {}
                    fold_name = fold + '-fold' 
                    folds_to_ablate = dict_folds.get(fold_name)
                    folds_to_ablate_copy = copy.deepcopy(folds_to_ablate)

                    # Loop over each fold
                    for i in range(0,int(fold)):
                        fold_id = 'fold' + str(i + 1)
                        fold_i_data = folds_to_ablate_copy.get(fold_id)
                        # Get training and evaluation data
                        train_fold_i = fold_i_data.get('train_data')
                        eval_fold_i  = fold_i_data.get('eval_data')
                        # Ablate training and evaluation data
                        ablated_train_fold_i = get_ablated_input(train_fold_i, data_to_remove = ablation, out_type = out_type, forceLoc = forceLoc)
                        ablated_eval_fold_i  = get_ablated_input(eval_fold_i, data_to_remove = ablation, out_type = out_type, forceLoc = forceLoc)
                        # Update dictionnary of 'k'-fold
                        fold_i_data.update({'train_data': ablated_train_fold_i, 'eval_data': ablated_eval_fold_i})
                        folds_to_ablate_copy.update({fold_id: fold_i_data})

                    # Save ablated dictionnary of 'k'-fold
                    dict_foldsonly.update({fold_name: folds_to_ablate_copy})

                # Save folds for each output type
                inout_ablated_data.update({out_type: dict_foldsonly})

            # Update final dictionnary
            all_folded_ablated_data.update({ablation: inout_ablated_data})

        ## Save folded data to pickle
        for ablation in ablationsList:
            ablated_outdata = all_folded_ablated_data.get(ablation)
            for out_type in outTypesList:
                ablated_inoutdata = ablated_outdata.get(out_type)
                for fold in foldsList:
                    fold_name = fold + '-fold'
                    ablatedFolded_data = ablated_inoutdata.get(fold_name)
                    savePathFile = savePath + ablation + '_' + out_type + '_' + fold_name + '.pkl'
                    with open(savePathFile, 'wb') as file:
                        pickle.dump(ablatedFolded_data, file)

        print('All folded and ablated data computed and saved to pickle')


def get_ablated_input(input_mat, data_to_remove = False, out_type = 'pos_vel', forceLoc = "wrist"):
    """
    Function returning simplified input matrix for ablation studies.

    Args:
      - input_mat      : len(batch)xIO array; Full input matrix [q3,q4,dq3,dq4,(FT_arm),FT_wrist,l_a,l_fa,output[4cols],mvt,subj]
      - data_to_remove : string             ; Type of data to remove for ablation
      - out_type       : string             ; Whether to predict position, velocity, or both
    Output:
      - ablated : 2D array; Ablated matrix
    """
    if out_type == "posvel":
        if data_to_remove == "None":
            # print("No ablation requested. Initial matrix is returned.")
            return input_mat
        elif data_to_remove == "all_but_pos":
            return np.concatenate([input_mat[:,0:2], input_mat[:,-6:]], axis = 1)
        elif data_to_remove == "all_but_posvel":
            return np.concatenate([input_mat[:,0:4], input_mat[:,-6:]], axis = 1)
        elif data_to_remove == "all_but_forcestorques":
            return np.concatenate([input_mat[:,4:10], input_mat[:,-6:]], axis = 1)
        elif data_to_remove == "all_but_forcestorquesant":
            return input_mat[:,4:]
        elif data_to_remove == "velocity":
            return np.concatenate([input_mat[:,0:2], input_mat[:,4:]], axis = 1)
        elif data_to_remove == "anthropo":
            if forceLoc == "both":
                return np.concatenate([input_mat[:,0:16], input_mat[:,18:]], axis = 1)
            else:
                return np.concatenate([input_mat[:,0:10], input_mat[:,12:]], axis = 1)
        elif data_to_remove == "forcestorques":
            if forceLoc == "both":
                return np.concatenate([input_mat[:,0:4], input_mat[:,16:]], axis = 1)
            else:
                return np.concatenate([input_mat[:,0:4], input_mat[:,10:]], axis = 1)
        elif data_to_remove == "forces":
            if forceLoc == "both":
                return np.concatenate([input_mat[:,0:4], input_mat[:,7:10], input_mat[:,13:]], axis = 1)
            else:
                return np.concatenate([input_mat[:,0:4], input_mat[:,7:]], axis = 1)
        elif data_to_remove == "torques":
            if forceLoc == "both":
                return np.concatenate([input_mat[:,0:7], input_mat[:,10:13], input_mat[:,16:]], axis = 1)
            else:
                return np.concatenate([input_mat[:,0:7], input_mat[:,10:]], axis = 1)
        elif data_to_remove == "Fx" and forceLoc == "wrist":
            return np.delete(input_mat, 4, axis = 1)
        elif data_to_remove == "Fy" and forceLoc == "wrist":
            return np.delete(input_mat, 5, axis = 1)
        elif data_to_remove == "Fz" and forceLoc == "wrist":
            return np.delete(input_mat, 6, axis = 1)
        elif data_to_remove == "Tx" and forceLoc == "wrist":
            return np.delete(input_mat, 7, axis = 1)
        elif data_to_remove == "Ty" and forceLoc == "wrist":
            return np.delete(input_mat, 8, axis = 1)
        elif data_to_remove == "Tz" and forceLoc == "wrist":
            return np.delete(input_mat, 9, axis = 1)
        else:
            print("Ablation case not handled. Initial matrix is returned.")
            return input_mat
    elif out_type == "pos":
        if data_to_remove == "None":
            # print("No ablation requested. Initial matrix without vel to predict is returned.")
            return np.concatenate([input_mat[:,0:-4], input_mat[:,-2:]], axis = 1)
        elif data_to_remove == "all_but_pos":
            return np.concatenate([input_mat[:,0:2], input_mat[:,-6:-4], input_mat[:,-2:]], axis = 1)
        elif data_to_remove == "all_but_posvel":
            return np.concatenate([input_mat[:,0:4], input_mat[:,-6:-4], input_mat[:,-2:]], axis = 1)
        elif data_to_remove == "all_but_forcestorques":
            return np.concatenate([input_mat[:,4:10], input_mat[:,-6:-4], input_mat[:,-2:]], axis = 1)
        elif data_to_remove == "all_but_forcestorquesant":
            return np.concatenate([input_mat[:,4:-4], input_mat[:,-2:]], axis = 1)
        elif data_to_remove == "velocity":
            return np.concatenate([input_mat[:,0:2], input_mat[:,4:-4], input_mat[:,-2:]], axis = 1)
        elif data_to_remove == "anthropo":
            return np.concatenate([input_mat[:,0:10], input_mat[:,-6:-4], input_mat[:,-2:]], axis = 1)
        elif data_to_remove == "forcestorques":
            return np.concatenate([input_mat[:,0:4], input_mat[:,-8:-4], input_mat[:,-2:]], axis = 1)
        elif data_to_remove == "forces":
            return np.concatenate([input_mat[:,0:4], input_mat[:,7:-4], input_mat[:,-2:]], axis = 1)
        elif data_to_remove == "torques":
            return np.concatenate([input_mat[:,0:7], input_mat[:,-8:-4], input_mat[:,-2:]], axis = 1)
        elif data_to_remove == "Fx" and forceLoc == "wrist":
            return np.delete(input_mat, [4, 14, 15], axis = 1)
        elif data_to_remove == "Fy" and forceLoc == "wrist":
            return np.delete(input_mat, [5, 14, 15], axis = 1)
        elif data_to_remove == "Fz" and forceLoc == "wrist":
            return np.delete(input_mat, [6, 14, 15], axis = 1)
        elif data_to_remove == "Tx" and forceLoc == "wrist":
            return np.delete(input_mat, [7, 14, 15], axis = 1)
        elif data_to_remove == "Ty" and forceLoc == "wrist":
            return np.delete(input_mat, [8, 14, 15], axis = 1)
        elif data_to_remove == "Tz" and forceLoc == "wrist":
            return np.delete(input_mat, [9, 14, 15], axis = 1)
        else:
            print("Ablation case not handled. Initial matrix is returned without vel to predict.")
            return np.concatenate([input_mat[:,0:-4], input_mat[:,-2:]], axis = 1)
    elif out_type == "vel":
        if data_to_remove == "None":
            # print("No ablation requested. Initial matrix without vel to predict is returned.")
            return np.concatenate([input_mat[:,0:-6], input_mat[:,-4:]], axis = 1)
        elif data_to_remove == "all_but_pos":
            return np.concatenate([input_mat[:,0:2], input_mat[:,-4:]], axis = 1)
        elif data_to_remove == "all_but_posvel":
            return np.concatenate([input_mat[:,0:4], input_mat[:,-4:]], axis = 1)
        elif data_to_remove == "all_but_forcestorques":
            return np.concatenate([input_mat[:,4:10], input_mat[:,-4:]], axis = 1)
        elif data_to_remove == "all_but_forcestorquesant":
            return np.concatenate([input_mat[:,4:-6], input_mat[:,-4:]], axis = 1)
        elif data_to_remove == "velocity":
            return np.concatenate([input_mat[:,0:2], input_mat[:,4:-6], input_mat[:,-4:]], axis = 1)
        elif data_to_remove == "anthropo":
            return np.concatenate([input_mat[:,0:10], input_mat[:,-4:]], axis = 1)
        elif data_to_remove == "forcestorques":
            return np.concatenate([input_mat[:,0:4], input_mat[:,-8:-6], input_mat[:,-4:]], axis = 1)
        elif data_to_remove == "forces":
            return np.concatenate([input_mat[:,0:4], input_mat[:,7:-6], input_mat[:,-4:]], axis = 1)
        elif data_to_remove == "torques":
            return np.concatenate([input_mat[:,0:7], input_mat[:,-8:-6], input_mat[:,-4:]], axis = 1)
        elif data_to_remove == "Fx" and forceLoc == "wrist":
            return np.delete(input_mat, [4, 12, 13], axis = 1)
        elif data_to_remove == "Fy" and forceLoc == "wrist":
            return np.delete(input_mat, [5, 12, 13], axis = 1)
        elif data_to_remove == "Fz" and forceLoc == "wrist":
            return np.delete(input_mat, [6, 12, 13], axis = 1)
        elif data_to_remove == "Tx" and forceLoc == "wrist":
            return np.delete(input_mat, [7, 12, 13], axis = 1)
        elif data_to_remove == "Ty" and forceLoc == "wrist":
            return np.delete(input_mat, [8, 12, 13], axis = 1)
        elif data_to_remove == "Tz" and forceLoc == "wrist":
            return np.delete(input_mat, [9, 12, 13], axis = 1)
        else:
            print("Ablation case not handled. Initial matrix is returned without pos to predict.")
            return np.concatenate([input_mat[:,0:-6], input_mat[:,-4:]], axis = 1)
    else:
        print("Ouput ablation case not handled. Initial matrix is returned.")
        return input_mat


def get_folds(calib_mat, assist_mat, nb_folds, subj_col = 0):
    """
    Function returning k-folds for training and evaluation of the
    exoskeleton-to-human mapping.

    Args:
      - calib_mat  : len(mat)xIO array   ; Ablated calibration dataset
      - assist_mat : len(mat)xIO array   ; Ablated assistance dataset
      - nb_folds   : string              ; Number of folds (applied at the subjects level)
      - subj_col   : int                 ; Index of the column containing the subjects' Ids
    Output:
      - folded_data : dict; dictionnary of k-folded data
    """
    ## Initialize lists of subjects
    subjListCalib  = np.unique(calib_mat[:,subj_col])
    subjListAssist = np.unique(assist_mat[:,subj_col])

    ## Get list of folds sizes for each dataset
    fSizes_calib = get_fold_sizes(subjListCalib, nb_folds)
    fSizes_assist = get_fold_sizes(subjListAssist, nb_folds, expe = 'ASSIST')

    ## Get and return dictionnary of folded data with training and evaluation sets
    folded_data = get_eval_and_train_folds(calib_mat, assist_mat, fSizes_calib, fSizes_assist, subjListCalib, subjListAssist, subj_col = subj_col)
    return folded_data


def get_fold_sizes(subjList, nb_folds, expe = 'CALIB'):
    """
    Function computing the folds sizes depending on whether the number of folds
    is a divider of the number of subjects or not.

    Args:
      - subjList : nbSubjsx1 array; Contains the list of subjects in a given dataset
      - nb_folds : string         ; Number of folds to extract
    Output:
      - sizesList : nb_foldsx1 array; List of sizes of folds
    """
    ## Handle Leave-One-Out (LOO) case
    if nb_folds == "LOO":
        sizesList = np.array([1]*len(subjList))
        return sizesList

    ## Initialization
    noDiv = False
    nb_subjs = len(subjList)
    nb_folds = int(nb_folds)

    ## Check exact division
    if nb_subjs % nb_folds != 0:
        print('Number of subjects in ' + expe + ' is not divisible by number of folds. Adjustments will be performed...')
        noDiv = True

    ## Build list of sizes
    if not noDiv:
        fold_size = int(nb_subjs / nb_folds)
        sizesList = np.array([fold_size]*nb_folds)
    else:
        # Get number of sujects to distribute in folds
        nb_add_subj = nb_subjs % nb_folds
        # Get basic size of folds
        fold_size_nominal = nb_subjs // nb_folds
        # Compute adjusted fold sizes
        sizesList = [fold_size_nominal] * nb_folds
        for i in range(0, nb_add_subj):
            sizesList[i] += 1
    
    ## Return list of sizes
    return sizesList


def get_eval_and_train_folds(calib_mat, assist_mat, fSizes_calib, fSizes_assist, subjListCalib, subjListAssist, subj_col = 0):
    """
    Function computing the evaluation and training folds for pre-computed fold sizes.

    Args:
      - calib_mat      : len(mat)xIO array; Ablated calibration dataset
      - assist_mat     : len(mat)xIO array; Ablated assistance dataset
      - fSizes_calib   : nb_foldsx1 array ; List of sizes of folds for the calibration dataset
      - fSizes_assist  : nb_foldsx1 array ; List of sizes of folds for the assistance dataset
      - subjListCalib  : 17x1 string array; List of subjects for the calibration experiment
      - subjListAssist : 17x1 string array; List of subjects for the assistance experiment
    Output:
      - folded_data : dictionnary; Dictionnary containning all folds for a given number of folds
    """
    ## Initialization
    nb_folds = len(fSizes_calib)
    subj_list_unused_calib  = subjListCalib.copy()
    subj_list_unused_assist = subjListAssist.copy()
    subj_list_used_calib    = []
    subj_list_used_assist   = []
    folded_data = {}
    # Random model
    rng = np.random.default_rng()

    ## Extract folds
    for i in range(0, nb_folds):
        # Get evaluation folds sizes
        ev_fold_size_calib  = fSizes_calib[i]
        ev_fold_size_assist = fSizes_assist[i]

        # Get random evaluation fold for calibration experiment
        eval_fold_calib      = rng.choice(subj_list_unused_calib, size = ev_fold_size_calib, replace = False)
        eval_fold_mask_calib = np.isin(calib_mat[:,subj_col], eval_fold_calib)
        eval_data_fold_calib = calib_mat[eval_fold_mask_calib]

        # Get random evaluation fold for assistance experiment
        eval_fold_assist      = rng.choice(subj_list_unused_assist, size = ev_fold_size_assist, replace = False)
        eval_fold_mask_assist = np.isin(assist_mat[:,subj_col], eval_fold_assist)
        eval_data_fold_assist = assist_mat[eval_fold_mask_assist]

        # Get full evaluation fold
        eval_fold_i = np.concatenate((eval_data_fold_calib, eval_data_fold_assist), axis = 0)

        # Remove already used subjects from calib assist lists
        shortlist_subj_calib  = list(set(subjListCalib)-set(eval_fold_calib))
        shortlist_subj_assist = list(set(subjListAssist)-set(eval_fold_assist))

        # Get corresponding training fold for calib
        training_fold_mask_calib = np.isin(calib_mat[:,subj_col], shortlist_subj_calib)
        training_data_fold_calib = calib_mat[training_fold_mask_calib]

        # Get corresponding training fold for assist
        training_fold_mask_assist = np.isin(assist_mat[:,subj_col], shortlist_subj_assist)
        training_data_fold_assist = assist_mat[training_fold_mask_assist]

        # Get full training fold
        train_fold_i = np.concatenate((training_data_fold_calib, training_data_fold_assist), axis = 0)

        # Build dict for the current fold
        fold_i = {"train_subjs_calib": shortlist_subj_calib, "eval_subjs_calib": eval_fold_calib,
                  "train_subjs_assist": shortlist_subj_assist, "eval_subjs_assist": eval_fold_assist,
                  "train_data": train_fold_i, "eval_data": eval_fold_i}
        fold_i_name = "fold" + str(i + 1)
        folded_data.update({fold_i_name: fold_i})

        # Save list of remaining subjects for evaluation for calib
        subj_list_used_calib.extend(eval_fold_calib)
        subj_list_unused_calib = list(set(subjListCalib)-set(subj_list_used_calib))

        # Save list of remaining subjects for evaluation for assist
        subj_list_used_assist.extend(eval_fold_assist)
        subj_list_unused_assist = list(set(subjListAssist)-set(subj_list_used_assist))

    ## Return folded data for a given number of folds
    return folded_data


def save_fitted_model(model):
    """
    Function saving a fitted model to pickle file.

    Args:
      - model: dict; Dictionnary containing the output of the model fitting
    """
    ## Extract simulation parameters
    saveDirFile = model.get('saveFittedDir')
    
    # Save fitted models
    with open(saveDirFile, 'wb') as file:
        pickle.dump(model, file)



