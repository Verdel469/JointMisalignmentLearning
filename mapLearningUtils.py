## Imports
# General
import numpy as np


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
    foldsList     = params_prepro.get('folds')
    forceLoc      = params_prepro.get('forceLoc')

    ## Loop over ablations and folds
    all_folded_ablated_data = {}
    for ablation in ablationsList:
        # Get ablated matrices
        ablated_calib  = get_ablated_input(data_calib, data_to_remove = ablation, forceLoc = forceLoc)
        ablated_assist = get_ablated_input(data_assist, data_to_remove = ablation, forceLoc = forceLoc)
        
        dict_folds = {}
        for fold in foldsList:
            # Get ablated and folded data
            folded_data = get_folds(ablated_calib, ablated_assist, fold, subj_col = -1)
            # Update folds dictionnary
            dict_folds.update({fold: folded_data})
        
        # Update final dictionnary
        all_folded_ablated_data.update({ablation: dict_folds})

    ## Return complete dictionnary
    return all_folded_ablated_data



def get_ablated_input(input_mat, data_to_remove = False, forceLoc = "wrist"):
    """
    Function returning simplified input matrix for ablation studies.

    Args:
      - input_mat      : len(batch)xIO array; Full input matrix [q3,q4,dq3,dq4,(FT_arm),FT_wrist,l_a,l_fa,mvt,subj,output]
      - data_to_remove : string             ; Type of data to remove for ablation
    """
    if not data_to_remove:
        print("No ablation requested. Initial matrix is returned.")
        return input_mat
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


def get_folds(calib_mat, assist_mat, nb_folds, subj_col = 0):
    """
    Function returning k-folds for training and evaluation of the
    exoskeleton-to-human mapping.

    Args:
      - calib_mat  : len(mat)xIO array   ; Ablated calibration dataset
      - assist_mat : len(mat)xIO array   ; Ablated assistance
      - nb_folds   : string              ; Number of folds (applied at the subjects level)
      - subj_col   : int                 ; Index of the column containing the subjects' Ids
    Output:
      - folded_data : dict; dictionnary of k-folded data
    """
    ## Initialization
    subj_list = np.unique(input_mat[:,subj_col])
    if len(subj_list) % nb_folds != 0:
        print("Number of subjects is not divisible by number of folds.")
        return False
    
    if nb_folds == "LOO":
        fold_size = 1
    else:
        fold_size = int(len(subj_list) / nb_folds)

    # Random model
    rng = np.random.default_rng()

    # Create local variables
    subj_list_unused = subj_list.copy()
    subj_list_used = []
    folded_data = {}

    ## Extract folds
    for i in range(1, nb_folds + 1):
        # Get random evaluation fold
        eval_fold = rng.choice(subj_list_unused, size = fold_size, replace = False)
        eval_fold_mask = np.isin(input_mat[:,subj_col], eval_fold)
        eval_data_fold = input_mat[eval_fold_mask]

        # Remove already used subjects from list
        shortlist_subj = list(set(subj_list)-set(eval_fold))

        # Get corresponding training fold
        training_fold = shortlist_subj
        training_fold_mask = np.isin(input_mat[:,subj_col], training_fold)
        training_data_fold = input_mat[training_fold_mask]

        # Build dict for the current fold
        fold_i = {"train_subjs": training_fold, "train_data": training_data_fold, "eval_subjs": eval_fold, "eval_data": eval_data_fold}
        fold_i_name = "fold" + str(i)
        folded_data.update({fold_i_name: fold_i})

        # Save list of remaining subjects for evaluation
        subj_list_used.extend(eval_fold)
        subj_list_unused = list(set(subj_list)-set(subj_list_used))

    return folded_data



        



