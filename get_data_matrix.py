## Imports
# General
import os
import pickle
import numpy as np
import pandas as pd
import json


def get_data(type, params):
    """
    Function to import data from experiments or build matrix of simulated data for testing.

    Args:
      - type : string ; "FAKE": generate random data, "CALIB": get calibration expe data, "ASSIST": get assist expe data
    Output:
      - data : Nsamplesx18 array ; Full data matrix with headers: [q3,q4,dq3,dq4,FT_wrist(6),l_a,l_fa,qs,qe,dqs,dqe,mvt,subj]
    """
    if type == "FAKE":
        data = get_fake_data()
    elif type == "CALIB":
        save_path_calib = params.get('save_path')
        if os.path.isfile(save_path_calib):
            with open(save_path_calib, 'rb') as file:
                data = pickle.load(file)
            print('Loaded calibration data from pickle.')
        else:
            data = get_calib_data(params)
            print('Loaded calibration data from files...')
            with open(save_path_calib, 'wb') as file:
                pickle.dump(data, file)
            print('Saved calibration data to pickle')
    elif type == "ASSIST":
        data = None
        print("ASSIST experiment not yet handled...")
    else:
        data = None
        print("Not a valid type of data extraction target!")
    return data

def get_fake_data(nb_subj = 20, len_subj = 10000):
    """
    Function to generate random data. Allows to debug folding and ablation functions.

    Args:
      - nb_subj  : int ; number of fake subjects to generate
      - len_subj : int ; number of samples to generate per subject
    Output:
      - all_fake_data : 18xNsamples ; Fake data matrix
    """
    # Extract subjects 
    subjList = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10',
                'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']
    
    if nb_subj < len(subjList):
        subjList = subjList[:nb_subj]

    # Build data matrix
    first = True
    for subj in subjList:
        list_subj = [subj]*10000
        list_subj = np.array(list_subj).reshape(-1,1)
        fake_subj_data = np.random.rand(10000,17)
        fake_subj_data = np.concatenate([fake_subj_data,list_subj], axis = 1)
        # Save data matrix
        if first:
            all_fake_data = fake_subj_data
            first = False
        else:
            all_fake_data = np.vstack((all_fake_data,fake_subj_data))
    
    return all_fake_data

def get_calib_data(params):
    """
    Function building the complete data matrix for the human and robot.

    Args:
      - params : dict ; Set of parameters regarding the CALIB experiment
    Output:
      - data : Nsamplesx18 array ; Full data matrix with headers: [q3,q4,dq3,dq4,FT_wrist(6),l_a,l_fa,qs,qe,dqs,dqe,trial,subj]
    """
    ## Initialization
    path_expe      = params.get('path')
    nb_subj        = params.get('nb_subj')
    cond           = params.get('condition')
    nb_trials      = params.get('nb_trials')
    data_type_list = params.get('dataTypes')
    list_variables = params.get('listOfVariables')
    duration_idx   = params.get('durationIndex')
    list_subjs     = []

    ## Loop over subjects, conditions, and trials
    for i in range(0, nb_subj):
        # Path to subject
        subj = 'S' + str(i+1)
        path_subj = path_expe + '/' + subj
        
        # Get subject informations
        subj_info_path = path_subj + '/' + subj + '.json'
        with open(subj_info_path, 'r', encoding='utf-8') as file:
            subj_info = json.load(file)
        
        # Get anthropometrics
        l_a  = subj_info.get('measures').get('arm')
        l_fa = subj_info.get('measures').get('forearm')

        # Path to condition [SJ: single-joint; MJ: multi-joint]
        path_cond = path_subj + '/' + cond

        # Empty list to store trials
        list_trials = []
        for j in range(0, nb_trials):
            # Path to trial
            path_trial = path_cond + '/T' + str(j)
            # Empty list to store data
            list_dtypes = []
            for type in data_type_list:
                # Get human, robot and force data
                path_data = path_trial + '/' + type + '.csv'
                data = pd.read_csv(path_data)
                if type == 'humanVelocities':
                    data = data.rename(columns = {'shoulder_elv': 'dshoulder_elv', 'elbow_flexion': 'delbow_flexion'})
                list_dtypes.append(data)

            # Concatenate al data of the trial
            data_one_trial = pd.concat(list_dtypes, axis = 1)

            # Add anthropo, trial and subject identifiers
            data_one_trial['l_a']   = l_a
            data_one_trial['l_fa']  = l_fa
            data_one_trial['trial'] = j
            data_one_trial['subj']  = subj

            # Keep only relevant variables
            data_one_trial = data_one_trial[list_variables]

            # Reorder columns
            data_one_trial.insert(10, 'l_a', data_one_trial.pop('l_a'))
            data_one_trial.insert(11, 'l_fa', data_one_trial.pop('l_fa'))
            data_one_trial.insert(16, 'trial', data_one_trial.pop('trial'))
            data_one_trial.insert(17, 'subj', data_one_trial.pop('subj'))

            # Copy dataframe to new variable to avoid fragmentation in memory when using insert
            data_one_trial_unfrag = data_one_trial.copy()

            # Keep slice of data to ensure common durations
            data_one_trial_slice = data_one_trial_unfrag.iloc[0:duration_idx]

            # Store trial
            list_trials.append(data_one_trial_slice.copy())

            # Reset dataframes before next iteration
            data_one_trial = pd.DataFrame({})
            data_one_trial_unfrag = pd.DataFrame({})
            data_one_trial_slice = pd.DataFrame({})

        # Concatenate all trials
        df_one_subj = pd.concat(list_trials, axis = 0)

        # Store subject
        list_subjs.append(df_one_subj)

    # Concatenate all subjects
    df_all_data = pd.concat(list_subjs, axis = 0)

    # Return complete array
    return df_all_data.to_numpy()

