## Imports
# General
import numpy as np


def get_data(type):
    """
    Function to import data from experiments or build matrix of simulated data for testing.

    Args:
      - type : string ; "FAKE": generate random data, "CALIB": get calibration expe data, "ASSIST": get assist expe data
    Output:
      - data : 18xNsamples ; Full data matrix with headers: [q3,q4,dq3,dq4,FT_wrist(6),l_a,l_fa,qs,qe,dqs,dqe,mvt,subj]
    """
    if type == "FAKE":
        data = get_fake_data()
    elif type == "CALIB":
        data = None
        print("CALIB experiment not yet handled...")
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