import pandas as pd
import numpy as np
import config
import sys
sys.path.insert(1, "C:\\Users\\simon\\OneDrive\\MaThese\\Expe_4")
import manage_expe as me
import motor_control_tools as mct
import os
import matplotlib.pyplot as plt
from math import *

# Boucle sur les sujets
for subject in config.subjects_label:
    print("\n########## {} ##########".format(subject))
    # Boucle sur les conditions
    for condition in config.spec.get("conditions"):
        print("||||||||| Condition {} |||||||||".format(condition))
        # Boucle sur les blocs
        for block in range(1,config.spec.get("block_per_condition").get(condition)+1):
            print("----- Block {} -----".format(block))
            file_base = '_'.join([subject,condition]) 
            # kin

            trials_sep_file_name = os.path.join(config.data_path,subject, 'trials_separators_' + file_base + '.csv')
            block_sep_file_name = os.path.join(config.data_path,subject, 'blocks_separators_' + file_base + '.csv')

            if os.path.isfile(trials_sep_file_name) and os.path.isfile(block_sep_file_name):
                print("Separator files already exists")
            else:
                df_sep = pd.DataFrame()
                df_block_bounds = pd.DataFrame()
                path_kin = os.path.join(config.data_path,subject,file_base + '_kin.csv')
                if os.path.exists(path_kin):
                    index_xyz = np.array(pd.read_csv(path_kin, usecols = ['Index_Dist X', 'Index_Dist Y', 'Index_Dist Z']))
                    index_xyz_filt = mct.signal.filter(index_xyz, config.spec.get("sample_rate_kin"))
                    V = mct.kinematic.tangential_velocity(index_xyz_filt, config.spec.get("sample_rate_kin"))
                    X = index_xyz_filt[:,2] + index_xyz_filt[:,0]
                    print("Data from Qualisys")
                    # Enlever début du block et couper la séquence de calibration des cibles
                    block_bounds_kin = me.block2trials.manual_cutting(X,V,config.spec.get("sample_rate_kin"), confirmation=False)
                    # TODO: block_bounds[1] : block_bounds[2] --> calibration cibles. à traiter. Enregistrer dans un fichier ? 
                
                    X_trials = X[block_bounds_kin[0]:block_bounds_kin[1]]
                    V_trials = V[block_bounds_kin[0]:block_bounds_kin[1]]
                    sep_indexes_kin = me.block2trials.cutting(X_trials, V_trials, config.spec.get("sample_rate_kin"), treshold=270)
                    for i in range(0,len(sep_indexes_kin)):
                        sep_indexes_kin[i]=int(sep_indexes_kin[i])
                    df_sep = df_sep.assign(sep_kin = sep_indexes_kin)
                    df_sep = df_sep.assign(sep_emg = sep_indexes_kin*int(floor(config.spec.get("sample_rate_emg")/config.spec.get("sample_rate_kin"))))
                    df_block_bounds = df_block_bounds.assign(sep_kin = block_bounds_kin)
                    df_block_bounds = df_block_bounds.assign(sep_emg = block_bounds_kin*int(floor(config.spec.get("sample_rate_emg")/config.spec.get("sample_rate_kin"))))
                else:
                    print("No kinematic data")

                # rob
                path_rob = os.path.join(config.data_path,subject,file_base + '_rob.csv')
                if not (condition == 'NE') and os.path.exists(path_rob):
                    print("Data from the robot")
                    X = -np.array(pd.read_csv(path_rob, usecols = ['pos_A']))[:,0]
                    V = abs(np.array(pd.read_csv(path_rob, usecols = ['vel_A']))[:,0]) \
                      + abs(np.array(pd.read_csv(path_rob, usecols = ['vel_FA']))[:,0])

                    # Enlever début du block et couper la séquence de calibration des cibles
                    block_bounds_rob = me.block2trials.manual_cutting(X,V,config.spec.get("sample_rate_rob"), confirmation=False)
                    X_trials = X[block_bounds_rob[0]:block_bounds_rob[1]]
                    V_trials = V[block_bounds_rob[0]:block_bounds_rob[1]]

                    sep_indexes_rob = me.block2trials.cutting(X_trials, V_trials, config.spec.get("sample_rate_rob"), treshold=13)
                    df_sep = df_sep.assign(sep_rob = sep_indexes_rob)
                    df_block_bounds = df_block_bounds.assign(sep_rob = block_bounds_rob)
                else:
                    print("No robot data")

                if not df_sep.empty:
                    df_sep.to_csv(trials_sep_file_name, index = False)
                if not df_block_bounds.empty:
                    df_block_bounds.to_csv(block_sep_file_name, index = False)

            # TODO: Gerer le cas ou pas le même nombre de sep

  
