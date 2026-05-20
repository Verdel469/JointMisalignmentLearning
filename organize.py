"""This script is made to organize the data and store it in a easily reusable form.

It takes the data from raw_data folder and store the organized data in data folder"""

# author : Simon Bastide
# mail : simon.bastide@outlook.com

import os
import numpy as np
import pandas as pd
import config
import sys
import clean_import
import motor_control_tools as mct
from pathlib import Path

for subject in config.subjects_label:
    print("\n########## {} ##########".format(subject))
    subject_path = os.path.join(config.data_path, subject)

    # Load, clean and write data of the identification process
    clean_ident_file_path_emg = os.path.join(subject_path, '_'.join([subject, 'ident', 'emg'])) + '.csv'
    clean_ident_file_path_kin = os.path.join(subject_path, '_'.join([subject, 'ident', 'kin'])) + '.csv'
    subject_label_raw_data = subject
    if os.path.isfile(clean_ident_file_path_emg):
        print("Identification emg file already exsts")
    else:
        df_emg_ident = clean_import.emg_data(
            subject,
            ident=True,
            raw_data_path=config.raw_data_path,
            conditions_without_number_in_file_names=config.spec.get("conditions_without_block_number_in_file_names"))
        if (type(df_emg_ident) is bool):
            print("emg file does not exist")
        else:
            df_emg_ident.to_csv(clean_ident_file_path_emg, index = False)
    if os.path.isfile(clean_ident_file_path_kin):
        print("Identification kinematic file already exsts")
    else:
        df_kin_ident = clean_import.kin_data(
            subject,
            ident=True,
            raw_data_path=config.raw_data_path,
        conditions_without_number_in_file_names=config.spec.get("conditions_without_block_number_in_file_names"))
        if (type(df_kin_ident) is bool):
            print("kin file does not exist")
        else:
            df_kin_ident.to_csv(clean_ident_file_path_kin, index = False)


    
    for condition in config.spec.get("conditions"):
        print("||||||||| Condition {} |||||||||".format(condition))
        for block in range(1,config.spec.get("block_per_condition").get(condition)+1):
            print("----- Block {} -----".format(block))            
            
            clean_file_path = os.path.join(subject_path, '_'.join([subject,condition]))
            Path(subject_path).mkdir(parents=True, exist_ok=True)
            clean_file_path_kin = clean_file_path + '_kin.csv'
            clean_file_path_emg = clean_file_path + '_emg.csv'
            clean_file_path_rob = clean_file_path + '_rob.csv'
            
            if os.path.isfile(clean_file_path_kin):
                print("Kinematic file already exists")
            else:
                # Import kinematic
                print("Loading kinematic datas...")
                df_kin = clean_import.kin_data(
                    subject_label=subject,
                    condition_label=condition,
                    block_number=block,
                    raw_data_path=config.raw_data_path,
                    conditions_without_number_in_file_names=config.spec.get("conditions_without_block_number_in_file_names")
                )
                if not (type(df_kin) is bool):
                    print("Saving kinematic datas...")
                    df_kin.to_csv(clean_file_path_kin, index = False)
            
            if os.path.isfile(clean_file_path_emg):
                print("Emg file already exists")
            else:
                # Import emgs
                print("Loading emg datas...")
                df_emg = clean_import.emg_data(
                    subject_label=subject,
                    condition_label=condition,
                    block_number=block,
                    raw_data_path=config.raw_data_path,
                    conditions_without_number_in_file_names=config.spec.get("conditions_without_block_number_in_file_names")
                )
                if not (type(df_emg) is bool):
                    print("Saving emg datas...")
                    df_emg.to_csv(clean_file_path_emg, index = False)
            
            if os.path.isfile(clean_file_path_rob):
                print("Robotic file already exists")
            else:
                # Import robot
                print("Loading robot datas...")
                df_rob = clean_import.robot_data(
                    subject_label=subject,
                    condition_label=condition,
                    block_number=block,
                    raw_data_path=config.raw_data_path,
                    )
                if not (type(df_rob) is bool):
                    print("Resampling robot datas...")
                    columns = ["iteration_time","pos_A", "pos_FA", "vel_A", "vel_FA",
                               "Fx_A","Fy_A","Fz_A",
                               "Tx_A", "Ty_A","Tz_A",
                               "Fx_FA","Fy_FA","Fz_FA",
                               "Tx_FA", "Ty_FA","Tz_FA"]
                    df_rob_resampled = pd.DataFrame()
                    for (columnName, columnData) in df_rob[columns].iteritems():
                        data_resampled = mct.signal.resample_non_uniform(
                                x = np.array(df_rob.iteration_time.cumsum()),
                                y = np.array(columnData),
                                new_sample_rate = config.spec.get("sample_rate_rob"),
                                )
                        df_rob_resampled = df_rob_resampled.assign(**{columnName : data_resampled})
                    print("Saving robot datas...")
                    df_rob_resampled.to_csv(clean_file_path_rob, index = False)

            
