"""This module contains tools for cleaning our datas"""

# author : Simon Bastide
# mail : simon.bastide@outlook.com

import os
import pandas as pd
import numpy as np
import sys
import config
import motor_control_tools as mct
import csv


def robot_data(subject_label, condition_label, block_number, raw_data_path):
    """Function to aggregate the datas of the robot. Only datas of the 4th axis are conserved.

    Args:
        subject_label (str): label of the subject
        condition_label (str): label of the condition
        block_number (int): number of the bloc
        raw_data_path (str): path to the file were are contained the raw datas

    Returns:
        pandas.Dataframe: One data frame containing the datas
    """
    missing_data = []
    path =  os.path.join(raw_data_path,subject_label)
    
    suffix = subject_label + '_' + condition_label + '.txt'

    # iteration times
    file_path = os.path.join(path,"iteration_times_" + suffix)
    if os.path.isfile(file_path):
        data = np.genfromtxt(file_path, delimiter = ';')
        if np.isnan(data[-1]):
            data = data[:-1]
        df = pd.DataFrame(data, columns = ['iteration_time'])
    else:
        missing_data.append(True)

    # positions
    file_path = os.path.join(path,"exo_positions_" + suffix)
    if os.path.isfile(file_path):
        data = np.genfromtxt(file_path, delimiter = ' ; ')
        data = np.reshape(data, (-1, 4))
        df = df.assign(pos_A = data[:len(df),2])
        df = df.assign(pos_FA = data[:len(df),3])
    else:
        missing_data.append(True)


    # vitesses
    file_path = os.path.join(path, "exo_vitesses_" + suffix)
    if os.path.isfile(file_path):
        data = np.genfromtxt(file_path, delimiter = ' ; ')
        data = np.reshape(data, (-1, 4))
        df = df.assign(vel_A = data[:len(df),2])
        df = df.assign(vel_FA = data[:len(df),3])
    else:
        missing_data.append(True)

    # Courants
##    file_path = os.path.join(path, "exo_courants_" + suffix)
##    if os.path.isfile(file_path):
##        data = np.genfromtxt(file_path, delimiter = ' ; ')
##        data = np.reshape(data, (-1, 4))
##        df = df.assign(courant = data[:len(df),3])
##    else:
##        missing_data.append(True)

    # Arm Force Sensor
    file_path = os.path.join(path, "FT_Arm_Sensor_" + suffix)
    if os.path.isfile(file_path):
        data = np.genfromtxt(os.path.join(path, "FT_Arm_Sensor_" + suffix), delimiter = ';')
        if np.isnan(data[-1]):
            data = data[:-1]
        data = np.reshape(data[:len(df)*6], (-1, 6))
        df = df.assign(
            Fx_A = data[:len(df),0],
            Fy_A = data[:len(df),1],
            Fz_A = data[:len(df),2],
            Tx_A = data[:len(df),3],
            Ty_A = data[:len(df),4],
            Tz_A = data[:len(df),5]
            )
    else:
        missing_data.append(True)

    # Wrist Force Sensor
    file_path = os.path.join(path, "FT_Wrist_Sensor_" + suffix)
    if os.path.isfile(file_path):
        data = np.genfromtxt(os.path.join(path, "FT_Wrist_Sensor_" + suffix), delimiter = ';')
        if np.isnan(data[-1]):
            data = data[:-1]
        data = np.reshape(data[:len(df)*6], (-1, 6))
        df = df.assign(
            Fx_FA = data[:len(df),0],
            Fy_FA = data[:len(df),1],
            Fz_FA = data[:len(df),2],
            Tx_FA = data[:len(df),3],
            Ty_FA = data[:len(df),4],
            Tz_FA = data[:len(df),5]
            )
    else:
        missing_data.append(True)
    
    if any(missing_data):
        print("!! No robot data !!")
        return False
    else:
        return df

def emg_data(subject_label, condition_label = None, block_number = None,
 raw_data_path = "raw_data", ident = False, conditions_without_number_in_file_names = None):
    """[summary]

    Args:
        subject_label ([type]): [description]
        condition_label ([type]): [description]
        block_number ([type]): [description]
        raw_data_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    path =  os.path.join(raw_data_path, subject_label)
    # TODO: Create an independent function to construct the files' names. These functions should import data only.
    if ident:
        file_name = subject_label + '_idH_a.tsv'
    else:
        
        file_name = subject_label + '_' + condition_label + '_a.tsv'
       
        
    file_path = os.path.join(path, file_name)
                
    if os.path.exists(file_path):

        fh = open(file_path)
        reader = csv.reader(fh, delimiter = '\t')
        line_names = []
               
        for line in reader:
            if line[0] == 'SAMPLE':
                line_names = line
            elif line[0] == 'plop':
                line_names = ["Force_sensor","Brachial","Biceps","Triceps_Lat","Triceps_Lng","Deltoid_Ant","Deltoid_Med","Deltoip_Post"]
        fh.close()
        
        df = pd.read_csv(file_path, sep = "\t", names = line_names, engine = 'python', skiprows=14, index_col=False)
        # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.drop('SAMPLE', axis = 1)
        df = df.drop('TIME', axis = 1)
        df = df.drop('', axis = 1)
    else:
        print("No emg data")
        return False
        
    return df

def kin_data(subject_label, condition_label = None, block_number = None,
 raw_data_path = 'raw_data', ident = False, conditions_without_number_in_file_names = None):
    """[summary]

    Args:
        subject_label ([type]): [description]
        condition_label ([type]): [description]
        block_number ([type]): [description]
        raw_data_path ([type]): [description]

    Returns:
        [type]: [description]
    """

    path =  os.path.join(raw_data_path, subject_label)
    # TODO: Create an independent function to construct the files' names. These functions should import data only.
    if ident:
        file_name = subject_label + '_idH.tsv'
    else:

        file_name = subject_label + '_' + condition_label + '.tsv'
    
    file_path = os.path.join(path, file_name)
    if os.path.exists(file_path):

        ## Change line 10 form file
        fh = open(file_path)
        reader = csv.reader(fh, delimiter = '\t')
        line_names = []

        i = 0
        j = -255
        for line in reader:
            if line[0] == 'MARKER_NAMES':
                line_names = line
            if line[0] == 'TRAJECTORY_TYPES':
                j = i
            if j+1 == i:
                line_headers = line
                break
            i = i+1

        fh.close()
        line_names_coords = []
        
        for name in line_names:
            if name != 'MARKER_NAMES':
                name_x = name + ' X'
                name_y = name + ' Y'
                name_z = name + ' Z'
                line_names_coords = np.append(line_names_coords,name_x)
                line_names_coords = np.append(line_names_coords,name_y)
                line_names_coords = np.append(line_names_coords,name_z)
                
        #print(line_names_coords)
        
        df = pd.read_csv(file_path, sep = "\t", names = line_headers,engine = 'python', skiprows=12)
        # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        # print(df)
        df = df.drop('Frame', axis = 1)
        df = df.drop('Time', axis = 1)
    else:
        print("No kinematic data")
        return False

    return df


def get_mass_and_deltaq(data_path, subj_id):
    """Renvoi la masse de l'avant bras du sujet 
    et le décalage angulaire moyen identifié
    au cours de la manip entre l'avant bras de sujet et
    l'axe 4 de l'exo

    Args:
        subj_id ([type]): [description]
    """
    file_name = os.path.join(
        data_path,
        "Limbs",
        "human_limb_identification_" + subj_id + ".txt",
    )
    with open(file_name, 'r') as f:
        values = f.read()
    return float(values.split(" ")[0]), float(values.split(" ")[1])


def get_anthropo(data_path, subj_id):
    file_name = os.path.join(data_path, "infos_participants.xlsx")
    forearm_mass_estim, delta_q_estim = get_mass_and_deltaq(config.raw_data_path, subj_id)
    data = pd.read_excel(file_name).query("subject == @subj_id").to_dict('records')[0]
    data.update({
        "forearm mass estimation" : forearm_mass_estim,
        "angular offset estimation" : delta_q_estim
    })
    data.update({
        "height" : data.get("height")/100,
        "arm length" : data.get("arm length")/100,
        "forearm length" : data.get("forearm length")/100,
        "hand length" : data.get("hand length")/100,
    })
    if not data.get("arm length") == data.get("arm length"):
        data.update({
            "arm length" : mct.anthropo.arm_length(data.get("height"))
        })
    if not data.get("forearm length") == data.get("forearm length"):
        data.update({
            "forearm length" : mct.anthropo.forearm_length(data.get("height"))
        })
    if not data.get("hand length") == data.get("hand length"):
        data.update({
            "hand length" : mct.anthropo.hand_length(data.get("height"))
        })
    return data


def get_emg_baseline(data_path, subj_id, kind = 'in'):
    # kind is needed if emg_baseline was recorded in differents conditions. Ex: in or out in the main manip.
    # emg baseline files have thus the name "..._rep_in.tsv" or "..._rep_out.tsv"
    if kind is None:
        file_name = os.path.join(
            data_path,
            subj_id,
            subj_id + "_rep_a.tsv",
        )
    else:
        file_name = os.path.join(
            data_path,
            subj_id,
            subj_id + "_rep_" + kind + "_a.tsv",
        )

    if os.path.exists(file_name):
        df = pd.read_csv(file_name, sep = "\t", engine = 'python', skiprows=13)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.drop('SAMPLE', axis = 1)
        df = df.rename({'TIME':'time'}, axis = 1)
    else:
        print("No emg data")
        return False
        
    return df


