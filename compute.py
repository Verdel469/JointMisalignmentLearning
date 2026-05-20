"""This script is made for compute parameters and store them in a .csv .
It takes the data from data folder. The script oraganize.py must be run to fill the data folder"""

# author : Simon Bastide
# mail : simon.bastide@outlook.com

import pandas as pd
import numpy as np
import sys
import clean_import
import config
import matplotlib.pyplot as plt
import os
import pickle
import json

import manage_expe as me
import motor_control_tools as mct


# TODO Compute functions do 2 things : compute parameters and save movement to .pickle. Refactor like this 2 functionality are separated

def save_params_files(list_of_dict_of_params, variable, listTargets, subj = None):
    if config.cutoff_velocity_treshold_is_rel:
        cut_off_id = "_cut-off-rel-"
    else:
        cut_off_id = '_cut-off-abs-'
    if config.take_overshoot_into_account:
        identifier = config.manip + "_" + variable + cut_off_id + str(config.cutoff_velocity_treshold) + "_ovrsht"
    else:
        identifier = config.manip + "_" + variable + cut_off_id + str(config.cutoff_velocity_treshold)

    if subj is not None:
        identifier = config.manip + "_" + variable + '_' + subj
    
    params_file = os.path.join("all_params", identifier + '.csv')

    all_params_df = pd.DataFrame(list_of_dict_of_params)
    all_targets_df = pd.concat(listTargets, axis = 0, ignore_index=True)
    all_params_df = all_params_df.rename(columns={"direction": "UpDown dir"})
    print(all_targets_df)
    all_params_df = all_params_df.merge(all_targets_df, how = 'outer', on = ['subject','condition','movement'])

    with open(params_file, 'a') as f:
        all_params_df.to_csv(f,index = False, mode = 'a', header=not f.tell())


def selector(compute_based_on):
    """Function to select on wich variable the parameters will be computed

    Args:
        compute_based_on (str): Name of the variable to be computed. Available choices 'all', 'index', 'elbow', 'rob'
    """
    if compute_based_on == 'index':
        make_loop(compute_index, 'index')
    elif compute_based_on == 'rob':
        make_loop(compute_rob, 'rob')
    elif compute_based_on == 'emg':
        make_loop(compute_emg, 'emg')
    elif compute_based_on == 'all':
        print("\nCompute on index\n")
        make_loop(compute_index, 'index')
        print("\nCompute on robot\n")
        make_loop(compute_rob, 'rob')
        print("\nCompute on emgs\n")
        make_loop(compute_emg, 'emg')

def make_loop(compute_function, variable):
    all_params = []
    list_df_targets = []
    
    for subject in config.subjects_label:
        print("\n########## {} ##########".format(subject))
        #subj_data = clean_import.get_anthropo(config.data_path, subject)
        subj_data = 0
        if variable == 'emg':
            emg_max_act_filt_rect = mct.emg_processing.get_emgs_maximum_activation(subject, config, on = 'filt_rect', time_in=False, just_one_block=True)
            emg_max_act_envelope = mct.emg_processing.get_emgs_maximum_activation(subject, config, on = 'envelope', time_in=False, just_one_block=True)
        for condition in config.spec.get("conditions"):
            print("Condition ------->  {}".format(condition))
            for block in range(1,config.spec.get("block_per_condition").get(condition)+1):
                print("----- Block {} -----".format(block))
                try :
                    block_data = load_block(subject, condition, block, variable)
                    # Changed sep_rob <--> sep_kin !
                    for mov_n in range(0, len(block_data.get('trials_sep').sep_kin) -1):
                        # Iteration sur le nombre de mouvement sur la cinématique (toujours
                        # des datas cinématique et nombre de mouvements egaux entre data cinématique
                        # et robotique)
                        print('#' + str(mov_n))#, end = ' ')
                        param_dict = mct.kinematic.get_id_dict(
                                    subj_id=subject,
                                    cond_id=condition,
                                    block_id=block,
                                    mov_id=mov_n,
                                    )
                        movement = get_movement(block_data, mov_n, variable)
                        if variable == 'emg':
                            updated_param_list_of_dict = compute_function(param_dict, movement, subj_data, emg_max_act_filt_rect, emg_max_act_envelope)
                            for updated_param_dict in updated_param_list_of_dict:
                                all_params.append(updated_param_dict.copy())
                        else:
                            print('Updating params dict...')
                            updated_param_dict = compute_function(param_dict, movement, subj_data, condition)
                            all_params.append(updated_param_dict.copy())
                            print('Updated.')
                    print("\n")
                except FileNotFoundError:
                    print(subject + " " + condition + " not found")
                    pass
                    print("File not Found")
            try :
                targets_path = get_SuccessiveTargets_path(config.raw_data_path,subject, condition)
                df_targets = pd.read_csv(targets_path, encoding = "ISO-8859-1", engine='python')
                df_targets = df_targets.drop(columns = ['Unnamed: 0'])
                list_df_targets.append(df_targets)
            except FileNotFoundError:
                print("path:" + targets_path)
                pass
                print("File not Found")
            
    save_params_files(all_params, variable, list_df_targets)
    # save_params_files(all_params, variable)

def get_SuccessiveTargets_path(gen_path,subj,cond):
    
    path = gen_path + "\\" + subj + "\\Successive_Targets_" + subj + "_" + cond + ".csv"

    return path


def get_movement(block_data, movement_number, variable):
    if variable in ['index', 'elbow']:
        block_start = block_data.get('blocks_sep').sep_kin[0]
        movement_start = block_data.get('trials_sep').sep_kin[movement_number] + block_start
        movement_stop = block_data.get('trials_sep').sep_kin[movement_number + 1] + block_start
        block_df = block_data.get('df')
        return block_df.iloc[movement_start:movement_stop].reset_index(drop = True)

    if variable == 'emg':
        block_start = int(block_data.get('blocks_sep').sep_emg[0])
        movement_start = int(block_data.get('trials_sep').sep_emg[movement_number]) + block_start
        movement_stop = int(block_data.get('trials_sep').sep_emg[movement_number + 1]) + block_start
        block_df = block_data.get('df')
        return block_df.iloc[movement_start:movement_stop].reset_index(drop = True)

    elif variable == 'rob':
        block_start = block_data.get('blocks_sep').sep_rob[0]
        movement_start = block_data.get('trials_sep').sep_rob[movement_number] + block_start
        movement_stop = block_data.get('trials_sep').sep_rob[movement_number + 1] + block_start
        
        return block_data.get('df').iloc[movement_start:movement_stop].reset_index(drop = True)

def load_block(subj_id, cond_id, block_id, variable):
    if config.spec.get('maxblockcond') != 1:
        file = '_'.join([subj_id, cond_id, str(block_id)])
    else:
        file = '_'.join([subj_id, cond_id])
    
    if variable in ['index', 'elbow']:
        file_path = os.path.join(config.data_path,subj_id, file + "_kin.csv")
        
    elif variable == 'rob':
        file_path = os.path.join(config.data_path,subj_id, file + "_rob.csv")
    elif variable == 'emg':
        file_path = os.path.join(config.data_path,subj_id, file + "_emg.csv")

    if os.path.isfile(file_path):
        trials_sep = pd.read_csv(os.path.join(config.data_path,subj_id,"trials_separators_" + file + ".csv")).astype(int)
        blocks_sep = pd.read_csv(os.path.join(config.data_path,subj_id,"blocks_separators_" + file + ".csv"))
        df = pd.read_csv(file_path)
        return {
            'trials_sep' : trials_sep,
            'blocks_sep' : blocks_sep,
            'df' : df,
        }
        
    else:
        print(file_path + " Not found")
        raise FileNotFoundError

def get_movement_path(params_dict, variable):
    """Return the movement name to save it as a .pickle file

    Args:
        params_dict (dict): [description]
        variable (str): [description]

    Returns:
        str: full path of the movement
    """
    if config.spec.get('maxblockcond') != 1:
        movement_name = "_".join([
            params_dict.get('subject'),
            params_dict.get('condition'),
            str(params_dict.get('block')),
            'm' + str(params_dict.get('movement')),
            ])
    else:
        movement_name = "_".join([
            params_dict.get('subject'),
            params_dict.get('condition'),
            'm' + str(params_dict.get('movement')),
            ])
    return config.movements_path + "/" + variable + "_" + movement_name + ".pickle"

def compute_index(params_dict, movement, subj_data, condition):
    """Compute parameters based on data of the index qualisys marker

    Args:
        params_dict ([type]): [description]
        movement ([type]): [description]

    Returns:
        [type]: [description]
    """
    movement_full_path = get_movement_path(params_dict, 'index')

    ## Get index data
    pos_xyz = mct.signal.filter(movement[['Index_Dist X', 'Index_Dist Y', 'Index_Dist Z']], config.spec.get("sample_rate_kin"), low_pass = 5)/1000 # Conversion en m
    
    index_vel = mct.kinematic.tangential_velocity(pos_xyz, config.spec.get("sample_rate_kin"))
    index_pos = mct.kinematic.travelled_distance(index_vel, config.spec.get("sample_rate_kin"))
    index_acc = mct.signal.diff_keep_length(index_vel, config.spec.get("sample_rate_kin"))

    ## Compute human upper-limb angles
    # Get acromion
    acromion = mct.signal.filter(movement[['Acromion X', 'Acromion Y', 'Acromion Z']], config.spec.get("sample_rate_kin"), low_pass = 5)/1000 # Conversion en m
    # Get elbow
    elb_int = mct.signal.filter(movement[['Elb_Int X', 'Elb_Int Y', 'Elb_Int Z']], config.spec.get("sample_rate_kin"), low_pass = 5)/1000 # Conversion en m
    elb_ext = mct.signal.filter(movement[['Elb_Ext X', 'Elb_Ext Y', 'Elb_Ext Z']], config.spec.get("sample_rate_kin"), low_pass = 5)/1000 # Conversion en m
    # Get wrist
    # wrist_int_prox = mct.signal.filter(movement[['Ort_Int_Prox X', 'Ort_Int_Prox Y', 'Ort_Int_Prox Z']], config.spec.get("sample_rate_kin"), low_pass = 5)/1000 # Conversion en m
    # wrist_int_dist = mct.signal.filter(movement[['Ort_Int_Dist X', 'Ort_Int_Dist Y', 'Ort_Int_Dist Z']], config.spec.get("sample_rate_kin"), low_pass = 5)/1000 # Conversion en m
    # wrist_ext_prox = mct.signal.filter(movement[['Ort_Ext_Prox X', 'Ort_Ext_Prox Y', 'Ort_Ext_Prox Z']], config.spec.get("sample_rate_kin"), low_pass = 5)/1000 # Conversion en m
    # wrist_mid_dist = mct.signal.filter(movement[['Ort_Mid_Dist X', 'Ort_Mid_Dist Y', 'Ort_Mid_Dist Z']], config.spec.get("sample_rate_kin"), low_pass = 5)/1000 # Conversion en m
    # wrist_ext_dist = mct.signal.filter(movement[['Ort_Ext_Dist X', 'Ort_Ext_Dist Y', 'Ort_Ext_Dist Z']], config.spec.get("sample_rate_kin"), low_pass = 5)/1000 # Conversion en m
    # Get joints
    elbow = (elb_int + elb_ext)/2
    wrist = wrist_ext_dist = mct.signal.filter(movement[['FA_Dist X', 'FA_Dist Y', 'FA_Dist Z']], config.spec.get("sample_rate_kin"), low_pass = 5)/1000 # Conversion en m (wrist_int_prox + wrist_int_dist + wrist_ext_prox + wrist_ext_dist)/4
    shoulder = np.subtract(acromion, np.multiply([0,0,1], 0.02)) # Centre of the shoulder set 2 cm below the acromion
    # Compute segments lenghts
    l_uparm = np.linalg.norm(elbow - acromion, axis = 1)
    l_farm  = np.linalg.norm(wrist - elbow, axis = 1)
    # Compute angles
    sho_angle = np.acos(np.dot((elbow - shoulder)/l_uparm[:,None],[0,0,-1]))
    elb_angle = np.acos(np.sum((wrist - elbow)/l_farm[:,None] * (elbow - shoulder)/l_uparm[:,None], axis = 1))
    
    ## Get segmentation bounds
    bounds = mct.kinematic.movement_bounds(
            position = index_pos,
            sample_rate = config.spec.get("sample_rate_kin"),
            velocity = index_vel,
            treshold = config.cutoff_velocity_treshold,
            relativ_treshold=config.cutoff_velocity_treshold_is_rel
            )
    
    ## Extract bounded index movement
    index_pos_bounded = index_pos[bounds[0]:bounds[1]]
    pos_x_bounded = pos_xyz[bounds[0]:bounds[1],0]
    pos_y_bounded = pos_xyz[bounds[0]:bounds[1],1]
    pos_z_bounded = pos_xyz[bounds[0]:bounds[1],2]
    index_vel_bounded = index_vel[bounds[0]:bounds[1]]
    index_acc_bounded = index_acc[bounds[0]:bounds[1]]

    ## Extract bounded joint angles & segments
    sho_angle_bounded = sho_angle[bounds[0]:bounds[1]]
    elb_angle_bounded = elb_angle[bounds[0]:bounds[1]]
    l_uparm_bounded   = l_uparm[bounds[0]:bounds[1]]
    l_farm_bounded    = l_farm[bounds[0]:bounds[1]]

    ## Extract movement direction
    if movement['Index_Dist Z'][bounds[0]+1]< movement['Index_Dist Z'][bounds[1]-1]:
        direction = 'up'
    else:
        direction = 'down'

    if mct.signal.cross_0_times(index_acc_bounded) > 1:
        valid = False
    else:
        valid = True
    params_dict.update({'valid':valid})

    # Eregistrement des trajectoires dans un .pickle
    if not os.path.exists(movement_full_path):
        index_mov = pd.DataFrame({
            'posNorm' : index_pos[:-1],
            'posX': pos_xyz[:,0],
            'posY': pos_xyz[:,1],
            'posZ': pos_xyz[:,2],
            'vel' : index_vel,
            'acc' : index_acc,
            'q_sh': sho_angle,
            'q_el': elb_angle,
            'l_ua': l_uparm,
            'l_fa': l_farm
        })
        movement_to_pickle = {
            'bounds' : bounds,
            'direction' : direction,
            'df_mov' : index_mov,
        }
        with open (movement_full_path, 'wb') as f:
            pickle.dump(movement_to_pickle, f, pickle.HIGHEST_PROTOCOL)
            
    
    params_dict.update({'direction' : direction})
    params_dict.update(
        mct.kinematic.params(
            position = index_pos_bounded,
            posx = pos_x_bounded,
            posy = pos_y_bounded,
            posz = pos_z_bounded,
            velocity = index_vel_bounded,
            acceleration = index_acc_bounded,
            sample_rate = config.spec.get("sample_rate_kin"),
            plot = False,
            
            )
    )
    params_dict.update({"file_name" : movement_full_path})

    return params_dict



def compute_rob(params_dict, movement, subj_data, condition):
    ## Select all imported data
    movement_full_path = get_movement_path(params_dict, 'rob')

    ## Select iteration times
    it_time = np.array(movement['iteration_time'])
    
    ## Select and compute all kinematic data
    # Positions
    rob_pos_FA = -mct.signal.filter(np.array(movement['pos_FA']), config.spec.get("sample_rate_rob"), low_pass = 5)
    rob_pos_A = -mct.signal.filter(np.array(movement['pos_A']), config.spec.get("sample_rate_rob"), low_pass = 5)
    # Velocities
    rob_vel_FA = mct.signal.filter(mct.signal.diff_keep_length(rob_pos_FA, it_time, True), config.spec.get("sample_rate_rob"), low_pass = 5)
    rob_vel_A = mct.signal.filter(mct.signal.diff_keep_length(rob_pos_A, it_time, True), config.spec.get("sample_rate_rob"), low_pass = 5)
    # Accelerations
    rob_acc_FA = mct.signal.diff_keep_length(rob_vel_FA, it_time, True)
    rob_acc_A = mct.signal.diff_keep_length(rob_vel_A, it_time, True)

    ## Select forces and torques data
    # Forces forearm
    fx_FA = mct.signal.filter(movement['Fx_FA'], config.spec.get("sample_rate_rob"), low_pass = 5)
    fy_FA = mct.signal.filter(movement['Fy_FA'], config.spec.get("sample_rate_rob"), low_pass = 5)
    fz_FA = mct.signal.filter(movement['Fz_FA'], config.spec.get("sample_rate_rob"), low_pass = 5)
    # Forces arm
    fx_A = mct.signal.filter(movement['Fx_A'], config.spec.get("sample_rate_rob"), low_pass = 5)
    fy_A = mct.signal.filter(movement['Fy_A'], config.spec.get("sample_rate_rob"), low_pass = 5)
    fz_A = mct.signal.filter(movement['Fz_A'], config.spec.get("sample_rate_rob"), low_pass = 5)
    # Torques forearm
    tx_FA = mct.signal.filter(movement['Tx_FA'], config.spec.get("sample_rate_rob"), low_pass = 5)
    ty_FA = mct.signal.filter(movement['Ty_FA'], config.spec.get("sample_rate_rob"), low_pass = 5)
    tz_FA = mct.signal.filter(movement['Tz_FA'], config.spec.get("sample_rate_rob"), low_pass = 5)
    # Torques arm
    tx_A = mct.signal.filter(movement['Tx_A'], config.spec.get("sample_rate_rob"), low_pass = 5)
    ty_A = mct.signal.filter(movement['Ty_A'], config.spec.get("sample_rate_rob"), low_pass = 5)
    tz_A = mct.signal.filter(movement['Tz_A'], config.spec.get("sample_rate_rob"), low_pass = 5)
    
    ## Compute movement bounds
    bounds = mct.kinematic.movement_bounds(
        position = rob_pos_FA,
        sample_rate = config.spec.get("sample_rate_rob"),
        velocity = np.abs(rob_vel_FA) + np.abs(rob_vel_A),
        treshold = config.cutoff_velocity_treshold/2,
        relativ_treshold=config.cutoff_velocity_treshold_is_rel,
        )

    ## Get bounded iteration times
    it_time_b = it_time[bounds[0]:bounds[1]]
    
    ## Get bounded kinematic data
    # Forearm
    rob_pos_FA_b = rob_pos_FA[bounds[0]:bounds[1]]
    rob_vel_FA_b = rob_vel_FA[bounds[0]:bounds[1]]
    rob_acc_FA_b = rob_acc_FA[bounds[0]:bounds[1]]
    # Arm
    rob_pos_A_b = rob_pos_A[bounds[0]:bounds[1]]
    rob_vel_A_b = rob_vel_A[bounds[0]:bounds[1]]
    rob_acc_A_b = rob_acc_A[bounds[0]:bounds[1]]

    ## Get bounded forces data
    # Forearm
    fx_FA_b = fx_FA[bounds[0]:bounds[1]]
    fy_FA_b = fy_FA[bounds[0]:bounds[1]]
    fz_FA_b = fz_FA[bounds[0]:bounds[1]]
    # Arm
    fx_A_b = fx_A[bounds[0]:bounds[1]]
    fy_A_b = fy_A[bounds[0]:bounds[1]]
    fz_A_b = fz_A[bounds[0]:bounds[1]]

    ## Get bounded torques data
    # Forearm
    tx_FA_b = tx_FA[bounds[0]:bounds[1]]
    ty_FA_b = ty_FA[bounds[0]:bounds[1]]
    tz_FA_b = tz_FA[bounds[0]:bounds[1]]
    # Arm
    tx_A_b = tx_A[bounds[0]:bounds[1]]
    ty_A_b = ty_A[bounds[0]:bounds[1]]
    tz_A_b = tz_A[bounds[0]:bounds[1]]

    ## Compute direction
    #if rob_pos_Forearm_bounded[0] < rob_pos_Forearm_bounded[-1]:
    #    direction = 'up'
    #else:
    #    direction = 'down'

    ## Acceleration signal is valid ?
    #if mct.signal.cross_0_times(rob_acc_Forearm_bounded) > 3:
    #    valid_acc = False
    #else:
    #    valid_acc = True
    #params_dict.update({'valid':valid_acc})
    #print('Validity updated in params dict.')

    ## Save movement file
    if not os.path.exists(movement_full_path):
        rob_mov = pd.DataFrame({ 'it_time': it_time,
            'pos_FA':rob_pos_FA, 'vel_FA':rob_vel_FA, 'acc_FA':rob_acc_FA,
            'Fx_FA' :-fx_FA    , 'Fy_FA' :-fy_FA    , 'Fz_FA' :-fz_FA    ,
            'Tx_FA' :-tx_FA    , 'Ty_FA' :-ty_FA    , 'Tz_FA' :-tz_FA    ,
            'pos_A' :rob_pos_A , 'vel_A' :rob_vel_A , 'acc_A' :rob_acc_A ,
            'Fx_A'  :-fx_A     , 'Fy_A'  :-fy_A     , 'Fz_A'  :-fz_A     ,
            'Tx_A'  :-tx_A     , 'Ty_A'  :-ty_A     , 'Tz_A'  :-tz_A
        })
        movement_to_pickle = {
            'bounds' : bounds, 'df_mov' : rob_mov
        }
        with open (movement_full_path, 'wb') as f:
            pickle.dump(movement_to_pickle, f, pickle.HIGHEST_PROTOCOL)

    ## Load movement 
    if not config.cutoff_velocity_treshold_is_rel:
        # si compute avec treshold abs --> add bounds_abs to movements .pickle
        with open (movement_full_path, 'rb')  as f:
            movement_to_modify = pickle.load(f)

        movement_to_modify.update({
            'bounds_abs' : bounds 
        })  

        with open (movement_full_path, 'wb') as f:
            pickle.dump(movement_to_modify, f, pickle.HIGHEST_PROTOCOL)

    ## Compute parameters
    params_dict.update({'direction' : 'NA'})
    params_dict.update(
        mct.comp_rob_params.rob_params(
            rob_pos_FA_b, rob_vel_FA_b, rob_acc_FA_b,
            -fx_FA_b    , -fy_FA_b    , -fz_FA_b,
            -tx_FA_b    , -ty_FA_b    , -tz_FA_b,
            it_time_b   , s = 'FA'    , sample_rate = config.spec.get("sample_rate_rob")
            )
    )
    params_dict.update(
        mct.comp_rob_params.rob_params(
            rob_pos_A_b, rob_vel_A_b, rob_acc_A_b,
            -fx_A_b    , -fy_A_b    , -fz_A_b,
            -tx_A_b    , -ty_A_b    , -tz_A_b,
            it_time_b  , s = 'A'    ,sample_rate = config.spec.get("sample_rate_rob")
            )
    )

    ## Save movement path
    params_dict.update({"file_name" : movement_full_path})

    return params_dict

def compute_emg(params_dict, movement, subj_data, emg_max_act_filt_rect, emg_max_act_envelope):

    movement_full_path = get_movement_path(params_dict, 'emg')
    movement_full_path_kin = get_movement_path(params_dict, 'index')
    
    # Get emg bounds according to kinematics (previously computed)
    params_emg = []
    if os.path.exists(movement_full_path_kin):
        with open(movement_full_path_kin, 'rb') as f:
            mov_kin = pickle.load(f)
        electromechanical_delay_frame = int(config.spec.get('sample_rate_emg') * config.electromechanical_delay)
        if config.cutoff_velocity_treshold_is_rel:
            bounds_emg = np.array(mov_kin.get('bounds'))*20#*10 - electromechanical_delay_frame
        else:
            bounds_emg = np.array(mov_kin.get('bounds_abs'))*20#*10 - electromechanical_delay_frame

        if config.take_overshoot_into_account:
            bounds_emg = np.array(mov_kin.get('bounds_with_overshoot'))*20# - electromechanical_delay_frame
        # plt.plot(movement)
        #plt.plot(movement[bounds_emg[0]:bounds_emg[1]])
        #plt.show()
        movement = movement.drop(columns = ['Force_sensor'])
        emgs = movement.dropna()
        # list of dict containing params needed here because the function return
        # multiple lines for the final dataframe containing all params (see make_loop)

        if not emgs.empty:
            #emgs = emgs.drop(columns = ['time'])
            print("input length",len(emgs))

            emgs_filt_rect = emgs.apply(mct.emg_processing.filt_rect, args=[config.spec.get("sample_rate_emg")]) #config.spec.get('sample_rate_emg')
            print("filt_rect length",len(emgs_filt_rect))
            emgs_filt_rect_norm = emgs_filt_rect/emg_max_act_filt_rect*100 # Normalisation par rapport au max sur toute la manip
            print("filt_rect norm length",len(emgs_filt_rect_norm))
            
            # Enveloppe et cocontraction coude
            flex_elbow_env = mct.emg_processing.envelope(np.mean([emgs_filt_rect['Biceps'],emgs_filt_rect['Brachial']], axis=0), config.spec.get("sample_rate_emg"))
            ext_elbow_env = mct.emg_processing.envelope(np.mean([emgs_filt_rect['Triceps_Lat'],emgs_filt_rect['Triceps_Lng']], axis=0), config.spec.get("sample_rate_emg"))
            coconctraction_elbow_all = mct.emg_processing.cocontraction(flex_elbow_env, ext_elbow_env)
            coconctraction_elbow_mvt = mct.emg_processing.cocontraction(
                 flex_elbow_env[bounds_emg[0]-electromechanical_delay_frame:bounds_emg[1]+electromechanical_delay_frame],
                 ext_elbow_env[bounds_emg[0]-electromechanical_delay_frame:bounds_emg[1]+electromechanical_delay_frame]
                )

            emgs_bounded = emgs.iloc[bounds_emg[0]:bounds_emg[1]]
            emgs_filt_rect_bounded = emgs_filt_rect_norm.iloc[bounds_emg[0]:bounds_emg[1]]
            
            if emgs_filt_rect is not None:
                for muscle_name, muscle_data in emgs_filt_rect.iteritems():
                    if mct.emg_processing.is_valid(muscle_data): 
                        
                        params_dict.update(
                            mct.emg_processing.simple_params(
                                emg = muscle_data,
                                bounds = bounds_emg,
                                emg_max_act_filt_rect = emg_max_act_filt_rect.get(muscle_name),
                                emg_max_act_envelope = emg_max_act_envelope.get(muscle_name),
                                name = muscle_name,
                                sample_rate = config.spec.get('sample_rate_emg'))
                        )
                        params_dict.update({"cocontraction_elbow_all": coconctraction_elbow_all})
                        params_dict.update({"coconctraction_elbow_mvt" : coconctraction_elbow_mvt})
                        params_dict.update({"file_name" : movement_full_path})
                    params_emg.append(params_dict.copy())
                    print('bounds:', params_dict.get('bounds'))
            
            if not os.path.exists(movement_full_path):
                emgs = emgs.merge(
                    emgs_filt_rect_norm,
                    left_index = True,
                    right_index = True,
                    suffixes = ('', '_filt_rect_norm'))
                print("output length filt rec norm",len(emgs_filt_rect_norm))
                print("output length",len(emgs))
                movement_to_pickle = {
                    'bounds' : params_dict.get('bounds'),
                    'direction' : mov_kin.get('direction'),
                    'df_mov' : emgs
                    }
                with open(movement_full_path, 'wb') as f:
                    pickle.dump(movement_to_pickle, f, pickle.HIGHEST_PROTOCOL)
            
            if not config.cutoff_velocity_treshold_is_rel:
                # si compute avec treshold abs --> add bounds_abs to movements .pickle
                with open (movement_full_path, 'rb')  as f:
                    movement_to_modify = pickle.load(f)

                movement_to_modify.update({
                    'bounds_abs' : bounds_emg 
                })  

                with open (movement_full_path, 'wb') as f:
                    pickle.dump(movement_to_modify, f, pickle.HIGHEST_PROTOCOL)

            if config.take_overshoot_into_account:
                # si compute avec overshoot --> add bounds_abs to movements .pickle
                with open (movement_full_path, 'rb')  as f:
                    movement_to_modify = pickle.load(f)

                movement_to_modify.update({
                    'bounds_with_overshoot' : bounds_emg
                })
                with open (movement_full_path, 'wb') as f:
                    pickle.dump(movement_to_modify, f, pickle.HIGHEST_PROTOCOL)
    else:
        params_emg.append(params_dict.copy())
        print("Emg not computed - kinematic movement file not found")
    
   

    return params_emg # list of dict

if __name__ == "__main__":
    selector(sys.argv[1])    
