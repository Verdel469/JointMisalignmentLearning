# Tools for analyse parameters in dataframe

import pandas as pd
import numpy as np
import motor_control_tools as mct


def rename_conditions(df):
    df['condition'] = df['condition'].map({'TR' : '1G',
                                            'SE' : 'Baseline',
                                            '0G' : '0G',
                                            '-1G': '-1G',
                                            'RG' : '-1G',
                                            '1G' : '1G',
                                            'SE_out' : 'Baseline-out'
                                            })
    return df


def clean_df_acc_profiles(all_params_df):
    all_params_valid = all_params_df.query("valid == True") # Valid marche que avec elbow pour le moment. Condition sur le nombre de fois ou le profil d'acceleration passe par 0
    print("Valid movements regarding acceleration : {:.3f}%".format(len(all_params_valid)/len(all_params_df)*100))
    return all_params_valid

def clean_df_sub_mvt_amplitude(all_params_df, overshoot_limit = 5):
    overshoot_limit = (overshoot_limit*np.pi)/180
    all_params_valid = all_params_df.query("subA < @overshoot_limit") # 
    print("Valid movements regarding overshoot amplitude : {:.3f}%".format(len(all_params_valid)/len(all_params_df)*100))
    return all_params_valid

def clean_df(all_params_df, overshoot_limit = 5):
    df_acc_profiles = clean_df_acc_profiles(all_params_df)
    df_sub_mvt_amp = clean_df_sub_mvt_amplitude(all_params_df, overshoot_limit = overshoot_limit)
    # print(df_acc_profiles)
    # print(df_sub_mvt_amp)
    # df_sub_mvt_amp = df_sub_mvt_amp.reset_index()
    df_clean = df_acc_profiles.merge(
        df_sub_mvt_amp[['subject', 'condition', 'block', 'movement', 'direction']],
        on = ['subject', 'condition', 'block', 'movement', 'direction'],
        how = 'inner')
    
    df_clean, description = mct.stats.remove_outliers_per_condition(df_clean, columns=['A', 'MD'])
    print(description)
    return df_clean

def find_subject_without_enought_valid_movements(clean_params, expected_movement_per_subject = 600, min_percent_valid = 80):

    subj_outliers_list = []
    for subj in clean_params.reset_index().subject.unique():
        percent_valid = len(clean_params.query("subject == @subj"))/expected_movement_per_subject *100 #600 --> expected movement number
        print(subj +  ": {:.2f}".format(len(clean_params.query("subject == @subj"))/expected_movement_per_subject *100))
        if percent_valid < min_percent_valid: #moins de 80% de mouvements valides
            subj_outliers_list.append(subj)
    print("{} subject are not valid".format(len(subj_outliers_list)))
    print("{} need to be dropped".format(subj_outliers_list))

    return subj_outliers_list #outlier subject are not droped from clean_params

def prep_asym_df(clean_params, is_emg = False, return_clean_param_without_subj_outliers = False): #to run after clean_all_params()
    # Calcul des asymmétries et Organisation de la df
    df_up = clean_params.query("direction == 'up'").reset_index()
    df_down = clean_params.query("direction == 'down'").reset_index()

    if is_emg:
        df_up = df_up.groupby(['subject',"condition", "direction", 'block','muscle_group']).mean().droplevel(level = 'direction')
        df_down = df_down.groupby(['subject',"condition", "direction", 'block','muscle_group']).mean().droplevel(level = 'direction')
    else:
        df_up = df_up.groupby(['subject',"condition", "direction", 'block']).mean().droplevel(level = 'direction')
        df_down = df_down.groupby(['subject',"condition", "direction", 'block']).mean().droplevel(level = 'direction')
    if not is_emg:
        parameters_for_asym = ['PA','PV','tPA','tPV', 'rtPA', 'rtPV', 'rtPA/rtPV']
        df_up_asym = df_up[parameters_for_asym]
        df_down_asym = df_down[parameters_for_asym]
        df_asym = df_down_asym - df_up_asym
        directional_asym = df_asym.reindex(['1G', '0G', '-1G'], level = 'condition')
    df_up = df_up.reindex(['1G', '0G', '-1G'], level = 'condition')
    df_down = df_down.reindex(['1G', '0G', '-1G'], level = 'condition')

    # prep block name for fig
    if not is_emg:
        directional_asym = directional_asym.reset_index()
        directional_asym['condition_by_blocks'] = directional_asym['condition'] + ' B' + directional_asym['block'].apply(str)
    df_up = df_up.reset_index()
    df_up['condition_by_blocks'] = df_up['condition'] + ' B' + df_up['block'].apply(str)

    df_down = df_down.reset_index()
    df_down['condition_by_blocks'] = df_down['condition'] + ' B' + df_down['block'].apply(str)

    # Remove outliers
    if not is_emg:
        directional_asym_cleaned = directional_asym.set_index(keys = ['subject', 'condition', 'block'])
    df_up_cleaned = df_up.set_index(keys = ['subject', 'condition', 'block'])
    df_down_cleaned = df_down.set_index(keys = ['subject', 'condition', 'block'])
    clean_params = clean_params.set_index(keys = ['subject', 'condition', 'block'])


    if not is_emg:
        if return_clean_param_without_subj_outliers:
            return df_up_cleaned, df_down_cleaned, directional_asym_cleaned, clean_params
        else:
            return df_up_cleaned, df_down_cleaned, directional_asym_cleaned
    else:
        if return_clean_param_without_subj_outliers:
            return df_up_cleaned, df_down_cleaned, clean_params
        else:
            return df_up_cleaned, df_down_cleaned
        

def remove_subjects(df, subj_outliers = None):
    """[summary]

    Args:
        df ([type]): df containing at least an index named 'subject'
        subj_outliers (list, optional): list of outliers subject label. Defaults to None.

    Returns:
        [type]: [description]
    """
    df = df.set_index(['subject','condition'])
    if subj_outliers is None:
        print("No outliers to drop")
        return df
    else:
        print("Subject outliers are being droped : ")
        for subj_to_drop in subj_outliers:
            print(subj_to_drop)
            df = df.drop(subj_to_drop, level = 'subject')
    return df.reset_index()


def flex_or_ext_elb(row):
    if row['muscle'] in ['Brachial', 'Biceps']:
        row['Elb Group'] = 'Flex'
    elif row['muscle'] in ['Triceps_Lat', 'Triceps_Lng']:
        row['Elb Group'] = 'Ext'
    else:
        row['Elb Group'] = None
    return row

def flex_or_ext_sho(row):
    if row['muscle'] in ['Biceps', 'Deltoid_Ant']:
        row['Sho Group'] = 'Flex'
    elif row['muscle'] in ['Triceps_Lng', 'Deltoip_Post']:
        row['Sho Group'] = 'Ext'
    else:
        row['Sho Group'] = None
    return row


