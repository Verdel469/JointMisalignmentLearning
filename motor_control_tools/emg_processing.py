"""This module contains tools for emg processing"""

# author : Simon Bastide
# mail : simon.bastide@outlook.com

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import integrate
import os
from scipy import stats


def filt_rect(emg, sample_rate, high_pass = 20, low_pass = 450, order = 4):
    """Rectify the signal and apply a band-pass butterworth filter

    This function is made to clean electromygraphie (emg) datas

    Args:
        emg (numpy.array): The emg signal
        sample_rate (int): Sample rate of the signal
        high_pass (int, optional): High frequency of the band-pass filter. Defaults to 20.
        low_pass (int, optional): Low frequency of the band-pass filter. Defaults to 450.
        order (int, optional): Order of the band-pass filter. Defaults to 4.

    Returns:
        numpy.array: The rectified and filtered signal
    """
    emg = emg[~np.isnan(emg)]
    high_pass = high_pass/(sample_rate/2) # cut-off frequency/Nyquist frequency
    low_pass = low_pass/(sample_rate/2)
    b, a = signal.butter(order, [high_pass, low_pass], btype = 'bandpass')
    emg_correctmean = emg - np.mean(emg)
    if 3 * max(len(a), len(b)) < len(emg_correctmean):
        emg_filt_rect = abs(signal.filtfilt(b, a, emg_correctmean))
    else:
        emg_filt_rect =  np.full(len(emg),0)
        # TODO: Warning signal here
        # Manage this case
    return emg_filt_rect

def envelope(emg, sample_rate, low_pass = 3, order = 5):
    """Apply a low pass filter to get the enveloppe of the emg signal

    Args:
        emg (numpy.array): The emg signal
        sample_rate (int): Sample rate of the signal
        low_pass (int, optional): Low frequency of the low-pass filter. Defaults to 10.
        order (int, optional): Order of the band-pass filter. Defaults to 4.

    Returns:
        numpy.array: filtered signal
    """
    # TODO: Change the name of the function --> filter. Option for params for emg enveloppe.
    low_pass = low_pass/(sample_rate/2)
    b, a = signal.butter(order, low_pass, btype = 'lowpass')
    if 3 * max(len(a), len(b)) < len(emg):
        emg_envelope = signal.filtfilt(b, a, emg)
    else:
        emg_envelope = np.full(100,0)
    return emg_envelope

def simple_params(emg, bounds, emg_max_act_filt_rect, emg_max_act_envelope, name, sample_rate):#, max_vel_loc):
    """Return simple parameters on emg signal. Rectified and filtered emg signal should be
    given in input (Use filt_rect() function before).

    parameters computed : 
        -maximum of the signal
        -median of the signal
        -maximum of the enveloppe
        -root mean square (rms) of the enveloppe

    Args:
        emg (numpy.array): The emg signal 
        name (string): emg's name. params will be return with the label 'name_param'
        sample_rate (int): Sample rate of the signal

    Returns:
        dict: dictionary containing computed parameters.
    """
    # Ligne avec lourd temps de calcul
    emg, phasic, tonic, updated_bounds = isolate_phasic_tonic_emg(emg, bounds, sample_rate) #calcul des paramétres sur le phasique
    # vel_loc donné avec un sample rate de 200 contre 2000 pour emg. De plus isolate_phasic_tonic_emg recoupe le signal emg. Donc
    # max_vel_loc doit être mis a jour pour corresponre aux nouveaux indices.

    # updated_max_vel_loc = max_vel_loc - int(updated_bounds[0]/10)

    if emg is not False: 

        # plt.plot(emg)
        # plt.plot(range(updated_bounds[0], updated_bounds[1]), emg[updated_bounds[0]: updated_bounds[1]])
        # plt.show()
        if updated_bounds is not False:
            emg_phasic_norm = (phasic/emg_max_act_filt_rect)*100
            start_burst, stop_burst, emg_env = burst_detection(
                emg = phasic,
                bounds = updated_bounds,
                max_env = emg_max_act_envelope,
                sample_rate = sample_rate
                )
            if stop_burst == start_burst:
                burst_duration = "undetected"
                burst_area = "undetected"
                burst_peak = "undetected"
            else:
                burst_duration = (stop_burst - start_burst)/sample_rate # sec
                burst_area = np.trapz(emg_phasic_norm[start_burst:stop_burst])
                burst_peak = np.max(emg_phasic_norm[start_burst:stop_burst])
            
            start_deact, stop_deact = deactivation_detection(
                envelope = emg_env,
                start_burst = start_burst)

            if start_deact == stop_deact:
                deact_duration = "undetected"
                deact_area = "undetected"
                deact_peak = "undetected"
            else:
                deact_duration = (stop_deact - start_deact)/sample_rate
                deact_area = np.trapz(emg_phasic_norm[start_deact:stop_deact])
                deact_peak = np.min(emg_phasic_norm[start_deact:stop_deact])
            # print(stop_burst, start_burst)
        
        emg_env = (envelope(emg, sample_rate, low_pass = 3)/emg_max_act_envelope)*100
        emg_norm = (emg/emg_max_act_filt_rect)*100

        emg_max = np.max(emg_norm[bounds[0]:bounds[1]])
        emg_med = np.median(emg_norm[bounds[0]:bounds[1]])

        if updated_bounds is not False:
            emg_max_env = np.max(emg_env[updated_bounds[0]:updated_bounds[1]])
            time_to_peak_env = np.argmax(emg_env[updated_bounds[0]:updated_bounds[1]])/sample_rate
        else:
            emg_max_env = np.max(emg_env[bounds[0]:bounds[1]])
            time_to_peak_env = np.argmax(emg_env[bounds[0]:bounds[1]])/sample_rate
        rms_env = np.sqrt(np.mean(emg_env[bounds[0]:bounds[1]]**2))
        rms_fr = np.sqrt(np.mean(emg_norm[bounds[0]:bounds[1]]**2))

        # if updated_max_vel_loc-updated_bounds[0] >  np.argmax(emg_env[updated_bounds[0]:updated_bounds[1]]):
        #     burst_type = "acceleration"
        # else:
        #     burst_type = "decceleration"
        if updated_bounds is not False:
            return ({'muscle' : name, 'max':emg_max, 'med':emg_med,\
                    'max_env':emg_max_env, 'rms_env': rms_env, 'rms_fr': rms_fr, 'burst_duration' : burst_duration,
                    'burst_area' : burst_area, 'burst_peak' : burst_peak, 'time_to_peak_env':time_to_peak_env,
                    'deact_duration' : deact_duration, 'deact_peak' : deact_peak,
                    'deact_area' : deact_area, 'bounds': updated_bounds})#, 'burst_type' : burst_type})
        else:
            return ({'muscle' : name, 'max':emg_max, 'med':emg_med,\
                    'max_env':emg_max_env, 'rms_env': rms_env, 'rms_fr': rms_fr, 'burst_duration' : None,
                    'burst_area' : None, 'burst_peak' : None, 'time_to_peak_env':time_to_peak_env,
                    'deact_duration' : None, 'deact_peak' : None,
                    'deact_area' : None, 'bounds': bounds})#, 'burst_type' : burst_type})
    else:
        return ({'muscle' : None, 'max':None, 'med':None,\
                'max_env':None, 'rms_env': None, 'rms_fr': None, 'burst_duration' : None,
                'burst_area' : None, 'burst_peak' : None, 'time_to_peak_env':None,
                'deact_duration' : None, 'deact_peak' : None,
                'deact_area' : None, 'bounds': None})


def burst_detection(emg, bounds, max_env, sample_rate, lim = 5):
    """[summary]

    Args:
        emg ([type]): emg signal
        bounds ([type]): movement bounds identified on kinematic data 
        max_env ([type]): Envelope max value for the subject ine the dataset. Used to normalise the signal
        sample_rate ([type]): sample rate of the signal
        lim (int, optional): Treshold used to detect the burst (% of max_env). Defaults to 5.
    """
    emg_env = (envelope(emg, sample_rate, low_pass = 3)/max_env)*100
    large_bounds = (bounds[0]-500,bounds[1]+500)
    loc_burst = emg_env[large_bounds[0]:large_bounds[1]]>5
    start = np.argmax(loc_burst) + large_bounds[0]
    stop =  np.argmin(loc_burst[np.argmax(loc_burst):])+large_bounds[0]+np.argmax(loc_burst)
    return (start, stop, emg_env)


def deactivation_detection(envelope, start_burst, limit = -5):

    deactivation = (envelope[0:start_burst]<limit)

    if deactivation[-1]: #is True
        end = len(deactivation)
    else:
        # on cherche premeir True en partant de la fin
        end = len(deactivation) - np.argmax(deactivation[::-1])

    # On cherche le premier False en partant de la fin identifié avant
    deactivation = deactivation[0:end]
    start = len(deactivation) - np.argmin(deactivation[::-1])

    return start, end


def cocontraction(envelope_emg1, envelope_emg2):
    """ Based on :
    K. Bowsher, D. Damiano, C. Vaughan, Joint torques and co-contraction during gait for normal
    and cerebral palsy children, in:Proceedings of the Second North American Congress on
    Biomechanics, 1992, pp. 319–320.
    And :
    Muscle co-activation around the knee in drop jumping using the
    co-contraction index E. Kellis ∗, F. Arabatzi, C. Papadopoulos

    Args:
        envelope_emg1 ([type]): [description]
        envelope_emg2 ([type]): [description]

    Returns:
        [type]: [description]
    """
    minimum = np.min([envelope_emg1, envelope_emg2], axis = 0)
    return np.trapz(minimum)/len(minimum)

def get_emgs_maximum_activation(subject_id, config, on = 'filt_rect', time_in = True, just_one_block = False, zscore = 1):
    """Run through a subject's data to obtain the rectified signal maximum activation of each
    muscle during the experiment.

    Args:
        subject_id (string): folder name of the subject. 'S1' for example
        config (module): Module config. obtained with an 'import config'

    Returns:
        dict: Dictionary with maximum activation for each muscle
    """
    max_subject = []
    for condition in config.spec.get('conditions'):
        for block in range(1,config.spec.get("block_per_condition").get(condition)+1):
            if not just_one_block:
                file = '_'.join([subject_id, condition, str(block)])
            else:
                file = '_'.join([subject_id, condition])
            emg_file_path = os.path.join(config.data_path,subject_id, file + "_emg.csv")

            if os.path.isfile(emg_file_path):
                trials_sep = pd.read_csv(os.path.join(config.data_path,subject_id,"trials_separators_" + file + ".csv")).astype(int)
                blocks_sep = pd.read_csv(os.path.join(config.data_path,subject_id,"blocks_separators_" + file + ".csv"))
                df_emg = pd.read_csv(emg_file_path)
                df_emg = df_emg.drop(columns = ['Force_sensor'])
                for mov_n, (start, stop) in enumerate(zip(trials_sep.sep_emg, trials_sep.sep_emg[1:])):
                    start = start + blocks_sep.sep_kin[0]
                    stop = stop + blocks_sep.sep_kin[0]
                    df_emg_mov = df_emg.iloc[start:stop]
                    df_emg_mov = df_emg_mov.dropna()
                    if not df_emg_mov.empty:
                        if on == 'filt_rect':
                            emg_filt_rect = abs(df_emg_mov.apply(filt_rect, args = [config.spec.get("sample_rate_emg")]))
                            # plt.plot(emg_filt_rect)
                            # plt.show(block = True)

                            max_subject.append(emg_filt_rect.max())
                        elif on == 'envelope':
                            emg_filt_rect = abs(df_emg_mov.apply(envelope, args = [config.spec.get("sample_rate_emg")]))
                            max_subject.append(emg_filt_rect.max())
    max_df = pd.DataFrame(max_subject)
    # plt.plot(max_df, marker = '*', c = 'b')
    
    cleaned_max_df = pd.DataFrame()
    # TODO Remplacer par un .apply()

    for colName, colData in max_df.iteritems():
        if zscore:
            outliers = np.abs(stats.zscore(colData, nan_policy='omit')) < zscore
            data_without_outliers = colData[outliers]
        else:
            data_without_outliers = colData
        cleaned_max_df = cleaned_max_df.assign(**{colName : data_without_outliers})
    
    # plt.plot(cleaned_max_df, marker = '*', c = 'r')
    # plt.show()

    if time_in:
        return dict(cleaned_max_df.drop(columns = 'time').max())
    else:
        return dict(cleaned_max_df.max())

def isolate_phasic_tonic_emg(emg, bounds, sample_rate = 2000):
    """Isolate tonic and phasic components from emg signal. Method from Gaveau et al. 2019
    A cross-species neural integration of gravity for motor optimisation :
    We used a well-known subtraction procedure that was proposed to isolate the phasic and tonic
    components of the full EMG signal (Buneo et al., 1994; d’Avella et al., 2006, 2008; Flanders
    and Herrmann, 1992; Flanders et al., 1994, 1996; Olesh et al., 2017; Prange et al., 472 2009b,
    2012; Russo et al., 2014). We computed the average values of the integrated EMG signals from
    1s to 0.5s before movement onset and from 0.5s to 1s after movement offset (Fig. 2 Suppl. Figure 2).
    We used these average values to compute the tonic component as a linear interpolation between them.
    Finally, we computed the phasic component by subtracting the tonic component from the full integrated EMG signal.

    Args:
        emg ([type]): Emg filtered and rectified
        bounds ([type]): [description]
        sample_rate (int, optional): [description]. Defaults to 2000.

    Returns:
        [type]: [description]
    """
    offset = int(sample_rate/2) #0.5sec

    bounds_for_av_start = (bounds[0]-2*offset, bounds[0]-offset)
    bounds_for_av_end = (bounds[1]+offset, bounds[1]+2*offset)

    if bounds_for_av_start[1]<50:
        print("Not enought points before the movement to compute tonic emg")
        return emg, False, False, False
    elif bounds[0]-2*offset <0:
        # pas assez de points avant le mvt mais au moins 250
        bounds_for_av_start = list(bounds_for_av_start)
        bounds_for_av_start[0] = 0 # on réduit la plage sur laquelle on calcul la mean
        bounds_for_av_start = np.array(bounds_for_av_start)

    if len(emg) - bounds_for_av_end[0]<50:
        print("Not enought points after the movement to compute tonic emg")
        return emg, False, False, False
    elif bounds[1]+2*offset>len(emg):
        # Au moins 250 pts pour calculer la moyenne sur la fin du signal
        bounds_for_av_end = list(bounds_for_av_end)
        bounds_for_av_end[1]=len(emg)
        bounds_for_av_end = np.array(bounds_for_av_end)

    emg_cut = emg[bounds_for_av_start[0]:bounds_for_av_end[1]]

    # Same method than Gaveau et al. 2019
    window_length = 10 #10ms at 2000Hz
    window_shift = 1
    integrated_signal = []
    # emg_filt_rect = emg_filt_rect[bounds[0]:bounds[1]]
    # TODO: Optimiser. Cette étape est trop longue
    for step in range(0, len(emg_cut)-window_length, window_shift):
        integrated_signal.append(integrate.trapz(emg_cut[step:step+window_length])/(window_length))
    
    
    # plt.plot(emg)
    # plt.plot(range(bounds_for_av_start[0]-10,bounds_for_av_end[1]), integrated_signal)
    # plt.show()
        
    integrated_signal = np.array(integrated_signal)

  
    new_bounds = bounds - bounds_for_av_start[0]
    bounds_for_av_end = bounds_for_av_end - bounds_for_av_start[0]
    bounds_for_av_start = bounds_for_av_start - bounds_for_av_start[0]

    av_start = np.mean(integrated_signal[bounds_for_av_start[0]:bounds_for_av_start[1]])
    av_start_idx = int(np.mean(bounds_for_av_start))
    av_end = np.mean(integrated_signal[bounds_for_av_end[0]:bounds_for_av_end[1]])
    av_end_idx = int(np.mean(bounds_for_av_end))
    
    # tonic = integrated_signal
    tonic = np.interp(range(0, len(integrated_signal)), [av_start_idx, av_end_idx], [av_start, av_end], )
    
    phasic = integrated_signal-tonic

    return emg, phasic, tonic, new_bounds

def hampel(vals_orig, k=10, t0=3):
    '''
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on either side of value)
    '''
    # from - https://stackoverflow.com/a/51731332/13360654
    # see also - https://towardsdatascience.com/outlier-detection-with-hampel-filter-85ddf523c73d 

    #Make copy so original not edited
    vals = vals_orig.copy()

    #Hampel Filter
    L = 1.4826
    rolling_median = vals.rolling(window=k, center=True).median()
    MAD = lambda x: np.median(np.abs(x - np.median(x)))
    rolling_MAD = vals.rolling(window=k, center=True).apply(MAD)
    threshold = t0 * L * rolling_MAD
    difference = np.abs(vals - rolling_median)

    '''
    Perhaps a condition should be added here in the case that the threshold value
    is 0.0; maybe do not mark as outlier. MAD may be 0.0 without the original values
    being equal. See differences between MAD vs SDV.
    '''

    outlier_idx = difference > threshold
    vals[outlier_idx] = np.nan
    return(vals)

def is_valid(emg):
    # False si une partie du signal est constant
    if np.sum(np.diff(emg)*2000 == 0) < 200:
        valid = True
    elif all(np.diff(emg)*2000 < 10e-6):
        valid = False
        print("Constant signal")
    elif any(np.diff(emg)*2000 > 0):
        valid = False
        print("A part of the signal is constant")

    return valid
