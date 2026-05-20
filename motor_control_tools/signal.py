import numpy as np
import pandas as pd
import scipy as sp


# TODO: Changer nom - conflits avec filter keyword python
def filter(data, sample_rate, low_pass = 10, order = 4):
    """Applique un filtre basique passe bas (butter, ordre 4) pour lisser le signal"""

    # TODO: Ameliorer la gestion des NaN. Pour le moment les NaN A la fin de l'enregistrement sont supprimé et les autres sont remplacés par 0.
    try:
        data = np.array(data)
        low_pass = low_pass/(sample_rate/2)
        b, a = sp.signal.butter(order, low_pass, btype = 'lowpass')
        data_filtered = np.zeros(data.shape)
        if len(data.shape) > 1:
            #Supression des nan en fin de signal 
            while np.isnan(data[-1,:]).any():
                data = data[:-1,:]
                data_filtered = data_filtered[:-1,:]
            data = np.nan_to_num(data)
            for col in range(data.shape[1]):
                data_filtered[:,col] = sp.signal.filtfilt(b, a, data[:,col])
        else:
            while np.isnan(data[-1]):
                data = data[:-1]
                data_filtered = data_filtered[:-1]
            data = np.nan_to_num(data)
            
            data_filtered = sp.signal.filtfilt(b, a, data)
        if np.isnan(data_filtered).any():
            print("Output contain NaN")
            
    except ValueError:
        print("Value Error, output will be full of 0")
        data_filtered =  np.full(1000,0)
    except IndexError:
        print("Data possibly full of nan, output will be full of 0")
        data_filtered =  np.full(1000,0)
    return data_filtered


# DISCLAIMER: This function is copied from https://github.com/nwhitehead/swmixer/blob/master/swmixer.py, 
#             which was released under LGPL. 
def resample_by_interpolation(signal, fixed_outlen = True, outlen = 1000, input_fs = 100, output_fs = 100):

    if fixed_outlen:
        n = outlen
    else:
        scale = output_fs / input_fs
        # calculate new length of sample
        n = round(len(signal) * scale)

    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    if len(signal.shape) > 1:
        for i in range(signal.shape[1]):
            resampled_signal = np.interp(
                np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
                np.linspace(0.0, 1.0, len(signal[:,i]), endpoint=False),  # known positions
                signal[:,i],  # known data points
            )
            if i == 0:
                resampled_signal_all = resampled_signal
            else:
                resampled_signal_all = np.column_stack((resampled_signal_all,resampled_signal))
    else:
        resampled_signal_all = np.interp(
            np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
            np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
            signal,  # known data points
        )
    return resampled_signal_all

def resample_non_uniform(x, y, new_sample_rate):
    """[summary]
    Fonction inspired from https://stackoverflow.com/a/20889651/13360654
    Args:
        x (numpy.array): non-uniform time (in s)
        y (numpy array): signal
    """
    f = sp.interpolate.interp1d(x, y)
    sample_rate = len(x)/x[-1]
    new_sample_rate = 200
    num = int(np.ceil((len(y)*new_sample_rate)/sample_rate))
    xx = np.linspace(x[0], x[-1], num)
    
    return f(xx)

def diff_keep_length(signal, sample_rate, spec_t = False):
    """Derive en conservant la longueur du signal. La dérnière valeur est doublée"""

    if spec_t:
        diff           = np.diff(signal)
        diff_len       = np.append(diff,diff[-1])
        signal_dot_len = diff_len/sample_rate
    else:
        signal_dot     = np.diff(signal)*sample_rate
        signal_dot_len = np.append(signal_dot,signal_dot[-1])
    return signal_dot_len

def diff_keep_length_duration(signal, duration):
    """Derive en conservant la longueur du signal. La dérnière valeur est doublée"""

    sample_rate    = len(signal)/duration
    dt             = 1/sample_rate
    diff           = np.diff(signal)
    diff_len       = np.append(diff,diff[-1])
    signal_dot_len = diff_len/dt

    return signal_dot_len

def cross_0_times(signal):
    """Compute the number of time that the signal cross the zero line

    Args:
        signal (array)

    Returns:
        int
    """
    return ((signal[:-1]*signal[1:]) < 0).sum()

def reversal_points(signal, distance_limit = 1):
    """[summary]

    Args:
        signal ([type]): [description]
        distance_limit (int, optional): limit distance between reversal point in number of frame. Defaults to 1.

    Returns:
        [type]: [description]
    """
    points = np.where((signal[:-1]*signal[1:]) < 0)[0]
    for pt1, pt2 in zip(points, points[1:]):
        if pt2-pt1 < distance_limit:
            points = points[points != pt2]
    return points
