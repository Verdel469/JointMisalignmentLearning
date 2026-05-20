import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from manage_expe import basic_tools as bt
from scipy import signal

def cutting(position, velocity, sample_rate, treshold = 1):
    sep_index = automatic_cutting(position, velocity, sample_rate, treshold)
    reponse = input("Separation ok ? (Y/N)   : ")
    if reponse.lower() == "y" or reponse == "":
        plt.close()
    else:
        delete_n_first_movs = input("Delete n movements at the start  ? (int)  : ")
        sep_index = sep_index[int(delete_n_first_movs):].copy()
        delete_n_last_movs = input("Delete n movements at the end ? (int)  : ")
        if not int(delete_n_last_movs) == 0:
            sep_index = sep_index[:-int(delete_n_last_movs)].copy()
        reponse = input("Manual separation ? (Y/N)  : ")
        if reponse.lower() == "y" or reponse == "":
            sep_index = manual_cutting(position, velocity, sample_rate)
        plt.close()
    return sep_index

def automatic_cutting(position, velocity, sample_rate, treshold = 1):
    peaks = signal.find_peaks(velocity,height = treshold, distance = 2*sample_rate)
    sep_index = np.zeros(peaks[0].shape)
    if len(sep_index) != 0:
        for n,peak in enumerate(peaks[0]):
            if n+1 >= len(peaks[0]):
                break
            sep_index[n] = (peak+peaks[0][n+1])/2
    else:
        sep_index = np.zeros((1,1))
    sep_index[-1] = len(velocity)
    sep_index = np.insert(sep_index,0,0)

    print("Number of peaks detected : {}".format(len(peaks[0])))
    print("Number of movement identified : {}".format(len(sep_index)-1))
    _, axes = plt.subplots(2,1,figsize = bt.cm2inch(20,20))
    axes[0].plot(velocity, label = 'Velocity')
    axes[0].plot(peaks[0],velocity[peaks[0]], '*', label = 'Peaks')
    axes[1].plot(position, label = 'Position')
    for id in sep_index:
        for ax in axes:
            ax.axvline(id, c = 'k')
    axes[0].axvline(id, c = 'k', label = 'Index for cutting')
    axes[0].axhline(treshold, c= 'r', label = 'Detection treshold')
    axes[0].legend()
    plt.show()
    #block = False
    return sep_index

def manual_cutting(position, velocity, sample_rate, confirmation = True):
    reponse = 'n'
    while reponse.lower() == 'n':
        plt.close()
        _, axes = plt.subplots(2,1,figsize = bt.cm2inch(20,20))
        axes[0].plot(velocity, label = 'Velocity', lw = 1)
        axes[1].plot(position, label = 'Position', lw = 1)
        axes[0].legend()
        plt.show(block = False)
        click_coordinates = (plt.ginput(n = -1, timeout = 0))
        sep_index = np.asarray([int(sep[0]) for sep in click_coordinates])
        for id in sep_index:
            for ax in axes:
                ax.axvline(id, c = 'k', lw = 1)
        print("Number of partitions : {}".format(len(sep_index)-1))
        if confirmation :
            plt.draw()
            reponse = input("Manual separation ok ? (Y/N)   : ")
        else:
            reponse = 'y'
    
    return sep_index
