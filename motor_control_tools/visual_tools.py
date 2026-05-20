"""Contain function used for visualize data"""

__author__ = "Simon Bastide"
__email__ = "simon.bastide@outlook.com"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Users/simon/OneDrive/sb_py_packages/manage_expe')
import manage_expe.basic_tools as bt
import pickle
import more_itertools
import re
import seaborn as sns


def set_latex_font(font_size = 10):
    plt.rc('font',  size = font_size)
    plt.rc('text', usetex = True)
    plt.rc('font', family='serif')

def clean_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def load_movement(file_name):
    with open(file_name, 'rb') as file:
        a = pickle.load(file)
    
    return a.get('bounds')[0], a.get('bounds')[1], a.get('df_mov')

def velocity_profiles_overview(df, vertical_offset):
    conditions = df.condition.unique()
    fig, axes = plt.subplots(1, len(conditions), dpi = 100, figsize = bt.cm2inch(15,18))
    for cond, ax in zip(conditions,axes):
        df_cond = df.query("condition == @cond") 
        nb_couple = int(len(list(df_cond.movement[::-1]))/2) #Nombre de couple de mouvements haut/bas
        # On part du principe que un mouvement vers le haut est toujours suivi d'un mouvement vers le bas
        couple = more_itertools.grouper(list(df_cond.movement),2) # iterateur sur les couples de mouvement haut bas : 1-->(0,1), 2 -->(2,3) etc..
        pt_resample = 100 # Normalisation des mouvements sur 100 points
        vertical_offset_increment = 0

        for up,down in couple:

            start, stop, a = load_movement(df_cond['file_name'].iloc[up])
        
            a.reset_index(inplace = True)
            a.start_theta = start
            a.stop_theta = stop

            start, stop, b = load_movement(df_cond['file_name'].iloc[down])
            b.reset_index(inplace = True)
            b.start_theta = start
            b.stop_theta = stop

            mvt_a_theta = bt.resample_by_interpolation(abs(a.vel[a.start_theta:a.stop_theta]), len(a.vel[a.start_theta:a.stop_theta]), pt_resample )
            mvt_b_theta = bt.resample_by_interpolation(abs(b.vel[b.start_theta:b.stop_theta]), len(b.vel[b.start_theta:b.stop_theta]), pt_resample )


            time = np.array(range(0,pt_resample))
            line_up = ax.plot(time, mvt_a_theta + vertical_offset_increment, c = 'tab:blue' , lw = 1) #c = cm.binary(cm_subsection)
            line_down = ax.plot(time, mvt_b_theta + vertical_offset_increment, c = 'tab:orange', lw = 1) #cm.Oranges(cm_subsection)

            vertical_offset_increment += vertical_offset/nb_couple
        

        ax.set_title(latex_compatible_string(cond))
        #ax.set_axis_off()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
    plt.legend(line_up+line_down, ['UP','DOWN'],loc='upper right', bbox_to_anchor=(2.2, 0.5,0,0))
    
            
    return fig, axes

def latex_compatible_string(my_string):
    return my_string.replace('_', '-')

def mean_profiles(datas, duration_mean, errors = 'SD'):
    time = np.linspace(0,duration_mean, len(datas))
    #datas_mean = np.mean(datas, axis = 1)
    datas_mean = np.mean(datas-datas[0,:], axis = 1)
    if errors == 'SD':
        errors_range = np.std(datas, axis = 1)
    elif errors == 'SEM':
        errors_range = np.std(datas, axis = 1)/datas.shape[1]
    else:
        print("errors input param must be 'SD' or 'SEM'")
    range_inf = datas_mean - errors_range
    range_sup = datas_mean + errors_range
    return time, datas_mean, range_inf, range_sup

def median_profiles(datas, duration_mean):
    time = np.linspace(0,duration_mean, len(datas))
    datas_med = np.median(datas, axis = 1)
    born_inf = np.percentile(datas, q = 25, axis = 1)
    born_sup = np.percentile(datas, q = 75, axis = 1)
    return time, datas_med, born_inf, born_sup

def compare_subjects(df, variables, sample_rate):
    """Plot variables. One axe by subject.
    
    Parameters
    ----------
    df : Pandas DataFrame
        df has to contain at least columns named : 
        - subject : str 
            subject id
        - condition : str
            condition id
        - movement : int
            numerous of the movement
        - file_name : str
            path used to load the movement file (.pickle file type)


    variables : string
        variable name. should be column name of DataFrame movement (loaded from .pickle file)
    """
    # TODO: Make it scalable with n subjects. Fix order of subjects. Legend conditions 
    # TODO: Split in different function : Compute means and std curves / plot  

    n_subj = len(df.subject.unique())
    n_col = 3
    
    
    fig, axes = plt.subplots(int(np.ceil(n_subj/n_col)),n_col, figsize = bt.cm2inch(25,18),\
         sharex=True, sharey=True)
   
    for (subj, df_subj), ax in zip(df.groupby('subject'), axes.flatten()):
        for cond, df_cond in df_subj.groupby('condition'):
            signal_to_plot = []
            durations = []
            for mvt_id, df_mvt in df_cond.groupby('movement'):

                start, stop, mvt = load_movement(df_mvt['file_name'].values[0])
               
                duration = (stop-start)/sample_rate
                durations.append(duration)
                signal_to_plot.append(bt.resample_by_interpolation(mvt[variables].iloc[start:stop],len(mvt[variables].iloc[start:stop]),1000))

            durations = np.array(durations).mean()
            t, averaged_signal, born_inf, born_sup = mean_profiles(np.array(signal_to_plot).T,durations)
            ax.plot(t,averaged_signal, label = cond)
            ax.fill_between(t,born_inf, born_sup, alpha = 0.5)
            ax.title.set_text(subj)
            clean_axes(ax)
    axes[0,0].legend()

    return fig, axes

def compare_conditions(df, variables, sample_rate):
    """Plot variables. One axe by subject.
    
    Parameters
    ----------
    df : Pandas DataFrame
        df has to contain at least columns named : 
        - subject : str 
            subject id
        - condition : str
            condition id
        - movement : int
            numerous of the movement
        - file_name : str
            path used to load the movement file (.pickle file type)
        - start : int
            index of movement start
        - stop : int
            index of movement stop

    variables : string
        variable name. should be column name of DataFrame movement (loaded from .pickle file)

    sample_rate : int
    
    one_plot : bool
        all in same axe or not
    """

    
    n_cond = len(df.condition.unique())
    n_col = 2
    
    fig, axes = plt.subplots(figsize = bt.cm2inch(25,18))
    all_signals_to_plot = {}
    all_born_inf = {}
    all_born_sup = {}
    all_times = {}

    signal_to_plot = []
    durations = []
    cond_int = 0
   
    for cond, df_cond in df.groupby('condition'):
        averaged_signals = []
        # find raw and column number for each subject (base 10 to base n_col)
        for subj, df_subj in df_cond.groupby('subject'):
        
            one_subj_signal = []
            one_subj_durations = []

            
            for mvt_id, df_mvt in df_subj.groupby('movement'):
                start, stop, mvt = load_movement(df_mvt['file_name'].values[0])

                one_subj_duration = (stop-start)/sample_rate
                one_subj_durations.append(one_subj_duration)
                one_subj_signal.append(\
                    bt.resample_by_interpolation(mvt[variables].iloc[start:stop],len(mvt[variables].iloc[start:stop]),1000))

            one_subj_durations = np.array(one_subj_durations).mean()
            _, one_subj_averaged_signal, _, _ = mean_profiles(np.array(one_subj_signal).T,one_subj_durations)
            averaged_signals.append(one_subj_averaged_signal)
            
        
        durations = np.array(one_subj_durations).mean()
        t, signal_to_plot, born_inf, born_sup = mean_profiles(np.array(averaged_signals).T, durations)

        
        all_signals_to_plot.update({cond : signal_to_plot})
        all_born_inf.update({cond : born_inf})
        all_born_sup.update({cond : born_sup})
        all_times.update({cond : t})
          
            
    
    for cond in df.condition.unique():
        axes.plot(all_times.get(cond), all_signals_to_plot.get(cond), label = cond)
        axes.fill_between(all_times.get(cond), all_born_inf.get(cond), all_born_sup.get(cond), alpha = 0.5)
    axes.legend()
    clean_axes(axes)

    return fig, axes


def one_subject():
    pass

def one_movement(movement, sample_rate, start = 0, stop = None, ylabels = None):

    """Plot variables. One axe by subject.
    
    Parameters
    ----------
    movement : Pandas DataFrame
        contain movement datas. Each column will be plot and should be correspond to a variable

    sample_rate : int

    start : int
        index determining the effective movement start
    
    stop : int
        index determining the effective movement stop

    ylabels : list
        list of string to be used as ylabels. length of ylabels should be equal to the number of columns in movement DataFrame
    """

    if stop is None:
        stop = len(movement)

    duration = (stop-start)/sample_rate
    time_mvt = np.linspace(start/sample_rate, (start/sample_rate)+duration, len(movement[start:stop]))
    time_tot = np.linspace(0, len(movement)/sample_rate, len(movement))

    fig, axes = plt.subplots(len(movement.columns),1)
    
    if ylabels is not None:
        for ylabel, ax in zip(ylabels, axes):
            ax.set_ylabel(ylabel, rotation = 0)
    
    for i, (ax, (col_name, var)) in enumerate(zip(axes, movement.items())):
        ax.plot(time_tot, var, c = 'gray')
        ax.plot(time_mvt, var[start:stop], c = 'k', lw = 3)
        # ax.set_xlim([0, 1.3])
        ax.grid(True, lw = 0.5, ls = ":", c = 'k')

        if i != len(movement.columns)-1:
            ax.set_xticklabels([])

    # TODO : Manage limits and share axes limits 

    # emg_lim = np.max([np.max(bic),np.max(bra), np.max(tlat), np.max(tlng)])
    
    plt.subplots_adjust(hspace = 0)
    # fup.align_ylabels(axup[:])
    # axup[0].set_ylim([-40,40])
    # axup[1].set_ylim([-20,260])
    # axup[2].set_ylim([-1400,1600])
    # axup[3].set_ylim([-5/100*emg_lim,emg_lim + 5])
    # axup[4].set_ylim([-5/100*emg_lim,emg_lim + 5])
    # axup[5].set_ylim([-5/100*emg_lim,emg_lim + 5])
    # axup[6].set_ylim([-5/100*emg_lim,emg_lim + 5])
    # axup[-1].set_xlabel('Time (sec)', labelpad = 5)

    return fig, axes


def plot_adaptation(data, variable, title, ylabel = None, hline = 0.5):
    # TODO : Adapter les limites et axv axh line au data qui sont plot. Ne marche que pour 50 trials pour le moment.
    if ylabel is None:
        ylabel = variable
    fig = plt.figure(figsize = bt.cm2inch(20,7), dpi = 300)
    ax1 = fig.add_subplot(1,1,1)
    plt.title(title)
    sns.lineplot(
        x="trials",
        y=variable,
        color = 'tab:grey',
        data=data.reset_index(),
        ax = ax1,
        lw = 0.5
        )
    sns.scatterplot(
        x="trials",
        y=variable,
        color = 'tab:grey',
        data=data.reset_index(),
        s=2,
        alpha = 0.5,
        ax = ax1
    )
    sns.scatterplot(
        x="trials",
        y=variable,
        color = 'tab:grey',
        data=data.groupby("trials").mean().reset_index(),
        s=10,
        alpha = 1,
        ax = ax1
    )
    ax1.axvline(x = 24.5, lw = 1, color = 'k', ls = '--')
    if not hline is None:
        ax1.axhline(hline, lw = 0.2, c = 'k')
    ax1.set_xlim(left = 0.8, right = 50.2)
    ax1.text(25/2, ax1.get_ylim()[1], "Block 1")
    ax1.text(25+25/2, ax1.get_ylim()[1], "Block 2")
    ax1.set_ylabel(ylabel)
    clean_axes(ax1)

    return fig, ax1
