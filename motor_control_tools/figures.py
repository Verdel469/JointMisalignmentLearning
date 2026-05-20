import seaborn as sns
import motor_control_tools as mct
from motor_control_tools.visual_tools import mean_profiles
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from manage_expe import basic_tools as bt
# TODO: Résoude le probléme de dépendance au fichier de config
# import config
import pickle
import motor_control_tools.emg_processing as emgPrc



def muscle_plat_verif(data, variable, title, ax, ylabel = "", xlabel = "", legend = True, labelbottom = False, **kwargs):
    g = sns.barplot(
        x = 'condition',
        y = variable,
        hue = 'muscle_group',
        data = data.reset_index(),
        ax= ax,
        hue_order = ['Flexors', 'Extensors'],
        order = ['1G', '0G', '-1G'],
        color = 'grey',
        edgecolor = ".2",
    )
    mct.visual_tools.clean_axes(ax)
    if legend == True:
        ax.legend(
            fontsize = font_size,
            bbox_to_anchor=(-.9, 1.2),
            loc = 'upper left',
            frameon = False,
            title = r'RMS ($\%$ of max)'
            )
    else:
        ax.get_legend().remove()

    ax.set_title(title, pad = -5)
    ax.grid(axis = 'y', which = 'major', lw = 0.2)
    ax.set_ylabel(ylabel, rotation = 'horizontal', ha='right')
    ax.set_xlabel(xlabel)
    ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=labelbottom) # labels along the bottom edge are off
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end, 2))

    for key, value in kwargs.items():
        if key == "rep_mean_rms":
            ax.axhline(y = value.mean(), lw = 0.7, ls = '--', c = 'k')
        if key == "ident_mean_rms":
            ax.axhline(y = value.mean(), lw = 0.7, ls = ':', c = 'k')
    #ax.axhline(y = rep_mean_rms.Flexors, lw = 0.7, ls = '--', c = 'lightgrey')

    return ax

def tau_plat_verif(data, variable, title, ax, ylabel = "", xlabel = "", legend = True, labelbottom = False):
    g = sns.barplot(
        x = 'condition',
        y = variable,
        data = data.reset_index(),
        ax= ax,
        order = ['1G', '0G', '-1G'],
        color = 'grey',
        edgecolor = ".2",
    )
    mct.visual_tools.clean_axes(ax)

    ax.legend()

    if legend == True:
        ax.legend(
            fontsize = 9,
            bbox_to_anchor=(-.9, 1.2),
            loc = 'upper left',
            frameon = False,
            title = r'RMSE (N.m)'
            )
    else:
        ax.get_legend().remove()
    ax.set_title(title, pad = -2)
    ax.grid(axis = 'y', which = 'major', lw = 0.2)
    ax.set_ylabel(ylabel, rotation = 'horizontal', ha='right')
    ax.set_xlabel(xlabel)
    ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=labelbottom) # labels along the bottom edge are off
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end, 0.5))
    
    return ax

def all_trajectories(dict_from_qual, dict_from_rob, dpi = 300, figsize = (12,17)):
    """

    Args:
        dict_from_qual ([type]): outpout of aggregate_all_trajectories_qual
        dict_from_rob ([type]): output of aggregate_all_trajectories_rob
        dpi (int, optional): [description]. Defaults to 300.
        figsize (tuple, optional): [description]. Defaults to (12,17).
    """

    fig, axes = plt.subplots(5,3, figsize = bt.cm2inch(figsize), sharex=True, sharey='row', frameon = False, dpi = dpi)

    axes[0,0].set_ylabel(r'Pos. ($rad$)')
    axes[1,0].set_ylabel(r'Vel. ($rad.s^{-1}$)')
    axes[2,0].set_ylabel(r'Acc. ($rad.s^{-2}$)')
    axes[3,0].set_ylabel(r'$\tau_{i}$ ($N.m$)')
    axes[4,0].set_ylabel(r'EMG ($\%$ of max)')

    desired_cond_order = {
        '1G':0,
        '0G':1,
        '-1G':2,
    }

    # titres dans le bon ordre 
    inv_map = {v: k for k, v in desired_cond_order.items()}
    axes[0,0].set_title('{}'.format(inv_map.get(0)))
    axes[0,1].set_title('{}'.format(inv_map.get(1)))
    axes[0,2].set_title('{}'.format(inv_map.get(2)))
    # print(dict_from_qual)
    for cond in desired_cond_order.keys():
        pos = dict_from_qual.get("pos").get(cond)
        vel = dict_from_qual.get("vel").get(cond)
        acc = dict_from_qual.get("acc").get(cond)
        bic = dict_from_qual.get("bic").get(cond)
        bra = dict_from_qual.get("bra").get(cond)
        tlat = dict_from_qual.get("tlat").get(cond)
        tlng = dict_from_qual.get("tlng").get(cond)
        tau = dict_from_rob.get("tau").get(cond)
        tau_theo = dict_from_rob.get("tau_theo").get(cond)

        points_mvt = 100
        points_deb_fin = 50

        # sortir les temps dans les fonctions aggregate
        t, pos_av, pos_born_inf, pos_born_sup = mean_profiles(pos.get('mvt'), points_mvt)
        t_deb, pos_deb_av, _,_, = mean_profiles(pos.get('deb'), points_deb_fin)
        t_fin, pos_fin_av, _,_, = mean_profiles(pos.get('fin'), points_deb_fin)

        _, vel_av, vel_born_inf, vel_born_sup = mean_profiles(vel.get('mvt'), points_mvt)
        _, vel_deb_av, _,_, = mean_profiles(vel.get('deb'), points_deb_fin)
        _, vel_fin_av, _,_, = mean_profiles(vel.get('fin'), points_deb_fin)

        _, acc_av, acc_born_inf, acc_born_sup = mean_profiles(acc.get('mvt'), points_mvt)
        _, acc_deb_av, _,_, = mean_profiles(acc.get('deb'), points_deb_fin)
        _, acc_fin_av, _,_, = mean_profiles(acc.get('fin'), points_deb_fin)

        t_emg, bic_av, bic_born_inf, bic_born_sup = mean_profiles(bic.get('mvt'), points_mvt)
        t_emg_deb, bic_deb_av, _,_, = mean_profiles(bic.get('deb'), points_deb_fin)
        t_emg_fin, bic_fin_av, _,_, = mean_profiles(bic.get('fin'), points_deb_fin)

        _, bra_av, bra_born_inf, bra_born_sup = mean_profiles(bra.get('mvt'), points_mvt)
        _, bra_deb_av, _,_, = mean_profiles(bra.get('deb'), points_deb_fin)
        _, bra_fin_av, _,_, = mean_profiles(bra.get('fin'), points_deb_fin)

        flex_av = np.mean(np.array([bic_av, bra_av]), axis=0)
        flex_deb_av = np.mean(np.array([bic_deb_av, bra_deb_av]), axis=0)
        flex_fin_av = np.mean(np.array([bic_fin_av, bra_fin_av]), axis=0)

        _, tlat_av, tlat_born_inf, tlat_born_sup = mean_profiles(tlat.get('mvt'), points_mvt)
        _, tlat_deb_av, _,_, = mean_profiles(tlat.get('deb'), points_deb_fin)
        _, tlat_fin_av, _,_, = mean_profiles(tlat.get('fin'), points_deb_fin)

        _, tlng_av, tlng_born_inf, tlng_born_sup = mean_profiles(tlng.get('mvt'), points_mvt)
        _, tlng_deb_av, _,_, = mean_profiles(tlng.get('deb'), points_deb_fin)
        _, tlng_fin_av, _,_, = mean_profiles(tlng.get('fin'), points_deb_fin)

        ext_av = np.mean(np.array([tlat_av, tlng_av]), axis=0)
        ext_deb_av = np.mean(np.array([tlat_deb_av, tlng_deb_av]), axis=0)
        ext_fin_av = np.mean(np.array([tlat_fin_av, tlng_fin_av]), axis=0)

        _, tau_av, tau_born_inf, tau_born_sup = mean_profiles(tau.get('mvt'), points_mvt)
        _, tau_deb_av, _,_, = mean_profiles(tau.get('deb'), points_deb_fin)
        _, tau_fin_av, _,_, = mean_profiles(tau.get('fin'), points_deb_fin)

        _, tau_theo_av, tau_theo_born_inf, tau_theo_born_sup = mean_profiles(tau_theo.get('mvt'), points_mvt)
        _, tau_theo_deb_av, _,_, = mean_profiles(tau_theo.get('deb'), points_deb_fin)
        _, tau_theo_fin_av, _,_, = mean_profiles(tau_theo.get('fin'), points_deb_fin)

        t = t+50
        t_fin = t_fin+t[-1]

        t_emg = t_emg+50
        t_emg_fin = t_emg_fin+t_emg[-1]

        axes[0, desired_cond_order.get(cond)].plot(t_deb, pos_deb_av, c = 'darkgrey')
        axes[0, desired_cond_order.get(cond)].plot(t_fin, pos_fin_av, c= 'darkgrey')
        axes[0, desired_cond_order.get(cond)].fill_between(t, pos_born_inf, pos_born_sup, color = 'lightsteelblue')
        axes[0, desired_cond_order.get(cond)].plot(t, pos_av, c = 'k')

        
        axes[1, desired_cond_order.get(cond)].plot(t_deb, vel_deb_av, c = 'darkgrey')
        axes[1, desired_cond_order.get(cond)].plot(t_fin, vel_fin_av, c= 'darkgrey')
        axes[1, desired_cond_order.get(cond)].fill_between(t, vel_born_inf, vel_born_sup, color = 'lightsteelblue')
        axes[1, desired_cond_order.get(cond)].plot(t[:-10], vel_av[:-10], c = 'k')

        
        axes[2, desired_cond_order.get(cond)].plot(t_deb, acc_deb_av, c = 'darkgrey')
        axes[2, desired_cond_order.get(cond)].plot(t_fin, acc_fin_av, c= 'darkgrey')
        axes[2, desired_cond_order.get(cond)].fill_between(t, acc_born_inf, acc_born_sup, color = 'lightsteelblue')
        axes[2, desired_cond_order.get(cond)].plot(t[:-10], acc_av[:-10], c = 'k')

        axes[3, desired_cond_order.get(cond)].plot(t_deb, tau_deb_av, c = 'darkgrey')
        axes[3, desired_cond_order.get(cond)].plot(t_fin, tau_fin_av, c= 'darkgrey')
        # axes[3, desired_cond_order.get(cond)].fill_between(t, tau_born_inf, tau_born_sup, color = 'lightsteelblue')
        axes[3, desired_cond_order.get(cond)].plot(t, tau_av, c = 'k')

        axes[3, desired_cond_order.get(cond)].plot(t_deb, tau_theo_deb_av, c = 'peachpuff')
        axes[3, desired_cond_order.get(cond)].plot(t_fin, tau_theo_fin_av, c= 'peachpuff')
        # axes[3, desired_cond_order.get(cond)].fill_between(t, tau_theo_born_inf, tau_theo_born_sup, color = 'lightcoral')
        axes[3, desired_cond_order.get(cond)].plot(t, tau_theo_av, c = 'tab:red')

        axes[4, desired_cond_order.get(cond)].plot(t_emg_deb, flex_deb_av, c = 'lightgreen')
        axes[4, desired_cond_order.get(cond)].plot(t_emg_fin, flex_fin_av, c= 'lightgreen')
        # axes[4, desired_cond_order.get(cond)].fill_between(t, tau_born_inf, tau_born_sup, color = 'lightsteelblue')
        axes[4, desired_cond_order.get(cond)].plot(t_emg, flex_av, c = 'tab:green')

        axes[4, desired_cond_order.get(cond)].plot(t_emg_deb, -ext_deb_av, c = 'peachpuff')
        axes[4, desired_cond_order.get(cond)].plot(t_emg_fin, -ext_fin_av, c= 'peachpuff')
        # axes[4, desired_cond_order.get(cond)].fill_between(t, tau_theo_born_inf, tau_theo_born_sup, color = 'lightcoral')
        axes[4, desired_cond_order.get(cond)].plot(t_emg, -ext_av, c = 'tab:red')

   
        axes[4, desired_cond_order.get(cond)].set_xticks([50, 150], minor=False)
        axes[4, desired_cond_order.get(cond)].set_xticklabels([0, 100], minor=False)


    fig.text(0.53, 0.02, r'$\%$ of movement duration', ha='center')
    
    for axs in axes:
        for ax in axs:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    fig.align_ylabels(axes[:, 0])

    # plt.show()
    return fig


def aggregate_all_trajectories_rob(df, sample_rate_rob = 200, unit = 'rad'):
    # pass df.query(direction == ?)
    # one traj by subject for pos, vel, acc, tau_i
    print(df)
    # based on rob data
    pos_all = {}
    vel_all = {}
    acc_all = {}
    fz_all = {}
    tau_all = {}
    tau_theo_all = {}

    pre_post_time = 0.5 # tps avant et aprés mouvement effectif
    pre_post_sample = int(pre_post_time*sample_rate_rob)
    print(pre_post_sample)


    
    for cond in df.condition.unique():
        df_ = df.query("condition == @cond")
        pos_cond = []
        vel_cond = []
        acc_cond = []
        tau_cond = []
        fz_cond = []
        tau_theo_cond = []
        pos_deb_cond = []
        pos_fin_cond = []
        vel_deb_cond = []
        vel_fin_cond = []
        acc_deb_cond = []
        acc_fin_cond = []
        tau_deb_cond = []
        tau_fin_cond = []
        fz_deb_cond = []
        fz_fin_cond = []
        tau_theo_deb_cond = []
        tau_theo_fin_cond = []
        
        for subj, df_subj in df_.groupby('subject'):
            print(subj)
            
            pos = []
            vel = []
            acc = []
            tau = []
            fz = []
            tau_theo = []
            pos_deb = []
            pos_fin = []
            vel_deb = []
            vel_fin = []
            acc_deb = []
            acc_fin = []
            tau_deb = []
            tau_fin = []
            tau_theo_deb = []
            tau_theo_fin = []

            fz_deb = []
            fz_fin = []

            # fz_theo_deb = []
            # fz_theo_fin = []
            
            durations = []
            for mvt_id, movement_row in df_subj.groupby('movement'):
                with open(movement_row['file_name'].values[0], 'rb') as data_file:
                    data_movement = pickle.load(data_file)
                    df_mov = data_movement.get('df_mov')

                start = data_movement.get('bounds')[0]
                end = data_movement.get('bounds')[1]
                
                # print(df_mov)
                start_prev = np.clip(start -pre_post_sample,0, start) 
                end_foll = np.clip(end +pre_post_sample,0, len(df_mov)-1)
                
                if unit == 'deg':
                    df_mov['pos'] = df_mov['pos']*180/np.pi
                    df_mov['vel'] = df_mov['vel']*180/np.pi
                    df_mov['acc'] = df_mov['acc']*180/np.pi
                        
                durations.append(movement_row['MD'].values[0])
                # Position
                pos.append(mct.signal.resample_by_interpolation(df_mov['pos'].iloc[start:end],len(df_mov['pos'].iloc[start:end]),1000))
                pos_deb.append(mct.signal.resample_by_interpolation(df_mov['pos'].iloc[start_prev:start],len(df_mov['pos'].iloc[start_prev:start]),pre_post_sample))
                pos_fin.append(mct.signal.resample_by_interpolation(df_mov['pos'].iloc[end:end_foll],len(df_mov['pos'].iloc[end:end_foll]),pre_post_sample))
                
                # Velocity
                vel.append(mct.signal.resample_by_interpolation(df_mov['vel'].iloc[start:end],len(df_mov['vel'].iloc[start:end]),1000))
                vel_deb.append(mct.signal.resample_by_interpolation(df_mov['vel'].iloc[start_prev:start],len(df_mov['vel'].iloc[start_prev:start]),pre_post_sample))
                vel_fin.append(mct.signal.resample_by_interpolation(df_mov['vel'].iloc[end:end_foll],len(df_mov['vel'].iloc[end:end_foll]),pre_post_sample))
                
                #Acceleration
                acc.append(mct.signal.resample_by_interpolation(df_mov['acc'].iloc[start:end],len(df_mov['acc'].iloc[start:end]),1000))
                acc_deb.append(mct.signal.resample_by_interpolation(df_mov['acc'].iloc[start_prev:start],len(df_mov['acc'].iloc[start_prev:start]),pre_post_sample))
                acc_fin.append(mct.signal.resample_by_interpolation(df_mov['acc'].iloc[end:end_foll],len(df_mov['acc'].iloc[end:end_foll]),pre_post_sample))

                # #fz
                fz.append(mct.signal.resample_by_interpolation(df_mov['Fz'].iloc[start:end],len(df_mov['Fz'].iloc[start:end]),1000))
                fz_deb.append(mct.signal.resample_by_interpolation(df_mov['Fz'].iloc[start_prev:start],len(df_mov['Fz'].iloc[start_prev:start]),pre_post_sample))
                fz_fin.append(mct.signal.resample_by_interpolation(df_mov['Fz'].iloc[end:end_foll],len(df_mov['Fz'].iloc[end:end_foll]),pre_post_sample))

                # #fz_theo
                # fz_theo.append(mct.signal.resample_by_interpolation(df_mov['Fz_theo'].iloc[start:end],len(df_mov['Fz_theo'].iloc[start:end]),1000))
                # fz_theo_deb.append(mct.signal.resample_by_interpolation(df_mov['Fz_theo'].iloc[start_prev:start],len(df_mov['Fz_theo'].iloc[start_prev:start]),pre_post_sample))
                # fz_theo_fin.append(mct.signal.resample_by_interpolation(df_mov['Fz_theo'].iloc[end:end_foll],len(df_mov['Fz_theo'].iloc[end:end_foll]),pre_post_sample))

                #tau_i
                tau.append(mct.signal.resample_by_interpolation(df_mov['tau_i'].iloc[start:end],len(df_mov['tau_i'].iloc[start:end]),1000))
                tau_deb.append(mct.signal.resample_by_interpolation(df_mov['tau_i'].iloc[start_prev:start],len(df_mov['tau_i'].iloc[start_prev:start]),pre_post_sample))
                tau_fin.append(mct.signal.resample_by_interpolation(df_mov['tau_i'].iloc[end:end_foll],len(df_mov['tau_i'].iloc[end:end_foll]),pre_post_sample))

                #tau_i_theorique
                tau_theo.append(mct.signal.resample_by_interpolation(df_mov['tau_i_theo'].iloc[start:end],len(df_mov['tau_i_theo'].iloc[start:end]),1000))
                tau_theo_deb.append(mct.signal.resample_by_interpolation(df_mov['tau_i_theo'].iloc[start_prev:start],len(df_mov['tau_i_theo'].iloc[start_prev:start]),pre_post_sample))
                tau_theo_fin.append(mct.signal.resample_by_interpolation(df_mov['tau_i_theo'].iloc[end:end_foll],len(df_mov['tau_i_theo'].iloc[end:end_foll]),pre_post_sample))
                
                
            

            durations = np.array(durations).mean()
            t, pos_av, pos_born_inf, pos_born_sup = mean_profiles(np.array(pos).T,durations)
            t_deb, pos_deb_av, pos_deb_born_inf, pos_deb_born_sup = mean_profiles(np.array(pos_deb).T,pre_post_time)
            t_fin, pos_fin_av, pos_fin_born_inf, pos_fin_born_sup = mean_profiles(np.array(pos_fin).T,pre_post_time)
            
            _, vel_av, vel_born_inf, vel_born_sup = mean_profiles(np.array(vel).T,durations)
            _, vel_deb_av, vel_deb_born_inf, vel_deb_born_sup = mean_profiles(np.array(vel_deb).T,pre_post_time)
            _, vel_fin_av, vel_fin_born_inf, vel_fin_born_sup = mean_profiles(np.array(vel_fin).T,pre_post_time)
            
            _, acc_av, acc_born_inf, acc_born_sup = mean_profiles(np.array(acc).T,durations)
            _, acc_deb_av, acc_deb_born_inf, acc_deb_born_sup = mean_profiles(np.array(acc_deb).T,pre_post_time)
            _, acc_fin_av, acc_fin_born_inf, acc_fin_born_sup = mean_profiles(np.array(acc_fin).T,pre_post_time)

            _, fz_av, fz_born_inf, fz_born_sup = mean_profiles(np.array(fz).T,durations)
            _, fz_deb_av, fz_deb_born_inf, fz_deb_born_sup = mean_profiles(np.array(fz_deb).T,pre_post_time)
            _, fz_fin_av, fz_fin_born_inf, fz_fin_born_sup = mean_profiles(np.array(fz_fin).T,pre_post_time)

            print(len(fz_av))
            
            # _, fz_theo_av, fz_theo_born_inf, fz_theo_born_sup = mean_profiles(np.array(fz_theo).T,durations)
            # _, fz_theo_deb_av, fz_theo_deb_born_inf, fz_theo_deb_born_sup = mean_profiles(np.array(fz_theo_deb).T,pre_post_time)
            # _, fz_theo_fin_av, fz_theo_fin_born_inf, fz_theo_fin_born_sup = mean_profiles(np.array(fz_theo_fin).T,pre_post_time)

            _, tau_av, tau_born_inf, tau_born_sup = mean_profiles(np.array(tau).T,durations)
            _, tau_deb_av, tau_deb_born_inf, tau_deb_born_sup = mean_profiles(np.array(tau_deb).T,pre_post_time)
            _, tau_fin_av, tau_fin_born_inf, tau_fin_born_sup = mean_profiles(np.array(tau_fin).T,pre_post_time)

            _, tau_theo_av, tau_theo_born_inf, tau_theo_born_sup = mean_profiles(np.array(tau_theo).T,durations)
            _, tau_theo_deb_av, tau_theo_deb_born_inf, tau_theo_deb_born_sup = mean_profiles(np.array(tau_theo_deb).T,pre_post_time)
            _, tau_theo_fin_av, tau_theo_fin_born_inf, tau_theo_fin_born_sup = mean_profiles(np.array(tau_theo_fin).T,pre_post_time)
            
                
            t = t+pre_post_time
            t_fin = t_fin+t[-1]


            # la j'ai trajecroire moyenne pour un sujet, une condition

            pos_cond.append(pos_av)
            vel_cond.append(vel_av)
            acc_cond.append(acc_av)
            fz_cond.append(fz_av)
            pos_deb_cond.append(pos_deb_av)
            pos_fin_cond.append(pos_fin_av)
            vel_deb_cond.append(vel_deb_av)
            vel_fin_cond.append(vel_fin_av)
            acc_deb_cond.append(acc_deb_av)
            acc_fin_cond.append(acc_fin_av)
            fz_deb_cond.append(fz_deb_av)
            fz_fin_cond.append(fz_fin_av)
            tau_cond.append(tau_av)
            tau_theo_cond.append(tau_theo_av)
            tau_deb_cond.append(tau_deb_av)
            tau_fin_cond.append(tau_fin_av)
            tau_theo_deb_cond.append(tau_theo_deb_av)
            tau_theo_fin_cond.append(tau_theo_fin_av)

        #     print(len(fz_cond))

        # print("coucou")

        pos_all.update({
            cond : {
                "deb" : np.array(pos_deb_cond).T,
                "mvt" : np.array(pos_cond).T,
                "fin" : np.array(pos_fin_cond).T,
            }
        })
        vel_all.update({
            cond : {
                "deb" : np.array(vel_deb_cond).T,
                "mvt" : np.array(vel_cond).T,
                "fin" : np.array(vel_fin_cond).T,
            }
        })
        acc_all.update({
            cond : {
                "deb" : np.array(acc_deb_cond).T,
                "mvt" : np.array(acc_cond).T,
                "fin" : np.array(acc_fin_cond).T,
            }
        })
        fz_all.update({
            cond : {
                "deb" : np.array(fz_deb_cond).T,
                "mvt" : np.array(fz_cond).T,
                "fin" : np.array(fz_fin_cond).T,
            }
        })
        tau_all.update({
            cond : {
                "deb" : np.array(tau_deb_cond).T,
                "mvt" : np.array(tau_cond).T,
                "fin" : np.array(tau_fin_cond).T,
            }
        })
        tau_theo_all.update({
            cond : {
                "deb" : np.array(tau_theo_deb_cond).T,
                "mvt" : np.array(tau_theo_cond).T,
                "fin" : np.array(tau_theo_fin_cond).T,
            }
        })
        
    return {
        "pos" : pos_all,
        "vel" : vel_all,
        "acc" : acc_all,
        "fz" : fz_all,
        "tau" : tau_all,
        "tau_theo" : tau_theo_all,
        }

def aggregate_all_trajectories_rob_one_subj(df, sample_rate_rob = 200, unit = 'rad'):
    # pass df.query(direction == ?)
    # one traj by subject for pos, vel, acc, tau_i
    print(df)
    # based on rob data
    pos_all = {}
    vel_all = {}
    acc_all = {}
    fz_all = {}
    tau_all = {}
    tau_theo_all = {}

    pre_post_time = 0.5 # tps avant et aprés mouvement effectif
    pre_post_sample = int(pre_post_time*sample_rate_rob)
    print(pre_post_sample)


    
    for cond in df.condition.unique():
        df_ = df.query("condition == @cond")
        pos_cond = []
        vel_cond = []
        acc_cond = []
        tau_cond = []
        fz_cond = []
        tau_theo_cond = []
        pos_deb_cond = []
        pos_fin_cond = []
        vel_deb_cond = []
        vel_fin_cond = []
        acc_deb_cond = []
        acc_fin_cond = []
        tau_deb_cond = []
        tau_fin_cond = []
        fz_deb_cond = []
        fz_fin_cond = []
        tau_theo_deb_cond = []
        tau_theo_fin_cond = []
        
        for subj, df_subj in df_.groupby('subject'):
            print(subj)
            
            pos = []
            vel = []
            acc = []
            tau = []
            fz = []
            tau_theo = []
            pos_deb = []
            pos_fin = []
            vel_deb = []
            vel_fin = []
            acc_deb = []
            acc_fin = []
            tau_deb = []
            tau_fin = []
            tau_theo_deb = []
            tau_theo_fin = []

            fz_deb = []
            fz_fin = []

            # fz_theo_deb = []
            # fz_theo_fin = []
            
            durations = []
            for mvt_id, movement_row in df_subj.groupby('movement'):
                with open(movement_row['file_name'].values[0], 'rb') as data_file:
                    data_movement = pickle.load(data_file)
                    df_mov = data_movement.get('df_mov')

                start = data_movement.get('bounds')[0]
                end = data_movement.get('bounds')[1]
                
                # print(df_mov)
                start_prev = np.clip(start -pre_post_sample,0, start) 
                end_foll = np.clip(end +pre_post_sample,0, len(df_mov)-1)
                
                if unit == 'deg':
                    df_mov['pos'] = df_mov['pos']*180/np.pi
                    df_mov['vel'] = df_mov['vel']*180/np.pi
                    df_mov['acc'] = df_mov['acc']*180/np.pi
                        
                durations = movement_row['MD'].values[0]
                # Position
                pos.append(mct.signal.resample_by_interpolation(df_mov['pos'].iloc[start:end],len(df_mov['pos'].iloc[start:end]),1000))
                pos_deb.append(mct.signal.resample_by_interpolation(df_mov['pos'].iloc[start_prev:start],len(df_mov['pos'].iloc[start_prev:start]),pre_post_sample))
                pos_fin.append(mct.signal.resample_by_interpolation(df_mov['pos'].iloc[end:end_foll],len(df_mov['pos'].iloc[end:end_foll]),pre_post_sample))
                
                # Velocity
                vel.append(mct.signal.resample_by_interpolation(df_mov['vel'].iloc[start:end],len(df_mov['vel'].iloc[start:end]),1000))
                vel_deb.append(mct.signal.resample_by_interpolation(df_mov['vel'].iloc[start_prev:start],len(df_mov['vel'].iloc[start_prev:start]),pre_post_sample))
                vel_fin.append(mct.signal.resample_by_interpolation(df_mov['vel'].iloc[end:end_foll],len(df_mov['vel'].iloc[end:end_foll]),pre_post_sample))
                
                #Acceleration
                acc.append(mct.signal.resample_by_interpolation(df_mov['acc'].iloc[start:end],len(df_mov['acc'].iloc[start:end]),1000))
                acc_deb.append(mct.signal.resample_by_interpolation(df_mov['acc'].iloc[start_prev:start],len(df_mov['acc'].iloc[start_prev:start]),pre_post_sample))
                acc_fin.append(mct.signal.resample_by_interpolation(df_mov['acc'].iloc[end:end_foll],len(df_mov['acc'].iloc[end:end_foll]),pre_post_sample))

                # #fz
                fz.append(mct.signal.resample_by_interpolation(df_mov['Fz'].iloc[start:end],len(df_mov['Fz'].iloc[start:end]),1000))
                fz_deb.append(mct.signal.resample_by_interpolation(df_mov['Fz'].iloc[start_prev:start],len(df_mov['Fz'].iloc[start_prev:start]),pre_post_sample))
                fz_fin.append(mct.signal.resample_by_interpolation(df_mov['Fz'].iloc[end:end_foll],len(df_mov['Fz'].iloc[end:end_foll]),pre_post_sample))

                # #fz_theo
                # fz_theo.append(mct.signal.resample_by_interpolation(df_mov['Fz_theo'].iloc[start:end],len(df_mov['Fz_theo'].iloc[start:end]),1000))
                # fz_theo_deb.append(mct.signal.resample_by_interpolation(df_mov['Fz_theo'].iloc[start_prev:start],len(df_mov['Fz_theo'].iloc[start_prev:start]),pre_post_sample))
                # fz_theo_fin.append(mct.signal.resample_by_interpolation(df_mov['Fz_theo'].iloc[end:end_foll],len(df_mov['Fz_theo'].iloc[end:end_foll]),pre_post_sample))

                #tau_i
                tau.append(mct.signal.resample_by_interpolation(df_mov['tau_i'].iloc[start:end],len(df_mov['tau_i'].iloc[start:end]),1000))
                tau_deb.append(mct.signal.resample_by_interpolation(df_mov['tau_i'].iloc[start_prev:start],len(df_mov['tau_i'].iloc[start_prev:start]),pre_post_sample))
                tau_fin.append(mct.signal.resample_by_interpolation(df_mov['tau_i'].iloc[end:end_foll],len(df_mov['tau_i'].iloc[end:end_foll]),pre_post_sample))

                #tau_i_theorique
                tau_theo.append(mct.signal.resample_by_interpolation(df_mov['tau_i_theo'].iloc[start:end],len(df_mov['tau_i_theo'].iloc[start:end]),1000))
                tau_theo_deb.append(mct.signal.resample_by_interpolation(df_mov['tau_i_theo'].iloc[start_prev:start],len(df_mov['tau_i_theo'].iloc[start_prev:start]),pre_post_sample))
                tau_theo_fin.append(mct.signal.resample_by_interpolation(df_mov['tau_i_theo'].iloc[end:end_foll],len(df_mov['tau_i_theo'].iloc[end:end_foll]),pre_post_sample))
                
                
            

                # durations = np.array(durations).mean()
                t, pos_av, pos_born_inf, pos_born_sup = mean_profiles(np.array(pos).T,durations)
                t_deb, pos_deb_av, pos_deb_born_inf, pos_deb_born_sup = mean_profiles(np.array(pos_deb).T,pre_post_time)
                t_fin, pos_fin_av, pos_fin_born_inf, pos_fin_born_sup = mean_profiles(np.array(pos_fin).T,pre_post_time)
                
                _, vel_av, vel_born_inf, vel_born_sup = mean_profiles(np.array(vel).T,durations)
                _, vel_deb_av, vel_deb_born_inf, vel_deb_born_sup = mean_profiles(np.array(vel_deb).T,pre_post_time)
                _, vel_fin_av, vel_fin_born_inf, vel_fin_born_sup = mean_profiles(np.array(vel_fin).T,pre_post_time)
                
                _, acc_av, acc_born_inf, acc_born_sup = mean_profiles(np.array(acc).T,durations)
                _, acc_deb_av, acc_deb_born_inf, acc_deb_born_sup = mean_profiles(np.array(acc_deb).T,pre_post_time)
                _, acc_fin_av, acc_fin_born_inf, acc_fin_born_sup = mean_profiles(np.array(acc_fin).T,pre_post_time)

                _, fz_av, fz_born_inf, fz_born_sup = mean_profiles(np.array(fz).T,durations)
                _, fz_deb_av, fz_deb_born_inf, fz_deb_born_sup = mean_profiles(np.array(fz_deb).T,pre_post_time)
                _, fz_fin_av, fz_fin_born_inf, fz_fin_born_sup = mean_profiles(np.array(fz_fin).T,pre_post_time)

                print(len(fz_av))
                
                # _, fz_theo_av, fz_theo_born_inf, fz_theo_born_sup = mean_profiles(np.array(fz_theo).T,durations)
                # _, fz_theo_deb_av, fz_theo_deb_born_inf, fz_theo_deb_born_sup = mean_profiles(np.array(fz_theo_deb).T,pre_post_time)
                # _, fz_theo_fin_av, fz_theo_fin_born_inf, fz_theo_fin_born_sup = mean_profiles(np.array(fz_theo_fin).T,pre_post_time)

                _, tau_av, tau_born_inf, tau_born_sup = mean_profiles(np.array(tau).T,durations)
                _, tau_deb_av, tau_deb_born_inf, tau_deb_born_sup = mean_profiles(np.array(tau_deb).T,pre_post_time)
                _, tau_fin_av, tau_fin_born_inf, tau_fin_born_sup = mean_profiles(np.array(tau_fin).T,pre_post_time)

                _, tau_theo_av, tau_theo_born_inf, tau_theo_born_sup = mean_profiles(np.array(tau_theo).T,durations)
                _, tau_theo_deb_av, tau_theo_deb_born_inf, tau_theo_deb_born_sup = mean_profiles(np.array(tau_theo_deb).T,pre_post_time)
                _, tau_theo_fin_av, tau_theo_fin_born_inf, tau_theo_fin_born_sup = mean_profiles(np.array(tau_theo_fin).T,pre_post_time)
                
                    
                t = t+pre_post_time
                t_fin = t_fin+t[-1]


                # la j'ai trajecroire moyenne pour un sujet, une condition

                pos_cond.append(pos_av)
                vel_cond.append(vel_av)
                acc_cond.append(acc_av)
                fz_cond.append(fz_av)
                pos_deb_cond.append(pos_deb_av)
                pos_fin_cond.append(pos_fin_av)
                vel_deb_cond.append(vel_deb_av)
                vel_fin_cond.append(vel_fin_av)
                acc_deb_cond.append(acc_deb_av)
                acc_fin_cond.append(acc_fin_av)
                fz_deb_cond.append(fz_deb_av)
                fz_fin_cond.append(fz_fin_av)
                tau_cond.append(tau_av)
                tau_theo_cond.append(tau_theo_av)
                tau_deb_cond.append(tau_deb_av)
                tau_fin_cond.append(tau_fin_av)
                tau_theo_deb_cond.append(tau_theo_deb_av)
                tau_theo_fin_cond.append(tau_theo_fin_av)

        #     print(len(fz_cond))

        # print("coucou")

        pos_all.update({
            cond : {
                "deb" : np.array(pos_deb_cond).T,
                "mvt" : np.array(pos_cond).T,
                "fin" : np.array(pos_fin_cond).T,
            }
        })
        vel_all.update({
            cond : {
                "deb" : np.array(vel_deb_cond).T,
                "mvt" : np.array(vel_cond).T,
                "fin" : np.array(vel_fin_cond).T,
            }
        })
        acc_all.update({
            cond : {
                "deb" : np.array(acc_deb_cond).T,
                "mvt" : np.array(acc_cond).T,
                "fin" : np.array(acc_fin_cond).T,
            }
        })
        fz_all.update({
            cond : {
                "deb" : np.array(fz_deb_cond).T,
                "mvt" : np.array(fz_cond).T,
                "fin" : np.array(fz_fin_cond).T,
            }
        })
        tau_all.update({
            cond : {
                "deb" : np.array(tau_deb_cond).T,
                "mvt" : np.array(tau_cond).T,
                "fin" : np.array(tau_fin_cond).T,
            }
        })
        tau_theo_all.update({
            cond : {
                "deb" : np.array(tau_theo_deb_cond).T,
                "mvt" : np.array(tau_theo_cond).T,
                "fin" : np.array(tau_theo_fin_cond).T,
            }
        })
        
    return {
        "pos" : pos_all,
        "vel" : vel_all,
        "acc" : acc_all,
        "fz" : fz_all,
        "tau" : tau_all,
        "tau_theo" : tau_theo_all,
        }

def aggregate_all_trajectories_qual_noEMG(df_kin, df_emg, sample_rate_kin = 179, sample_rate_emg = 2000, unit = 'rad'):
    # one traj by subject for pos, vel, acc, flexors, extensors
    #print(df_emg)
    #print(df_emg.columns)
    #print(df_kin)
    

    pos_all = {}
    vel_all = {}
    acc_all = {}
    q_sh_all = {}
    q_el_all = {}

    pre_post_time = 0.5
    pre_post_sample = int(pre_post_time*sample_rate_kin)

    for cond in df_kin.condition.unique():
        print(cond)
        df_kin_ = df_kin.query("condition == @cond")

        pos_cond = {}
        vel_cond = {}
        acc_cond = {}
        q_sh_cond = {}
        q_el_cond = {}
        
        for targ in range(1,6):
            df_kin_targ = df_kin_.query("target == @targ")

            posX_targ = []
            posZ_targ = []
            vel_targ = []
            acc_targ = []
            q_sh_targ = []
            q_el_targ = []
            posX_deb_targ = []
            posX_fin_targ = []
            posZ_deb_targ = []
            posZ_fin_targ = []
            vel_deb_targ = []
            vel_fin_targ = []
            acc_deb_targ = []
            acc_fin_targ = []
            q_sh_deb_targ = []
            q_sh_fin_targ = []
            q_el_deb_targ = []
            q_el_fin_targ = []

            posX_sup_targ = []
            posX_inf_targ = []
            posZ_sup_targ = []
            posZ_inf_targ = []
            vel_sup_targ = []
            vel_inf_targ = []
            acc_sup_targ = []
            acc_inf_targ = []
            q_sh_sup_targ = []
            q_sh_inf_targ = []
            q_el_sup_targ = []
            q_el_inf_targ = []
           
            # if df_kin_.subject.unique() != df_emg_.subject.unique():
            #     print("WARNING - df kin and df emg doesn't have the same subjects")
            for (subj, df_subj) in df_kin_targ.groupby('subject'):

                #if cond == '1G':
                #    quantile_PV = df_subj['PV'].quantile(0.5)
                #    df_subj = df_subj[df_subj['PV'] > quantile_PV]
                # print(subj, df_subj, subj_emg, df_subj_emg)
                #print(subj)
                 #Kinematic
                posX = []
                posZ = []
                vel = []
                acc = []
                q_sh = []
                q_el = []
                posX_deb = []
                posX_fin = []
                posZ_deb = []
                posZ_fin = []
                vel_deb = []
                vel_fin = []
                acc_deb = []
                acc_fin = []
                q_sh_deb = []
                q_sh_fin = []
                q_el_deb = []
                q_el_fin = []
                
                durations = []
                for mvt_id, movement_row in df_subj.groupby('movement'):
                    with open(movement_row['file_name'].values[0], 'rb') as data_file:
                        data_movement = pickle.load(data_file)
                        df_mov = data_movement.get('df_mov')

                    start = data_movement.get('bounds')[0]
                    end = data_movement.get('bounds')[1]
                    
                    start_prev = np.clip(start - pre_post_sample,0, start) 
                    end_foll = np.clip(end +pre_post_sample,0, len(df_mov)-1)
                    
                    if unit == 'deg':
                        df_mov['pos'] = df_mov['pos']*180/np.pi -90
                        df_mov['vel'] = df_mov['vel']*180/np.pi 
                        df_mov['acc'] = df_mov['acc']*180/np.pi 
                            
                    #print(df_mov)
                    durations.append(movement_row['MD'].values[0])
                    # Position on X axis
                    posX.append(mct.signal.resample_by_interpolation(df_mov['posX'].iloc[start:end],len(df_mov['posX'].iloc[start:end]),1000))
                    posX_deb.append(mct.signal.resample_by_interpolation(df_mov['posX'].iloc[start_prev:start],len(df_mov['posX'].iloc[start_prev:start]),pre_post_sample))
                    if len(df_mov['posX'].iloc[end:end_foll]) > 0 :
                        posX_fin.append(mct.signal.resample_by_interpolation(df_mov['posX'].iloc[end:end_foll],len(df_mov['posX'].iloc[end:end_foll]),pre_post_sample))
                    
                    # Position on Z axis
                    posZ.append(mct.signal.resample_by_interpolation(df_mov['posZ'].iloc[start:end],len(df_mov['posZ'].iloc[start:end]),1000))
                    posZ_deb.append(mct.signal.resample_by_interpolation(df_mov['posZ'].iloc[start_prev:start],len(df_mov['posZ'].iloc[start_prev:start]),pre_post_sample))
                    if len(df_mov['posZ'].iloc[end:end_foll]) > 0 :
                        posZ_fin.append(mct.signal.resample_by_interpolation(df_mov['posZ'].iloc[end:end_foll],len(df_mov['posZ'].iloc[end:end_foll]),pre_post_sample))
                        
                    # Velocity
                    vel.append(mct.signal.resample_by_interpolation(df_mov['vel'].iloc[start:end],len(df_mov['vel'].iloc[start:end]),1000))
                    vel_deb.append(mct.signal.resample_by_interpolation(df_mov['vel'].iloc[start_prev:start],len(df_mov['vel'].iloc[start_prev:start]),pre_post_sample))
                    if len(df_mov['vel'].iloc[end:end_foll]) > 0 :
                        vel_fin.append(mct.signal.resample_by_interpolation(df_mov['vel'].iloc[end:end_foll],len(df_mov['vel'].iloc[end:end_foll]),pre_post_sample))
                    
                    # Acceleration
                    acc.append(mct.signal.resample_by_interpolation(df_mov['acc'].iloc[start:end],len(df_mov['acc'].iloc[start:end]),1000))
                    acc_deb.append(mct.signal.resample_by_interpolation(df_mov['acc'].iloc[start_prev:start],len(df_mov['acc'].iloc[start_prev:start]),pre_post_sample))
                    if len(df_mov['acc'].iloc[end:end_foll]) > 0 :
                        acc_fin.append(mct.signal.resample_by_interpolation(df_mov['acc'].iloc[end:end_foll],len(df_mov['acc'].iloc[end:end_foll]),pre_post_sample))

                    # Shoulder angle
                    q_sh.append(mct.signal.resample_by_interpolation(df_mov['q_sh'].iloc[start:end],len(df_mov['q_sh'].iloc[start:end]),1000))
                    q_sh_deb.append(mct.signal.resample_by_interpolation(df_mov['q_sh'].iloc[start_prev:start],len(df_mov['q_sh'].iloc[start_prev:start]),pre_post_sample))
                    if len(df_mov['q_sh'].iloc[end:end_foll]) > 0 :
                        q_sh_fin.append(mct.signal.resample_by_interpolation(df_mov['q_sh'].iloc[end:end_foll],len(df_mov['q_sh'].iloc[end:end_foll]),pre_post_sample))

                    # Elbow angle
                    q_el.append(mct.signal.resample_by_interpolation(df_mov['q_el'].iloc[start:end],len(df_mov['q_el'].iloc[start:end]),1000))
                    q_el_deb.append(mct.signal.resample_by_interpolation(df_mov['q_el'].iloc[start_prev:start],len(df_mov['q_el'].iloc[start_prev:start]),pre_post_sample))
                    if len(df_mov['q_el'].iloc[end:end_foll]) > 0 :
                        q_el_fin.append(mct.signal.resample_by_interpolation(df_mov['q_el'].iloc[end:end_foll],len(df_mov['q_el'].iloc[end:end_foll]),pre_post_sample))

                durations = np.array(durations).mean()
                t, posX_av, posX_born_inf, posX_born_sup = mean_profiles(np.array(posX).T,durations)
                t_deb, posX_deb_av, posX_deb_born_inf, posX_deb_born_sup = mean_profiles(np.array(posX_deb).T,pre_post_time)
                t_fin, posX_fin_av, posX_fin_born_inf, posX_fin_born_sup = mean_profiles(np.array(posX_fin).T,pre_post_time)
                
                _, posZ_av, posZ_born_inf, posZ_born_sup = mean_profiles(np.array(posZ).T,durations)
                _, posZ_deb_av, posZ_deb_born_inf, posZ_deb_born_sup = mean_profiles(np.array(posZ_deb).T,pre_post_time)
                _, posZ_fin_av, posZ_fin_born_inf, posZ_fin_born_sup = mean_profiles(np.array(posZ_fin).T,pre_post_time)
                
                _, vel_av, vel_born_inf, vel_born_sup = mean_profiles(np.array(vel).T,durations)
                _, vel_deb_av, vel_deb_born_inf, vel_deb_born_sup = mean_profiles(np.array(vel_deb).T,pre_post_time)
                _, vel_fin_av, vel_fin_born_inf, vel_fin_born_sup = mean_profiles(np.array(vel_fin).T,pre_post_time)
                
                _, acc_av, acc_born_inf, acc_born_sup = mean_profiles(np.array(acc).T,durations)
                _, acc_deb_av, acc_deb_born_inf, acc_deb_born_sup = mean_profiles(np.array(acc_deb).T,pre_post_time)
                _, acc_fin_av, acc_fin_born_inf, acc_fin_born_sup = mean_profiles(np.array(acc_fin).T,pre_post_time)

                _, q_sh_av, q_sh_born_inf, q_sh_born_sup = mean_profiles(np.array(q_sh).T,durations)
                _, q_sh_deb_av, q_sh_deb_born_inf, q_sh_deb_born_sup = mean_profiles(np.array(q_sh_deb).T,pre_post_time)
                _, q_sh_fin_av, q_sh_fin_born_inf, q_sh_fin_born_sup = mean_profiles(np.array(q_sh_fin).T,pre_post_time)

                _, q_el_av, q_el_born_inf, q_el_born_sup = mean_profiles(np.array(q_el).T,durations)
                _, q_el_deb_av, q_el_deb_born_inf, q_el_deb_born_sup = mean_profiles(np.array(q_el_deb).T,pre_post_time)
                _, q_el_fin_av, q_el_fin_born_inf, q_el_fin_born_sup = mean_profiles(np.array(q_el_fin).T,pre_post_time)
                            
                t = t+pre_post_time
                t_fin = t_fin+t[-1]

                posX_targ.append(posX_av)
                posZ_targ.append(posZ_av)
                vel_targ.append(vel_av)
                acc_targ.append(acc_av)
                q_sh_targ.append(q_sh_av)
                q_el_targ.append(q_el_av)
                
                posX_deb_targ.append(posX_deb_av)
                posX_fin_targ.append(posX_fin_av)
                posZ_deb_targ.append(posZ_deb_av)
                posZ_fin_targ.append(posZ_fin_av)
                vel_deb_targ.append(vel_deb_av)
                vel_fin_targ.append(vel_fin_av)
                acc_deb_targ.append(acc_deb_av)
                acc_fin_targ.append(acc_fin_av)
                q_sh_deb_targ.append(q_sh_deb_av)
                q_sh_fin_targ.append(q_sh_fin_av)
                q_el_deb_targ.append(q_el_deb_av)
                q_el_fin_targ.append(q_el_fin_av)
                
                posX_sup_targ.append(posX_born_sup)
                posX_inf_targ.append(posX_born_inf)
                posZ_sup_targ.append(posZ_born_sup)
                posZ_inf_targ.append(posZ_born_inf)
                vel_sup_targ.append(vel_born_sup)
                vel_inf_targ.append(vel_born_inf)
                acc_sup_targ.append(acc_born_sup)
                acc_inf_targ.append(acc_born_inf)
                q_sh_sup_targ.append(q_sh_born_sup)
                q_sh_inf_targ.append(q_sh_born_inf)
                q_el_sup_targ.append(q_el_born_sup)
                q_el_inf_targ.append(q_el_born_inf)

            # Save lists per target in conditions dictionnaries
            pos_cond.update({
                targ : {
                    "debX" : np.array(posX_deb_targ).T,
                    "mvtX" : np.array(posX_targ).T,
                    "finX" : np.array(posX_fin_targ).T,
                    "supX" : np.array(posX_sup_targ).T,
                    "infX" : np.array(posX_inf_targ).T,
                    "debZ" : np.array(posZ_deb_targ).T,
                    "mvtZ" : np.array(posZ_targ).T,
                    "finZ" : np.array(posZ_fin_targ).T,
                    "supZ" : np.array(posZ_sup_targ).T,
                    "infZ" : np.array(posZ_inf_targ).T,
                }
            })
            vel_cond.update({
                targ : {
                    "deb" : np.array(vel_deb_targ).T,
                    "mvt" : np.array(vel_targ).T,
                    "fin" : np.array(vel_fin_targ).T,
                    "sup" : np.array(vel_sup_targ).T,
                    "inf" : np.array(vel_inf_targ).T,
                }
            })
            acc_cond.update({
                targ : {
                    "deb" : np.array(acc_deb_targ).T,
                    "mvt" : np.array(acc_targ).T,
                    "fin" : np.array(acc_fin_targ).T,
                    "sup" : np.array(acc_sup_targ).T,
                    "inf" : np.array(acc_inf_targ).T,
                }
            })
            q_sh_cond.update({
                targ : {
                    "deb" : np.array(q_sh_deb_targ).T,
                    "mvt" : np.array(q_sh_targ).T,
                    "fin" : np.array(q_sh_fin_targ).T,
                    "sup" : np.array(q_sh_sup_targ).T,
                    "inf" : np.array(q_sh_inf_targ).T,
                }
            })
            q_el_cond.update({
                targ : {
                    "deb" : np.array(q_el_deb_targ).T,
                    "mvt" : np.array(q_el_targ).T,
                    "fin" : np.array(q_el_fin_targ).T,
                    "sup" : np.array(q_el_sup_targ).T,
                    "inf" : np.array(q_el_inf_targ).T,
                }
            })

        # Save dictionnaries of trajectories per condition
        pos_all.update({cond : pos_cond})
        vel_all.update({cond : vel_cond})
        acc_all.update({cond : acc_cond})
        q_sh_all.update({cond: q_sh_cond})
        q_el_all.update({cond: q_el_cond})

    return {
        "pos" : pos_all,
        "vel" : vel_all,
        "acc" : acc_all,
        "q_sh": q_sh_all,
        "q_el": q_el_all
    }

def aggregate_interEfforts_trajectories(df, sample_rate_rob = 200):
    """Aggregates arm and forearm interaction forces"""    

    fzA_all = {}
    normA_all = {}
    fzFA_all = {}
    normFA_all = {}

    pre_post_time = 0.5
    pre_post_sample = int(pre_post_time*sample_rate_rob)

    for cond in df.condition.unique():
        print(cond)
        df_rob_ = df.query("condition == @cond")

        fzA_cond = {}
        normA_cond = {}
        fzFA_cond = {}
        normFA_cond = {}
        
        for targ in range(1,6):
            df_rob_targ = df_rob_.query("target == @targ")

            # Lists to store data of each participant per target
            fzA_targ = []
            normA_targ = []
            fzFA_targ = []
            normFA_targ = []
            # Lists to store start and end data of each participant per target
            fzA_deb_targ = []
            fzA_fin_targ = []
            normA_deb_targ = []
            normA_fin_targ = []
            fzFA_deb_targ = []
            fzFA_fin_targ = []
            normFA_deb_targ = []
            normFA_fin_targ = []
            # Lists to store sup and inf bounds of each participant per target
            fzA_sup_targ = []
            fzA_inf_targ = []
            normA_sup_targ = []
            normA_inf_targ = []
            fzFA_sup_targ = []
            fzFA_inf_targ = []
            normFA_sup_targ = []
            normFA_inf_targ = []
        
            for (subj, df_subj) in df_rob_targ.groupby('subject'):
                fzA = []
                fzFA = []
                normA = []
                normFA = []
                fzA_deb = []
                fzA_fin = []
                fzFA_deb = []
                fzFA_fin = []
                normA_deb = []
                normA_fin = []
                normFA_deb = []
                normFA_fin = []
                
                durations = []
                for mvt_id, movement_row in df_subj.groupby('movement'):
                    with open(movement_row['file_name'].values[0], 'rb') as data_file:
                        data_movement = pickle.load(data_file)
                        df_mov = data_movement.get('df_mov')

                    start = data_movement.get('bounds')[0]
                    end = data_movement.get('bounds')[1]
                    
                    start_prev = np.clip(start - pre_post_sample,0, start) 
                    end_foll = np.clip(end +pre_post_sample,0, len(df_mov)-1)
                            
                    #print(df_mov)
                    durations.append(movement_row['MD'].values[0])
                    # Fz interaction force at the arm
                    fzA.append(mct.signal.resample_by_interpolation(df_mov['Fz_A'].iloc[start:end],len(df_mov['Fz_A'].iloc[start:end]),1000))
                    fzA_deb.append(mct.signal.resample_by_interpolation(df_mov['Fz_A'].iloc[start_prev:start],len(df_mov['Fz_A'].iloc[start_prev:start]),pre_post_sample))
                    if len(df_mov['Fz_A'].iloc[end:end_foll]) > 0 :
                        fzA_fin.append(mct.signal.resample_by_interpolation(df_mov['Fz_A'].iloc[end:end_foll],len(df_mov['Fz_A'].iloc[end:end_foll]),pre_post_sample))
                    
                    # Fz interaction force at the arm
                    fzFA.append(mct.signal.resample_by_interpolation(df_mov['Fz_FA'].iloc[start:end],len(df_mov['Fz_FA'].iloc[start:end]),1000))
                    fzFA_deb.append(mct.signal.resample_by_interpolation(df_mov['Fz_FA'].iloc[start_prev:start],len(df_mov['Fz_FA'].iloc[start_prev:start]),pre_post_sample))
                    if len(df_mov['Fz_A'].iloc[end:end_foll]) > 0 :
                        fzFA_fin.append(mct.signal.resample_by_interpolation(df_mov['Fz_FA'].iloc[end:end_foll],len(df_mov['Fz_FA'].iloc[end:end_foll]),pre_post_sample))

                    # Compute norm of the interaction forces at the arm
                    norm_A = np.sqrt(np.array(df_mov['Fx_A'])**2 + np.array(df_mov['Fy_A'])**2 + np.array(df_mov['Fz_A'])**2)
                    norm_FA = np.sqrt(np.array(df_mov['Fx_FA'])**2 + np.array(df_mov['Fy_FA'])**2 + np.array(df_mov['Fz_FA'])**2)

                    # Resampled norm at the arm
                    normA.append(mct.signal.resample_by_interpolation(norm_A[start:end],len(norm_A[start:end]),1000))
                    normA_deb.append(mct.signal.resample_by_interpolation(norm_A[start_prev:start],len(norm_A[start_prev:start]),pre_post_sample))
                    if len(norm_A[end:end_foll]) > 0 :
                        normA_fin.append(mct.signal.resample_by_interpolation(norm_A[end:end_foll],len(norm_A[end:end_foll]),pre_post_sample))

                    # Resampled norm at the forearm
                    normFA.append(mct.signal.resample_by_interpolation(norm_FA[start:end],len(norm_FA[start:end]),1000))
                    normFA_deb.append(mct.signal.resample_by_interpolation(norm_FA[start_prev:start],len(norm_FA[start_prev:start]),pre_post_sample))
                    if len(norm_A[end:end_foll]) > 0 :
                        normFA_fin.append(mct.signal.resample_by_interpolation(norm_FA[end:end_foll],len(norm_FA[end:end_foll]),pre_post_sample))

                durations = np.array(durations).mean()
                # Average Fz at the arm per subject
                t, fzA_av, fzA_born_inf, fzA_born_sup = mean_profiles(np.array(fzA).T,durations)
                t_deb, fzA_deb_av, fzA_deb_born_inf, fzA_deb_born_sup = mean_profiles(np.array(fzA_deb).T,pre_post_time)
                t_fin, fzA_fin_av, fzA_fin_born_inf, fzA_fin_born_sup = mean_profiles(np.array(fzA_fin).T,pre_post_time)
                # Average Fz at the forearm per subject
                _, fzFA_av, fzFA_born_inf, fzFA_born_sup = mean_profiles(np.array(fzFA).T,durations)
                _, fzFA_deb_av, fzFA_deb_born_inf, fzFA_deb_born_sup = mean_profiles(np.array(fzFA_deb).T,pre_post_time)
                _, fzFA_fin_av, fzFA_fin_born_inf, fzFA_fin_born_sup = mean_profiles(np.array(fzFA_fin).T,pre_post_time)
                # Average norm at the arm per subject
                _, normA_av, normA_born_inf, normA_born_sup = mean_profiles(np.array(normA).T,durations)
                _, normA_deb_av, normA_deb_born_inf, normA_deb_born_sup = mean_profiles(np.array(normA_deb).T,pre_post_time)
                _, normA_fin_av, normA_fin_born_inf, normA_fin_born_sup = mean_profiles(np.array(normA_fin).T,pre_post_time)
                # Average norm at the forearm per subject
                _, normFA_av, normFA_born_inf, normFA_born_sup = mean_profiles(np.array(normFA).T,durations)
                _, normFA_deb_av, normFA_deb_born_inf, normFA_deb_born_sup = mean_profiles(np.array(normFA_deb).T,pre_post_time)
                _, normFA_fin_av, normFA_fin_born_inf, normFA_fin_born_sup = mean_profiles(np.array(normFA_fin).T,pre_post_time)
                # Handle time            
                t = t+pre_post_time
                t_fin = t_fin+t[-1]

                # Save data per target
                fzA_targ.append(fzA_av)
                fzFA_targ.append(fzFA_av)
                normA_targ.append(normA_av)
                normFA_targ.append(normFA_av)
                
                fzA_deb_targ.append(fzA_deb_av)
                fzA_fin_targ.append(fzA_fin_av)
                fzFA_deb_targ.append(fzFA_deb_av)
                fzFA_fin_targ.append(fzFA_fin_av)
                normA_deb_targ.append(normA_deb_av)
                normA_fin_targ.append(normA_fin_av)
                normFA_deb_targ.append(normFA_deb_av)
                normFA_fin_targ.append(normFA_fin_av)
                
                fzA_sup_targ.append(fzA_born_sup)
                fzA_inf_targ.append(fzA_born_inf)
                fzFA_sup_targ.append(fzFA_born_sup)
                fzFA_inf_targ.append(fzFA_born_inf)
                normA_sup_targ.append(normA_born_sup)
                normA_inf_targ.append(normA_born_inf)
                normFA_sup_targ.append(normFA_born_sup)
                normFA_inf_targ.append(normFA_born_inf)

            # Save lists per target in conditions dictionnaries
            fzA_cond.update({
                targ : {
                    "deb" : np.array(fzA_deb_targ).T,
                    "mvt" : np.array(fzA_targ).T,
                    "fin" : np.array(fzA_fin_targ).T,
                    "sup" : np.array(fzA_sup_targ).T,
                    "inf" : np.array(fzA_inf_targ).T
                }
            })
            fzFA_cond.update({
                targ : {
                    "deb" : np.array(fzFA_deb_targ).T,
                    "mvt" : np.array(fzFA_targ).T,
                    "fin" : np.array(fzFA_fin_targ).T,
                    "sup" : np.array(fzFA_sup_targ).T,
                    "inf" : np.array(fzFA_inf_targ).T
                }
            })
            normA_cond.update({
                targ : {
                    "deb" : np.array(normA_deb_targ).T,
                    "mvt" : np.array(normA_targ).T,
                    "fin" : np.array(normA_fin_targ).T,
                    "sup" : np.array(normA_sup_targ).T,
                    "inf" : np.array(normA_inf_targ).T
                }
            })
            normFA_cond.update({
                targ : {
                    "deb" : np.array(normFA_deb_targ).T,
                    "mvt" : np.array(normFA_targ).T,
                    "fin" : np.array(normFA_fin_targ).T,
                    "sup" : np.array(normFA_sup_targ).T,
                    "inf" : np.array(normFA_inf_targ).T
                }
            })

        # Save dictionnaries of trajectories per condition
        fzA_all.update({cond : fzA_cond})
        fzFA_all.update({cond : fzFA_cond})
        normA_all.update({cond : normA_cond})
        normFA_all.update({cond : normFA_cond})

    return {
        "fzA" : fzA_all,
        "fzFA" : fzFA_all,
        "normA" : normA_all,
        "normFA" : normFA_all
    }


def aggregate_all_trajectories_qual(df_kin, df_emg, sample_rate_kin = 200, sample_rate_emg = 2000, unit = 'rad'):
    # one traj by subject for pos, vel, acc, flexors, extensors
    df_emg = df_emg.merge(df_kin[['subject', 'condition', 'block', 'movement', 'direction','PV']], how = 'inner', on = ['subject', 'condition', 'block', 'movement'])
    #print(df_emg.columns)
    #print(df_kin.columns)
    #print(df_emg)

    #Keep only fastest movements

    pos_all = {}
    vel_all = {}
    acc_all = {}
    bic_all = {}
    bra_all = {}
    tlat_all = {}
    tlng_all = {}
    dant_all = {}
    dmed_all = {}
    dpost_all = {}

    pre_post_time = 0.5
    pre_post_sample = int(pre_post_time*sample_rate_kin)
    pre_post_sample_emg = int(pre_post_time*sample_rate_emg)

    for cond in df_kin.condition.unique():
        df_kin_ = df_kin.query("condition == @cond")
        df_emg_ = df_emg.query("condition == @cond")

        pos_cond = []
        vel_cond = []
        acc_cond = []
        pos_deb_cond = []
        pos_fin_cond = []
        vel_deb_cond = []
        vel_fin_cond = []
        acc_deb_cond = []
        acc_fin_cond = []

        bic_cond = []
        bic_sup_cond = []
        bic_inf_cond = []
        bra_cond = []
        bra_sup_cond = []
        bra_inf_cond = []
        tlng_cond = []
        tlng_sup_cond = []
        tlng_inf_cond = []
        tlat_cond = []
        tlat_sup_cond = []
        tlat_inf_cond = []
        dant_cond = []
        dant_sup_cond = []
        dant_inf_cond = []
        dmed_cond = []
        dmed_sup_cond = []
        dmed_inf_cond = []
        dpost_cond = []
        dpost_sup_cond = []
        dpost_inf_cond = []
        bic_deb_cond = []
        bic_fin_cond = []
        bra_deb_cond = []
        bra_fin_cond = []
        tlng_deb_cond = []
        tlng_fin_cond = []
        tlat_deb_cond = []
        tlat_fin_cond = []
        dant_deb_cond = []
        dant_fin_cond = []
        dmed_deb_cond = []
        dmed_fin_cond = []
        dpost_deb_cond = []
        dpost_fin_cond = []
    
       
        # if df_kin_.subject.unique() != df_emg_.subject.unique():
        #     print("WARNING - df kin and df emg doesn't have the same subjects")
        for (subj, df_subj), (subj_emg, df_subj_emg) in zip(df_kin_.groupby('subject'), df_emg_.groupby('subject')):
            # print(subj, df_subj, subj_emg, df_subj_emg)
            if cond == '1G':
                quantile_PV = df_subj['PV'].quantile(0.75)
                df_subj_emg = df_subj_emg[df_subj_emg['PV'] > quantile_PV]
            #print(subj)
             #Kinematic
            pos = []
            vel = []
            acc = []
            pos_deb = []
            pos_fin = []
            vel_deb = []
            vel_fin = []
            acc_deb = []
            acc_fin = []
            
            durations = []
            for mvt_id, movement_row in df_subj.groupby('movement'):
                with open(movement_row['file_name'].values[0], 'rb') as data_file:
                    data_movement = pickle.load(data_file)
                    df_mov = data_movement.get('df_mov')

                start = data_movement.get('bounds')[0]
                end = data_movement.get('bounds')[1]
                
                start_prev = np.clip(start -pre_post_sample,0, start) 
                end_foll = np.clip(end +pre_post_sample,0, len(df_mov)-1)
                
                if unit == 'deg':
                    df_mov['pos'] = df_mov['pos']*180/np.pi -90
                    df_mov['vel'] = df_mov['vel']*180/np.pi 
                    df_mov['acc'] = df_mov['acc']*180/np.pi 
                        
                durations.append(movement_row['MD'].values[0])
                # Position
                pos.append(mct.signal.resample_by_interpolation(df_mov['posNorm'].iloc[start:end],len(df_mov['posNorm'].iloc[start:end]),1000))
                pos_deb.append(mct.signal.resample_by_interpolation(df_mov['posNorm'].iloc[start_prev:start],len(df_mov['posNorm'].iloc[start_prev:start]),pre_post_sample))
                if len(df_mov['posNorm'].iloc[end:end_foll]) > 0:
                    pos_fin.append(mct.signal.resample_by_interpolation(df_mov['posNorm'].iloc[end:end_foll],len(df_mov['posNorm'].iloc[end:end_foll]),pre_post_sample))
                
                # Velocity
                vel.append(mct.signal.resample_by_interpolation(df_mov['vel'].iloc[start:end],len(df_mov['vel'].iloc[start:end]),1000))
                vel_deb.append(mct.signal.resample_by_interpolation(df_mov['vel'].iloc[start_prev:start],len(df_mov['vel'].iloc[start_prev:start]),pre_post_sample))
                if len(df_mov['vel'].iloc[end:end_foll]) > 0:
                    vel_fin.append(mct.signal.resample_by_interpolation(df_mov['vel'].iloc[end:end_foll],len(df_mov['vel'].iloc[end:end_foll]),pre_post_sample))
                
                #Acceleration
                acc.append(mct.signal.resample_by_interpolation(df_mov['acc'].iloc[start:end],len(df_mov['acc'].iloc[start:end]),1000))
                acc_deb.append(mct.signal.resample_by_interpolation(df_mov['acc'].iloc[start_prev:start],len(df_mov['acc'].iloc[start_prev:start]),pre_post_sample))
                if len(df_mov['acc'].iloc[end:end_foll]) > 0:
                    acc_fin.append(mct.signal.resample_by_interpolation(df_mov['acc'].iloc[end:end_foll],len(df_mov['acc'].iloc[end:end_foll]),pre_post_sample))


            durations = np.array(durations).mean()
            t, pos_av, pos_born_inf, pos_born_sup = mean_profiles(np.array(pos).T,durations)
            t_deb, pos_deb_av, pos_deb_born_inf, pos_deb_born_sup = mean_profiles(np.array(pos_deb).T,pre_post_time)
            t_fin, pos_fin_av, pos_fin_born_inf, pos_fin_born_sup = mean_profiles(np.array(pos_fin).T,pre_post_time)
            
            _, vel_av, vel_born_inf, vel_born_sup = mean_profiles(np.array(vel).T,durations)
            _, vel_deb_av, vel_deb_born_inf, vel_deb_born_sup = mean_profiles(np.array(vel_deb).T,pre_post_time)
            _, vel_fin_av, vel_fin_born_inf, vel_fin_born_sup = mean_profiles(np.array(vel_fin).T,pre_post_time)
            
            _, acc_av, acc_born_inf, acc_born_sup = mean_profiles(np.array(acc).T,durations)
            _, acc_deb_av, acc_deb_born_inf, acc_deb_born_sup = mean_profiles(np.array(acc_deb).T,pre_post_time)
            _, acc_fin_av, acc_fin_born_inf, acc_fin_born_sup = mean_profiles(np.array(acc_fin).T,pre_post_time)
                        
            t = t+pre_post_time
            t_fin = t_fin+t[-1]

            pos_cond.append(pos_av)
            vel_cond.append(vel_av)
            acc_cond.append(acc_av)
            pos_deb_cond.append(pos_deb_av)
            pos_fin_cond.append(pos_fin_av)
            vel_deb_cond.append(vel_deb_av)
            vel_fin_cond.append(vel_fin_av)
            acc_deb_cond.append(acc_deb_av)
            acc_fin_cond.append(acc_fin_av)
            

            #emg

            bic = []
            bra = []
            tlng = []
            tlat = []
            dant = []
            dmed = []
            dpost = []

            bic_deb = []
            bic_fin = []
            bra_deb = []
            bra_fin = []
            tlng_deb = []
            tlng_fin = []
            tlat_deb = []
            tlat_fin = []
            dant_deb = []
            dant_fin = []
            dmed_deb = []
            dmed_fin = []
            dpost_deb = []
            dpost_fin = []

            for mvt_id, movement_row in df_subj_emg.groupby('movement'):
                try:
                    with open(movement_row['file_name'].values[0], 'rb') as data_file:
                        data_movement = pickle.load(data_file)
                        df_mov = data_movement.get('df_mov')

                    start = int(data_movement.get('bounds')[0])-200
                    end = int(data_movement.get('bounds')[1])
                    
                    start_prev = np.clip(start -pre_post_sample_emg,0, start)
                    end_foll = np.clip(end +pre_post_sample_emg,0, len(df_mov)-1)
                    
                    # bic
                    bic.append(mct.signal.resample_by_interpolation(df_mov['Biceps_filt_rect_norm'].iloc[start:end],len(df_mov['Biceps'].iloc[start:end]),1000))
                    bic_deb.append(mct.signal.resample_by_interpolation(df_mov['Biceps_filt_rect_norm'].iloc[start_prev:start],len(df_mov['Biceps'].iloc[start_prev:start]),pre_post_sample_emg))
                    bic_fin.append(mct.signal.resample_by_interpolation(df_mov['Biceps_filt_rect_norm'].iloc[end:end_foll],len(df_mov['Biceps'].iloc[end:end_foll]),pre_post_sample_emg))
                    
                    # bra
                    bra.append(mct.signal.resample_by_interpolation(df_mov['Brachial_filt_rect_norm'].iloc[start:end],len(df_mov['Brachial'].iloc[start:end]),1000))
                    bra_deb.append(mct.signal.resample_by_interpolation(df_mov['Brachial_filt_rect_norm'].iloc[start_prev:start],len(df_mov['Brachial'].iloc[start_prev:start]),pre_post_sample_emg))
                    bra_fin.append(mct.signal.resample_by_interpolation(df_mov['Brachial_filt_rect_norm'].iloc[end:end_foll],len(df_mov['Brachial'].iloc[end:end_foll]),pre_post_sample_emg))
                    
                    # tlat
                    tlat.append(mct.signal.resample_by_interpolation(df_mov['Triceps_Lat_filt_rect_norm'].iloc[start:end],len(df_mov['Triceps_Lat'].iloc[start:end]),1000))
                    tlat_deb.append(mct.signal.resample_by_interpolation(df_mov['Triceps_Lat_filt_rect_norm'].iloc[start_prev:start],len(df_mov['Triceps_Lat'].iloc[start_prev:start]),pre_post_sample_emg))
                    tlat_fin.append(mct.signal.resample_by_interpolation(df_mov['Triceps_Lat_filt_rect_norm'].iloc[end:end_foll],len(df_mov['Triceps_Lat'].iloc[end:end_foll]),pre_post_sample_emg))

                    # tlng
                    tlng.append(mct.signal.resample_by_interpolation(df_mov['Triceps_Lng_filt_rect_norm'].iloc[start:end],len(df_mov['Triceps_Lng'].iloc[start:end]),1000))
                    tlng_deb.append(mct.signal.resample_by_interpolation(df_mov['Triceps_Lng_filt_rect_norm'].iloc[start_prev:start],len(df_mov['Triceps_Lng'].iloc[start_prev:start]),pre_post_sample_emg))
                    tlng_fin.append(mct.signal.resample_by_interpolation(df_mov['Triceps_Lng_filt_rect_norm'].iloc[end:end_foll],len(df_mov['Triceps_Lng'].iloc[end:end_foll]),pre_post_sample_emg))

                    # tlng
                    dant.append(mct.signal.resample_by_interpolation(df_mov['Deltoid_Ant_filt_rect_norm'].iloc[start:end],len(df_mov['Deltoid_Ant'].iloc[start:end]),1000))
                    dant_deb.append(mct.signal.resample_by_interpolation(df_mov['Deltoid_Ant_filt_rect_norm'].iloc[start_prev:start],len(df_mov['Deltoid_Ant'].iloc[start_prev:start]),pre_post_sample_emg))
                    dant_fin.append(mct.signal.resample_by_interpolation(df_mov['Deltoid_Ant_filt_rect_norm'].iloc[end:end_foll],len(df_mov['Deltoid_Ant'].iloc[end:end_foll]),pre_post_sample_emg))

                    # tlng
                    dmed.append(mct.signal.resample_by_interpolation(df_mov['Deltoid_Med_filt_rect_norm'].iloc[start:end],len(df_mov['Deltoid_Med'].iloc[start:end]),1000))
                    dmed_deb.append(mct.signal.resample_by_interpolation(df_mov['Deltoid_Med_filt_rect_norm'].iloc[start_prev:start],len(df_mov['Deltoid_Med'].iloc[start_prev:start]),pre_post_sample_emg))
                    dmed_fin.append(mct.signal.resample_by_interpolation(df_mov['Deltoid_Med_filt_rect_norm'].iloc[end:end_foll],len(df_mov['Deltoid_Med'].iloc[end:end_foll]),pre_post_sample_emg))

                    # tlng
                    dpost.append(mct.signal.resample_by_interpolation(df_mov['Deltoip_Post_filt_rect_norm'].iloc[start:end],len(df_mov['Deltoip_Post'].iloc[start:end]),1000))
                    dpost_deb.append(mct.signal.resample_by_interpolation(df_mov['Deltoip_Post_filt_rect_norm'].iloc[start_prev:start],len(df_mov['Deltoip_Post'].iloc[start_prev:start]),pre_post_sample_emg))
                    dpost_fin.append(mct.signal.resample_by_interpolation(df_mov['Deltoip_Post_filt_rect_norm'].iloc[end:end_foll],len(df_mov['Deltoip_Post'].iloc[end:end_foll]),pre_post_sample_emg))


                except ZeroDivisionError:
                    print("Zero Division Error - Nothing have been appended")


            t, bic_av, bic_born_inf, bic_born_sup = mean_profiles(np.array(bic).T,durations, errors='SEM')
            t_deb, bic_deb_av, bic_deb_born_inf, bic_deb_born_sup = mean_profiles(np.array(bic_deb).T,pre_post_time, errors='SEM')
            t_fin, bic_fin_av, bic_fin_born_inf, bic_fin_born_sup = mean_profiles(np.array(bic_fin).T,pre_post_time, errors='SEM')
            
            _, bra_av, bra_born_inf, bra_born_sup = mean_profiles(np.array(bra).T,durations, errors='SEM')
            _, bra_deb_av, bra_deb_born_inf, bra_deb_born_sup = mean_profiles(np.array(bra_deb).T,pre_post_time, errors='SEM')
            _, bra_fin_av, bra_fin_born_inf, bra_fin_born_sup = mean_profiles(np.array(bra_fin).T,pre_post_time, errors='SEM')
            
            _, tlat_av, tlat_born_inf, tlat_born_sup = mean_profiles(np.array(tlat).T,durations, errors='SEM')
            _, tlat_deb_av, tlat_deb_born_inf, tlat_deb_born_sup = mean_profiles(np.array(tlat_deb).T,pre_post_time, errors='SEM')
            _, tlat_fin_av, tlat_fin_born_inf, tlat_fin_born_sup = mean_profiles(np.array(tlat_fin).T,pre_post_time, errors='SEM')

            _, tlng_av, tlng_born_inf, tlng_born_sup = mean_profiles(np.array(tlng).T,durations, errors='SEM')
            _, tlng_deb_av, tlng_deb_born_inf, tlng_deb_born_sup = mean_profiles(np.array(tlng_deb).T,pre_post_time, errors='SEM')
            _, tlng_fin_av, tlng_fin_born_inf, tlng_fin_born_sup = mean_profiles(np.array(tlng_fin).T,pre_post_time, errors='SEM')

            _, dant_av, dant_born_inf, dant_born_sup = mean_profiles(np.array(dant).T,durations, errors='SEM')
            _, dant_deb_av, dant_deb_born_inf, dant_deb_born_sup = mean_profiles(np.array(dant_deb).T,pre_post_time, errors='SEM')
            _, dant_fin_av, dant_fin_born_inf, dant_fin_born_sup = mean_profiles(np.array(dant_fin).T,pre_post_time, errors='SEM')

            _, dmed_av, dmed_born_inf, dmed_born_sup = mean_profiles(np.array(dmed).T,durations, errors='SEM')
            _, dmed_deb_av, dmed_deb_born_inf, dmed_deb_born_sup = mean_profiles(np.array(dmed_deb).T,pre_post_time, errors='SEM')
            _, dmed_fin_av, dmed_fin_born_inf, dmed_fin_born_sup = mean_profiles(np.array(dmed_fin).T,pre_post_time, errors='SEM')

            _, dpost_av, dpost_born_inf, dpost_born_sup = mean_profiles(np.array(dpost).T,durations, errors='SEM')
            _, dpost_deb_av, dpost_deb_born_inf, dpost_deb_born_sup = mean_profiles(np.array(dpost_deb).T,pre_post_time, errors='SEM')
            _, dpost_fin_av, dpost_fin_born_inf, dpost_fin_born_sup = mean_profiles(np.array(dpost_fin).T,pre_post_time, errors='SEM')
                        
            t = t+pre_post_time
            t_fin = t_fin+t[-1]

            bic_cond.append(bic_av)
            bic_sup_cond.append(bic_born_sup)
            bic_inf_cond.append(bic_born_inf)
            
            bra_cond.append(bra_av)
            bra_sup_cond.append(bra_born_sup)
            bra_inf_cond.append(bra_born_inf)
            
            tlng_cond.append(tlng_av)
            tlng_sup_cond.append(tlng_born_sup)
            tlng_inf_cond.append(tlng_born_inf)
            
            tlat_cond.append(tlat_av)
            tlat_sup_cond.append(tlat_born_sup)
            tlat_inf_cond.append(tlat_born_inf)

            dant_cond.append(dant_av)
            dant_sup_cond.append(dant_born_sup)
            dant_inf_cond.append(dant_born_inf)

            dmed_cond.append(dmed_av)
            dmed_sup_cond.append(dmed_born_sup)
            dmed_inf_cond.append(dmed_born_inf)

            dpost_cond.append(dpost_av)
            dpost_sup_cond.append(dpost_born_sup)
            dpost_inf_cond.append(dpost_born_inf)

            bic_deb_cond.append(bic_deb_av)
            bic_fin_cond.append(bic_fin_av)
            bra_deb_cond.append(bra_deb_av)
            bra_fin_cond.append(bra_fin_av)
            tlng_deb_cond.append(tlng_deb_av)
            tlng_fin_cond.append(tlng_fin_av)
            tlat_deb_cond.append(tlat_deb_av)
            tlat_fin_cond.append(tlat_fin_av)
            
            dant_deb_cond.append(dant_deb_av)
            dant_fin_cond.append(dant_fin_av)
            dmed_deb_cond.append(dmed_deb_av)
            dmed_fin_cond.append(dmed_fin_av)
            dpost_deb_cond.append(dpost_deb_av)
            dpost_fin_cond.append(dpost_fin_av)


        pos_all.update({
            cond : {
                "deb" : np.array(pos_deb_cond).T,
                "mvt" : np.array(pos_cond).T,
                "fin" : np.array(pos_fin_cond).T,
            }
        })
        vel_all.update({
            cond : {
                "deb" : np.array(vel_deb_cond).T,
                "mvt" : np.array(vel_cond).T,
                "fin" : np.array(vel_fin_cond).T,
            }
        })
        acc_all.update({
            cond : {
                "deb" : np.array(acc_deb_cond).T,
                "mvt" : np.array(acc_cond).T,
                "fin" : np.array(acc_fin_cond).T,
            }
        })
        bic_all.update({
            cond : {
                "deb" : np.array(bic_deb_cond).T,
                "mvt" : np.array(bic_cond).T,
                "fin" : np.array(bic_fin_cond).T,
                "sup" : np.array(bic_sup_cond).T,
                "inf" : np.array(bic_inf_cond).T,
            }
        })
        bra_all.update({
            cond : {
                "deb" : np.array(bra_deb_cond).T,
                "mvt" : np.array(bra_cond).T,
                "fin" : np.array(bra_fin_cond).T,
                "sup" : np.array(bra_sup_cond).T,
                "inf" : np.array(bra_inf_cond).T,
            }
        })
        tlat_all.update({
            cond : {
                "deb" : np.array(tlat_deb_cond).T,
                "mvt" : np.array(tlat_cond).T,
                "fin" : np.array(tlat_fin_cond).T,
                "sup" : np.array(tlat_sup_cond).T,
                "inf" : np.array(tlat_inf_cond).T,
            }
        })
        tlng_all.update({
            cond : {
                "deb" : np.array(tlng_deb_cond).T,
                "mvt" : np.array(tlng_cond).T,
                "fin" : np.array(tlng_fin_cond).T,
                "sup" : np.array(tlng_sup_cond).T,
                "inf" : np.array(tlng_inf_cond).T,
            }
        })

        dant_all.update({
            cond : {
                "deb" : np.array(dant_deb_cond).T,
                "mvt" : np.array(dant_cond).T,
                "fin" : np.array(dant_fin_cond).T,
                "sup" : np.array(dant_sup_cond).T,
                "inf" : np.array(dant_inf_cond).T,
            }
        })

        dmed_all.update({
            cond : {
                "deb" : np.array(dmed_deb_cond).T,
                "mvt" : np.array(dmed_cond).T,
                "fin" : np.array(dmed_fin_cond).T,
                "sup" : np.array(dmed_sup_cond).T,
                "inf" : np.array(dmed_inf_cond).T,
            }
        })

        dpost_all.update({
            cond : {
                "deb" : np.array(dpost_deb_cond).T,
                "mvt" : np.array(dpost_cond).T,
                "fin" : np.array(dpost_fin_cond).T,
                "sup" : np.array(dpost_sup_cond).T,
                "inf" : np.array(dpost_inf_cond).T,
            }
        })


    return {
        "pos" : pos_all,
        "vel" : vel_all,
        "acc" : acc_all,
        "bic" : bic_all,
        "bra" : bra_all,
        "tlat" : tlat_all,
        "tlng" : tlng_all,
        "dant" : dant_all,
        "dmed" : dmed_all,
        "dpost" : dpost_all
        }


def aggregate_all_trajectories_qual_one_subj(df_kin, df_emg, sample_rate_kin = 200, sample_rate_emg = 2000, unit = 'rad'):
    # one traj by subject for pos, vel, acc, flexors, extensors
    df_emg = df_emg.merge(df_kin[['subject', 'condition', 'block', 'movement', 'direction']], how = 'inner', on = ['subject', 'condition', 'block', 'movement'])
    print(df_emg.columns)
    print(df_kin.columns)
    print(df_emg)

    pos_all = {}
    vel_all = {}
    acc_all = {}
    bic_all = {}
    bra_all = {}
    tlat_all = {}
    tlng_all = {}

    pre_post_time = 0.5
    pre_post_sample = int(pre_post_time*sample_rate_kin)
    pre_post_sample_emg = int(pre_post_time*sample_rate_emg)

    for cond in df_kin.condition.unique():
        df_kin_ = df_kin.query("condition == @cond")
        df_emg_ = df_emg.query("condition == @cond")

        pos_cond = []
        vel_cond = []
        acc_cond = []
        pos_deb_cond = []
        pos_fin_cond = []
        vel_deb_cond = []
        vel_fin_cond = []
        acc_deb_cond = []
        acc_fin_cond = []

        bic_cond = []
        bra_cond = []
        tlng_cond = []
        tlat_cond = []
        bic_deb_cond = []
        bic_fin_cond = []
        bra_deb_cond = []
        bra_fin_cond = []
        tlng_deb_cond = []
        tlng_fin_cond = []
        tlat_deb_cond = []
        tlat_fin_cond = []
    
       
        # if df_kin_.subject.unique() != df_emg_.subject.unique():
        #     print("WARNING - df kin and df emg doesn't have the same subjects")
        for (subj, df_subj), (subj_emg, df_subj_emg) in zip(df_kin_.groupby('subject'), df_emg_.groupby('subject')):
            # print(subj, df_subj, subj_emg, df_subj_emg)
            print(subj)
             #Kinematic
            pos = []
            vel = []
            acc = []
            pos_deb = []
            pos_fin = []
            vel_deb = []
            vel_fin = []
            acc_deb = []
            acc_fin = []
            
            durations = []
            for mvt_id, movement_row in df_subj.groupby('movement'):
                with open(movement_row['file_name'].values[0], 'rb') as data_file:
                    data_movement = pickle.load(data_file)
                    df_mov = data_movement.get('df_mov')

                start = data_movement.get('bounds')[0]
                end = data_movement.get('bounds')[1]
                
                start_prev = np.clip(start -pre_post_sample,0, start) 
                end_foll = np.clip(end +pre_post_sample,0, len(df_mov)-1)
                
                if unit == 'deg':
                    df_mov['pos'] = df_mov['pos']*180/np.pi -90
                    df_mov['vel'] = df_mov['vel']*180/np.pi 
                    df_mov['acc'] = df_mov['acc']*180/np.pi 
                        
                durations = movement_row['MD'].values[0]
                # Position
                pos.append(mct.signal.resample_by_interpolation(df_mov['pos'].iloc[start:end],len(df_mov['pos'].iloc[start:end]),1000))
                pos_deb.append(mct.signal.resample_by_interpolation(df_mov['pos'].iloc[start_prev:start],len(df_mov['pos'].iloc[start_prev:start]),pre_post_sample))
                pos_fin.append(mct.signal.resample_by_interpolation(df_mov['pos'].iloc[end:end_foll],len(df_mov['pos'].iloc[end:end_foll]),pre_post_sample))
                
                # Velocity
                vel.append(mct.signal.resample_by_interpolation(df_mov['vel'].iloc[start:end],len(df_mov['vel'].iloc[start:end]),1000))
                vel_deb.append(mct.signal.resample_by_interpolation(df_mov['vel'].iloc[start_prev:start],len(df_mov['vel'].iloc[start_prev:start]),pre_post_sample))
                vel_fin.append(mct.signal.resample_by_interpolation(df_mov['vel'].iloc[end:end_foll],len(df_mov['vel'].iloc[end:end_foll]),pre_post_sample))
                
                #Acceleration
                acc.append(mct.signal.resample_by_interpolation(df_mov['acc'].iloc[start:end],len(df_mov['acc'].iloc[start:end]),1000))
                acc_deb.append(mct.signal.resample_by_interpolation(df_mov['acc'].iloc[start_prev:start],len(df_mov['acc'].iloc[start_prev:start]),pre_post_sample))
                acc_fin.append(mct.signal.resample_by_interpolation(df_mov['acc'].iloc[end:end_foll],len(df_mov['acc'].iloc[end:end_foll]),pre_post_sample))


                # durations = np.array(durations).mean()
                t, pos_av, pos_born_inf, pos_born_sup = mean_profiles(np.array(pos).T,durations)
                t_deb, pos_deb_av, pos_deb_born_inf, pos_deb_born_sup = mean_profiles(np.array(pos_deb).T,pre_post_time)
                t_fin, pos_fin_av, pos_fin_born_inf, pos_fin_born_sup = mean_profiles(np.array(pos_fin).T,pre_post_time)
                
                _, vel_av, vel_born_inf, vel_born_sup = mean_profiles(np.array(vel).T,durations)
                _, vel_deb_av, vel_deb_born_inf, vel_deb_born_sup = mean_profiles(np.array(vel_deb).T,pre_post_time)
                _, vel_fin_av, vel_fin_born_inf, vel_fin_born_sup = mean_profiles(np.array(vel_fin).T,pre_post_time)
                
                _, acc_av, acc_born_inf, acc_born_sup = mean_profiles(np.array(acc).T,durations)
                _, acc_deb_av, acc_deb_born_inf, acc_deb_born_sup = mean_profiles(np.array(acc_deb).T,pre_post_time)
                _, acc_fin_av, acc_fin_born_inf, acc_fin_born_sup = mean_profiles(np.array(acc_fin).T,pre_post_time)
                            
                t = t+pre_post_time
                t_fin = t_fin+t[-1]

                pos_cond.append(pos_av)
                vel_cond.append(vel_av)
                acc_cond.append(acc_av)
                pos_deb_cond.append(pos_deb_av)
                pos_fin_cond.append(pos_fin_av)
                vel_deb_cond.append(vel_deb_av)
                vel_fin_cond.append(vel_fin_av)
                acc_deb_cond.append(acc_deb_av)
                acc_fin_cond.append(acc_fin_av)
            

            #emg

            bic = []
            bra = []
            tlng = []
            tlat = []

            bic_deb = []
            bic_fin = []
            bra_deb = []
            bra_fin = []
            tlng_deb = []
            tlng_fin = []
            tlat_deb = []
            tlat_fin = []
            

            for mvt_id, movement_row in df_subj_emg.groupby('movement'):
                try:
                    with open(movement_row['file_name'].values[0], 'rb') as data_file:
                        data_movement = pickle.load(data_file)
                        df_mov = data_movement.get('df_mov')

                    start = data_movement.get('bounds')[0]
                    end = data_movement.get('bounds')[1]
                    
                    start_prev = np.clip(start -pre_post_sample_emg,0, start) 
                    end_foll = np.clip(end +pre_post_sample_emg,0, len(df_mov)-1)
                    
                    # bic
                    bic.append(mct.signal.resample_by_interpolation(df_mov['bic_filt_rect_norm'].iloc[start:end],len(df_mov['bic'].iloc[start:end]),1000))
                    bic_deb.append(mct.signal.resample_by_interpolation(df_mov['bic_filt_rect_norm'].iloc[start_prev:start],len(df_mov['bic'].iloc[start_prev:start]),pre_post_sample_emg))
                    bic_fin.append(mct.signal.resample_by_interpolation(df_mov['bic_filt_rect_norm'].iloc[end:end_foll],len(df_mov['bic'].iloc[end:end_foll]),pre_post_sample_emg))
                    
                    # bra
                    bra.append(mct.signal.resample_by_interpolation(df_mov['bra_filt_rect_norm'].iloc[start:end],len(df_mov['bra'].iloc[start:end]),1000))
                    bra_deb.append(mct.signal.resample_by_interpolation(df_mov['bra_filt_rect_norm'].iloc[start_prev:start],len(df_mov['bra'].iloc[start_prev:start]),pre_post_sample_emg))
                    bra_fin.append(mct.signal.resample_by_interpolation(df_mov['bra_filt_rect_norm'].iloc[end:end_foll],len(df_mov['bra'].iloc[end:end_foll]),pre_post_sample_emg))
                    
                    # tlat
                    tlat.append(mct.signal.resample_by_interpolation(df_mov['tlat_filt_rect_norm'].iloc[start:end],len(df_mov['tlat'].iloc[start:end]),1000))
                    tlat_deb.append(mct.signal.resample_by_interpolation(df_mov['tlat_filt_rect_norm'].iloc[start_prev:start],len(df_mov['tlat'].iloc[start_prev:start]),pre_post_sample_emg))
                    tlat_fin.append(mct.signal.resample_by_interpolation(df_mov['tlat_filt_rect_norm'].iloc[end:end_foll],len(df_mov['tlat'].iloc[end:end_foll]),pre_post_sample_emg))

                    # tlng
                    tlng.append(mct.signal.resample_by_interpolation(df_mov['tlng_filt_rect_norm'].iloc[start:end],len(df_mov['tlng'].iloc[start:end]),1000))
                    tlng_deb.append(mct.signal.resample_by_interpolation(df_mov['tlng_filt_rect_norm'].iloc[start_prev:start],len(df_mov['tlng'].iloc[start_prev:start]),pre_post_sample_emg))
                    tlng_fin.append(mct.signal.resample_by_interpolation(df_mov['tlng_filt_rect_norm'].iloc[end:end_foll],len(df_mov['tlng'].iloc[end:end_foll]),pre_post_sample_emg))

                except ZeroDivisionError:
                    print("Zero Division Error - Nothing have been appended")

            
                t, bic_av, bic_born_inf, bic_born_sup = mean_profiles(np.array(bic).T,durations, errors='SEM')
                t_deb, bic_deb_av, bic_deb_born_inf, bic_deb_born_sup = mean_profiles(np.array(bic_deb).T,pre_post_time, errors='SEM')
                t_fin, bic_fin_av, bic_fin_born_inf, bic_fin_born_sup = mean_profiles(np.array(bic_fin).T,pre_post_time, errors='SEM')
                
                _, bra_av, bra_born_inf, bra_born_sup = mean_profiles(np.array(bra).T,durations, errors='SEM')
                _, bra_deb_av, bra_deb_born_inf, bra_deb_born_sup = mean_profiles(np.array(bra_deb).T,pre_post_time, errors='SEM')
                _, bra_fin_av, bra_fin_born_inf, bra_fin_born_sup = mean_profiles(np.array(bra_fin).T,pre_post_time, errors='SEM')
                
                _, tlat_av, tlat_born_inf, tlat_born_sup = mean_profiles(np.array(tlat).T,durations, errors='SEM')
                _, tlat_deb_av, tlat_deb_born_inf, tlat_deb_born_sup = mean_profiles(np.array(tlat_deb).T,pre_post_time, errors='SEM')
                _, tlat_fin_av, tlat_fin_born_inf, tlat_fin_born_sup = mean_profiles(np.array(tlat_fin).T,pre_post_time, errors='SEM')

                _, tlng_av, tlng_born_inf, tlng_born_sup = mean_profiles(np.array(tlng).T,durations, errors='SEM')
                _, tlng_deb_av, tlng_deb_born_inf, tlng_deb_born_sup = mean_profiles(np.array(tlng_deb).T,pre_post_time, errors='SEM')
                _, tlng_fin_av, tlng_fin_born_inf, tlng_fin_born_sup = mean_profiles(np.array(tlng_fin).T,pre_post_time, errors='SEM')
                            
                t = t+pre_post_time
                t_fin = t_fin+t[-1]

                bic_cond.append(bic_av)
                bra_cond.append(bra_av)
                tlng_cond.append(tlng_av)
                tlat_cond.append(tlat_av)

                bic_deb_cond.append(bic_deb_av)
                bic_fin_cond.append(bic_fin_av)
                bra_deb_cond.append(bra_deb_av)
                bra_fin_cond.append(bra_fin_av)
                tlng_deb_cond.append(tlng_deb_av)
                tlng_fin_cond.append(tlng_fin_av)
                tlat_deb_cond.append(tlat_deb_av)
                tlat_fin_cond.append(tlat_fin_av)


        pos_all.update({
            cond : {
                "deb" : np.array(pos_deb_cond).T,
                "mvt" : np.array(pos_cond).T,
                "fin" : np.array(pos_fin_cond).T,
            }
        })
        vel_all.update({
            cond : {
                "deb" : np.array(vel_deb_cond).T,
                "mvt" : np.array(vel_cond).T,
                "fin" : np.array(vel_fin_cond).T,
            }
        })
        acc_all.update({
            cond : {
                "deb" : np.array(acc_deb_cond).T,
                "mvt" : np.array(acc_cond).T,
                "fin" : np.array(acc_fin_cond).T,
            }
        })
        bic_all.update({
            cond : {
                "deb" : np.array(bic_deb_cond).T,
                "mvt" : np.array(bic_cond).T,
                "fin" : np.array(bic_fin_cond).T,
            }
        })
        bra_all.update({
            cond : {
                "deb" : np.array(bra_deb_cond).T,
                "mvt" : np.array(bra_cond).T,
                "fin" : np.array(bra_fin_cond).T,
            }
        })
        tlat_all.update({
            cond : {
                "deb" : np.array(tlat_deb_cond).T,
                "mvt" : np.array(tlat_cond).T,
                "fin" : np.array(tlat_fin_cond).T,
            }
        })
        tlng_all.update({
            cond : {
                "deb" : np.array(tlng_deb_cond).T,
                "mvt" : np.array(tlng_cond).T,
                "fin" : np.array(tlng_fin_cond).T,
            }
        })


    return {
        "pos" : pos_all,
        "vel" : vel_all,
        "acc" : acc_all,
        "bic" : bic_all,
        "bra" : bra_all,
        "tlat" : tlat_all,
        "tlng" : tlng_all,
        }


def one_subj_description(SUBJ, DIR, df):
    # based on rob data
    df_ = df.query("subject == @SUBJ & direction == @DIR")
    fig, axes = plt.subplots(5,3, figsize = bt.cm2inch(12,17), sharex=True, sharey='row', frameon = False, dpi = 100)
    pre_post_time = 0.5
    pre_post_sample = int(pre_post_time*config.spec.get("sample_rate_rob"))
    cond_count = 0
    desired_cond_order = {
        '1G':0,
        '0G':1,
        '-1G':2,
    }
    for cond, df_cond in df_.groupby('condition'):
        pos = []
        vel = []
        acc = []
        tau = []
        tau_theo = []
        fz = []
        fz_theo = []
        
        pos_deb = []
        pos_fin = []
        vel_deb = []
        vel_fin = []
        acc_deb = []
        acc_fin = []
        tau_deb = []
        tau_fin = []
        tau_theo_deb = []
        tau_theo_fin = []

        fz_deb = []
        fz_fin = []

        fz_theo_deb = []
        fz_theo_fin = []
        
        durations = []
        for mvt_id, movement_row in df_cond.groupby('movement'):
            with open(movement_row['file_name'].values[0], 'rb') as data_file:
                data_movement = pickle.load(data_file)
                df_mov = data_movement.get('df_mov')

            start = data_movement.get('bounds')[0]
            end = data_movement.get('bounds')[1]
            
            start_prev = np.clip(start -pre_post_sample,0, start) 
            end_foll = np.clip(end +pre_post_sample,0, len(df_mov)-1)
            
            
                       
            durations.append(movement_row['MD'].values[0])
            # Position
            pos.append(mct.signal.resample_by_interpolation(df_mov['pos'].iloc[start:end],len(df_mov['pos'].iloc[start:end]),1000))
            pos_deb.append(mct.signal.resample_by_interpolation(df_mov['pos'].iloc[start_prev:start],len(df_mov['pos'].iloc[start_prev:start]),pre_post_sample))
            pos_fin.append(mct.signal.resample_by_interpolation(df_mov['pos'].iloc[end:end_foll],len(df_mov['pos'].iloc[end:end_foll]),pre_post_sample))
            
            # Velocity
            vel.append(mct.signal.resample_by_interpolation(df_mov['vel'].iloc[start:end],len(df_mov['vel'].iloc[start:end]),1000))
            vel_deb.append(mct.signal.resample_by_interpolation(df_mov['vel'].iloc[start_prev:start],len(df_mov['vel'].iloc[start_prev:start]),pre_post_sample))
            vel_fin.append(mct.signal.resample_by_interpolation(df_mov['vel'].iloc[end:end_foll],len(df_mov['vel'].iloc[end:end_foll]),pre_post_sample))
            
            #Acceleration
            acc.append(mct.signal.resample_by_interpolation(df_mov['acc'].iloc[start:end],len(df_mov['acc'].iloc[start:end]),1000))
            acc_deb.append(mct.signal.resample_by_interpolation(df_mov['acc'].iloc[start_prev:start],len(df_mov['acc'].iloc[start_prev:start]),pre_post_sample))
            acc_fin.append(mct.signal.resample_by_interpolation(df_mov['acc'].iloc[end:end_foll],len(df_mov['acc'].iloc[end:end_foll]),pre_post_sample))

            #fz
            fz.append(mct.signal.resample_by_interpolation(df_mov['Fz'].iloc[start:end],len(df_mov['Fz'].iloc[start:end]),1000))
            fz_deb.append(mct.signal.resample_by_interpolation(df_mov['Fz'].iloc[start_prev:start],len(df_mov['Fz'].iloc[start_prev:start]),pre_post_sample))
            fz_fin.append(mct.signal.resample_by_interpolation(df_mov['Fz'].iloc[end:end_foll],len(df_mov['Fz'].iloc[end:end_foll]),pre_post_sample))

            #fz_theo
            fz_theo.append(mct.signal.resample_by_interpolation(df_mov['Fz_theo'].iloc[start:end],len(df_mov['Fz_theo'].iloc[start:end]),1000))
            fz_theo_deb.append(mct.signal.resample_by_interpolation(df_mov['Fz_theo'].iloc[start_prev:start],len(df_mov['Fz_theo'].iloc[start_prev:start]),pre_post_sample))
            fz_theo_fin.append(mct.signal.resample_by_interpolation(df_mov['Fz_theo'].iloc[end:end_foll],len(df_mov['Fz_theo'].iloc[end:end_foll]),pre_post_sample))

            #tau_i
            tau.append(mct.signal.resample_by_interpolation(df_mov['tau_i'].iloc[start:end],len(df_mov['tau_i'].iloc[start:end]),1000))
            tau_deb.append(mct.signal.resample_by_interpolation(df_mov['tau_i'].iloc[start_prev:start],len(df_mov['tau_i'].iloc[start_prev:start]),pre_post_sample))
            tau_fin.append(mct.signal.resample_by_interpolation(df_mov['tau_i'].iloc[end:end_foll],len(df_mov['tau_i'].iloc[end:end_foll]),pre_post_sample))

            #tau_i_theorique
            tau_theo.append(mct.signal.resample_by_interpolation(df_mov['tau_i_theo'].iloc[start:end],len(df_mov['tau_i_theo'].iloc[start:end]),1000))
            tau_theo_deb.append(mct.signal.resample_by_interpolation(df_mov['tau_i_theo'].iloc[start_prev:start],len(df_mov['tau_i_theo'].iloc[start_prev:start]),pre_post_sample))
            tau_theo_fin.append(mct.signal.resample_by_interpolation(df_mov['tau_i_theo'].iloc[end:end_foll],len(df_mov['tau_i_theo'].iloc[end:end_foll]),pre_post_sample))
            
            
           

        durations = np.array(durations).mean()
        t, pos_av, pos_born_inf, pos_born_sup = mean_profiles(np.array(pos).T,durations)
        t_deb, pos_deb_av, pos_deb_born_inf, pos_deb_born_sup = mean_profiles(np.array(pos_deb).T,pre_post_time)
        t_fin, pos_fin_av, pos_fin_born_inf, pos_fin_born_sup = mean_profiles(np.array(pos_fin).T,pre_post_time)
        
        _, vel_av, vel_born_inf, vel_born_sup = mean_profiles(np.array(vel).T,durations)
        _, vel_deb_av, vel_deb_born_inf, vel_deb_born_sup = mean_profiles(np.array(vel_deb).T,pre_post_time)
        _, vel_fin_av, vel_fin_born_inf, vel_fin_born_sup = mean_profiles(np.array(vel_fin).T,pre_post_time)
        
        _, acc_av, acc_born_inf, acc_born_sup = mean_profiles(np.array(acc).T,durations)
        _, acc_deb_av, acc_deb_born_inf, acc_deb_born_sup = mean_profiles(np.array(acc_deb).T,pre_post_time)
        _, acc_fin_av, acc_fin_born_inf, acc_fin_born_sup = mean_profiles(np.array(acc_fin).T,pre_post_time)

        _, fz_av, fz_born_inf, fz_born_sup = mean_profiles(np.array(fz).T,durations)
        _, fz_deb_av, fz_deb_born_inf, fz_deb_born_sup = mean_profiles(np.array(fz_deb).T,pre_post_time)
        _, fz_fin_av, fz_fin_born_inf, fz_fin_born_sup = mean_profiles(np.array(fz_fin).T,pre_post_time)
        
        _, fz_theo_av, fz_theo_born_inf, fz_theo_born_sup = mean_profiles(np.array(fz_theo).T,durations)
        _, fz_theo_deb_av, fz_theo_deb_born_inf, fz_theo_deb_born_sup = mean_profiles(np.array(fz_theo_deb).T,pre_post_time)
        _, fz_theo_fin_av, fz_theo_fin_born_inf, fz_theo_fin_born_sup = mean_profiles(np.array(fz_theo_fin).T,pre_post_time)

        _, tau_av, tau_born_inf, tau_born_sup = mean_profiles(np.array(tau).T,durations)
        _, tau_deb_av, tau_deb_born_inf, tau_deb_born_sup = mean_profiles(np.array(tau_deb).T,pre_post_time)
        _, tau_fin_av, tau_fin_born_inf, tau_fin_born_sup = mean_profiles(np.array(tau_fin).T,pre_post_time)

        _, tau_theo_av, tau_theo_born_inf, tau_theo_born_sup = mean_profiles(np.array(tau_theo).T,durations)
        _, tau_theo_deb_av, tau_theo_deb_born_inf, tau_theo_deb_born_sup = mean_profiles(np.array(tau_theo_deb).T,pre_post_time)
        _, tau_theo_fin_av, tau_theo_fin_born_inf, tau_theo_fin_born_sup = mean_profiles(np.array(tau_theo_fin).T,pre_post_time)
        
        
                        
        t = t+pre_post_time
        t_fin = t_fin+t[-1]
        

        axes[0,desired_cond_order.get(cond)].plot(t,pos_av, c = 'k')
        axes[0,desired_cond_order.get(cond)].plot(t_deb,pos_deb_av,c = 'darkgrey')# colors.get(cond)
        axes[0,desired_cond_order.get(cond)].plot(t_fin,pos_fin_av,c = 'darkgrey')
        axes[0,desired_cond_order.get(cond)].fill_between(t,pos_born_inf, pos_born_sup,  color = 'lightsteelblue')
        
        axes[1,desired_cond_order.get(cond)].plot(t,vel_av, c = 'k')
        axes[1,desired_cond_order.get(cond)].plot(t_deb,vel_deb_av,c = 'darkgrey')
        axes[1,desired_cond_order.get(cond)].plot(t_fin,vel_fin_av,c = 'darkgrey')
        axes[1,desired_cond_order.get(cond)].fill_between(t,vel_born_inf, vel_born_sup,  color = 'lightsteelblue')
        
        
        axes[2,desired_cond_order.get(cond)].plot(t,acc_av, c = 'k')
        axes[2,desired_cond_order.get(cond)].plot(t_deb,acc_deb_av,c = 'darkgrey')
        axes[2,desired_cond_order.get(cond)].plot(t_fin,acc_fin_av,c = 'darkgrey')
        axes[2,desired_cond_order.get(cond)].fill_between(t,acc_born_inf, acc_born_sup,  color = 'lightsteelblue')

        axes[3,desired_cond_order.get(cond)].plot(t,fz_av, c = 'k')
        axes[3,desired_cond_order.get(cond)].plot(t_deb,fz_deb_av,c = 'darkgrey')
        axes[3,desired_cond_order.get(cond)].plot(t_fin,fz_fin_av,c = 'darkgrey')
        axes[3,desired_cond_order.get(cond)].fill_between(t,fz_born_inf, fz_born_sup,  color = 'lightsteelblue')

        axes[3,desired_cond_order.get(cond)].plot(t,fz_av, c = 'k')
        axes[3,desired_cond_order.get(cond)].plot(t_deb,fz_deb_av,c = 'darkgrey')
        axes[3,desired_cond_order.get(cond)].plot(t_fin,fz_fin_av,c = 'darkgrey')
        axes[3,desired_cond_order.get(cond)].fill_between(t,fz_born_inf, fz_born_sup,  color = 'lightsteelblue')

        axes[3,desired_cond_order.get(cond)].plot(t,fz_theo_av, c = 'tab:red')
        axes[3,desired_cond_order.get(cond)].plot(t_deb,fz_theo_deb_av,c = 'lightcoral')
        axes[3,desired_cond_order.get(cond)].plot(t_fin,fz_theo_fin_av,c = 'lightcoral')
        axes[3,desired_cond_order.get(cond)].fill_between(t,fz_theo_born_inf, fz_theo_born_sup,  color = 'mistyrose')

        axes[4,desired_cond_order.get(cond)].plot(t,tau_av, c = 'k')
        axes[4,desired_cond_order.get(cond)].plot(t_deb,tau_deb_av,c = 'darkgrey')
        axes[4,desired_cond_order.get(cond)].plot(t_fin,tau_fin_av,c = 'darkgrey')
        axes[4,desired_cond_order.get(cond)].fill_between(t,tau_born_inf, tau_born_sup,  color = 'lightsteelblue')

        axes[4,desired_cond_order.get(cond)].plot(t,tau_theo_av, c = 'tab:red')
        axes[4,desired_cond_order.get(cond)].plot(t_deb,tau_theo_deb_av,c = 'lightcoral')
        axes[4,desired_cond_order.get(cond)].plot(t_fin,tau_theo_fin_av,c = 'lightcoral')
        axes[4,desired_cond_order.get(cond)].fill_between(t,tau_theo_born_inf, tau_theo_born_sup,  color = 'mistyrose')
        

        
        
        cond_count +=1

        
    
    axes[0,0].set_ylabel(r'Pos. ($rad$)')
    axes[1,0].set_ylabel(r'Vel. ($rad.s^{-1}$)')
    axes[2,0].set_ylabel(r'Acc. ($rad.s^{-2}$)')
    axes[3,0].set_ylabel(r'Fz ($N$)')
    axes[4,0].set_ylabel(r'$\tau_{i}$ ($N.m$)')
    # axes[3,0].set_ylabel(r'$\tau_{i}$ ($N.m$)')
    
    
#     axes[0,0].set_title(r'\textbf{Ctrl}')
#     axes[0,1].set_title(r'\textbf{FC}')
#     axes[0,2].set_title(r'\textbf{PCFr}')
#     axes[0,3].set_title(r'\textbf{CLPC}')
    inv_map = {v: k for k, v in desired_cond_order.items()}
    axes[0,0].set_title('{}'.format(inv_map.get(0)))
    axes[0,1].set_title('{}'.format(inv_map.get(1)))
    axes[0,2].set_title('{}'.format(inv_map.get(2)))
    
    fig.text(0.53, 0.02, r'Time ($s$)', ha='center')
    
    for axs in axes:
        for ax in axs:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlim(len(pos)/config.spec.get("sample_rate_rob"))
    #print(str([k for k,v in subj_map.items() if v == SUBJ]))
    fig.align_ylabels(axes[:, 0])
    # for axs in axes[3:5,:]:
    #     for ax in axs:
    #         ax.set_ylim(bottom = -0.005)
    #plt.tight_layout()
    return fig


def one_subj_description_emg(SUBJ, DIR, df_kin, df_emg):
    # based on qualisys data

    df_emg = df_emg.merge(df_kin[['subject', 'condition', 'block', 'movement', 'direction']], how = 'inner', on = ['subject', 'condition', 'block', 'movement'])
    print(df_emg.columns)
    print(df_kin.columns)
    df_kin_ = df_kin.query("subject == @SUBJ & direction == @DIR")
    df_emg_ = df_emg.query("subject == @SUBJ & direction == @DIR")


    fig, axes = plt.subplots(5,3, figsize = bt.cm2inch(20,30), sharex=True, sharey='row', frameon = False, dpi = 300)
    pre_post_time = 0.5
    pre_post_sample = int(pre_post_time*config.spec.get("sample_rate_kin"))
    pre_post_sample_emg = int(pre_post_time*config.spec.get("sample_rate_emg"))
    cond_count = 0
    desired_cond_order = {
        '1G':0,
        '0G':1,
        '-1G':2,
    }

    #Kinematic
    for cond, df_cond in df_kin_.groupby('condition'):
        pos = []
        vel = []
        acc = []
        
        pos_deb = []
        pos_fin = []
        vel_deb = []
        vel_fin = []
        acc_deb = []
        acc_fin = []
        
        durations = []

        for mvt_id, movement_row in df_cond.groupby('movement'):
            with open(movement_row['file_name'].values[0], 'rb') as data_file:
                data_movement = pickle.load(data_file)
                df_mov = data_movement.get('df_mov')

            start = data_movement.get('bounds')[0]
            end = data_movement.get('bounds')[1]
            
            start_prev = np.clip(start -pre_post_sample,0, start) 
            end_foll = np.clip(end +pre_post_sample,0, len(df_mov)-1)
            
                       
            durations.append(movement_row['MD'].values[0])
            # Position
            pos.append(mct.signal.resample_by_interpolation(df_mov['pos'].iloc[start:end],len(df_mov['pos'].iloc[start:end]),1000))
            pos_deb.append(mct.signal.resample_by_interpolation(df_mov['pos'].iloc[start_prev:start],len(df_mov['pos'].iloc[start_prev:start]),pre_post_sample))
            pos_fin.append(mct.signal.resample_by_interpolation(df_mov['pos'].iloc[end:end_foll],len(df_mov['pos'].iloc[end:end_foll]),pre_post_sample))
            
            # Velocity
            vel.append(mct.signal.resample_by_interpolation(df_mov['vel'].iloc[start:end],len(df_mov['vel'].iloc[start:end]),1000))
            vel_deb.append(mct.signal.resample_by_interpolation(df_mov['vel'].iloc[start_prev:start],len(df_mov['vel'].iloc[start_prev:start]),pre_post_sample))
            vel_fin.append(mct.signal.resample_by_interpolation(df_mov['vel'].iloc[end:end_foll],len(df_mov['vel'].iloc[end:end_foll]),pre_post_sample))
            
            #Acceleration
            acc.append(mct.signal.resample_by_interpolation(df_mov['acc'].iloc[start:end],len(df_mov['acc'].iloc[start:end]),1000))
            acc_deb.append(mct.signal.resample_by_interpolation(df_mov['acc'].iloc[start_prev:start],len(df_mov['acc'].iloc[start_prev:start]),pre_post_sample))
            acc_fin.append(mct.signal.resample_by_interpolation(df_mov['acc'].iloc[end:end_foll],len(df_mov['acc'].iloc[end:end_foll]),pre_post_sample))


        durations = np.array(durations).mean()
        t, pos_av, pos_born_inf, pos_born_sup = mean_profiles(np.array(pos).T,durations)
        t_deb, pos_deb_av, pos_deb_born_inf, pos_deb_born_sup = mean_profiles(np.array(pos_deb).T,pre_post_time)
        t_fin, pos_fin_av, pos_fin_born_inf, pos_fin_born_sup = mean_profiles(np.array(pos_fin).T,pre_post_time)
        
        _, vel_av, vel_born_inf, vel_born_sup = mean_profiles(np.array(vel).T,durations)
        _, vel_deb_av, vel_deb_born_inf, vel_deb_born_sup = mean_profiles(np.array(vel_deb).T,pre_post_time)
        _, vel_fin_av, vel_fin_born_inf, vel_fin_born_sup = mean_profiles(np.array(vel_fin).T,pre_post_time)
        
        _, acc_av, acc_born_inf, acc_born_sup = mean_profiles(np.array(acc).T,durations)
        _, acc_deb_av, acc_deb_born_inf, acc_deb_born_sup = mean_profiles(np.array(acc_deb).T,pre_post_time)
        _, acc_fin_av, acc_fin_born_inf, acc_fin_born_sup = mean_profiles(np.array(acc_fin).T,pre_post_time)
                    
        t = t+pre_post_time
        t_fin = t_fin+t[-1]
        
        axes[0,desired_cond_order.get(cond)].plot(t,pos_av, c = 'k')
        axes[0,desired_cond_order.get(cond)].plot(t_deb,pos_deb_av,c = 'darkgrey')# colors.get(cond)
        axes[0,desired_cond_order.get(cond)].plot(t_fin,pos_fin_av,c = 'darkgrey')
        axes[0,desired_cond_order.get(cond)].fill_between(t,pos_born_inf, pos_born_sup,  color = 'lightsteelblue')
        
        axes[1,desired_cond_order.get(cond)].plot(t,vel_av, c = 'k')
        axes[1,desired_cond_order.get(cond)].plot(t_deb,vel_deb_av,c = 'darkgrey')
        axes[1,desired_cond_order.get(cond)].plot(t_fin,vel_fin_av,c = 'darkgrey')
        axes[1,desired_cond_order.get(cond)].fill_between(t,vel_born_inf, vel_born_sup,  color = 'lightsteelblue')
        
        
        axes[2,desired_cond_order.get(cond)].plot(t,acc_av, c = 'k')
        axes[2,desired_cond_order.get(cond)].plot(t_deb,acc_deb_av,c = 'darkgrey')
        axes[2,desired_cond_order.get(cond)].plot(t_fin,acc_fin_av,c = 'darkgrey')
        axes[2,desired_cond_order.get(cond)].fill_between(t,acc_born_inf, acc_born_sup,  color = 'lightsteelblue')

        
        cond_count +=1

    #Emgs
    cond_count = 0
    for cond, df_cond in df_emg_.groupby('condition'):
        bic = []
        bra = []
        tlng = []
        tlat = []

        bic_deb = []
        bic_fin = []
        bra_deb = []
        bra_fin = []
        tlng_deb = []
        tlng_fin = []
        tlat_deb = []
        tlat_fin = []
        

        for mvt_id, movement_row in df_cond.groupby('movement'):
            with open(movement_row['file_name'].values[0], 'rb') as data_file:
                data_movement = pickle.load(data_file)
                df_mov = data_movement.get('df_mov')

            start = data_movement.get('bounds')[0]
            end = data_movement.get('bounds')[1]
            
            start_prev = np.clip(start -pre_post_sample_emg,0, start) 
            end_foll = np.clip(end +pre_post_sample_emg,0, len(df_mov)-1)
            
            # bic
            bic.append(mct.signal.resample_by_interpolation(df_mov['bic_filt_rect_norm'].iloc[start:end],len(df_mov['bic'].iloc[start:end]),1000))
            bic_deb.append(mct.signal.resample_by_interpolation(df_mov['bic_filt_rect_norm'].iloc[start_prev:start],len(df_mov['bic'].iloc[start_prev:start]),pre_post_sample_emg))
            bic_fin.append(mct.signal.resample_by_interpolation(df_mov['bic_filt_rect_norm'].iloc[end:end_foll],len(df_mov['bic'].iloc[end:end_foll]),pre_post_sample_emg))
            
            # bra
            bra.append(mct.signal.resample_by_interpolation(df_mov['bra_filt_rect_norm'].iloc[start:end],len(df_mov['bra'].iloc[start:end]),1000))
            bra_deb.append(mct.signal.resample_by_interpolation(df_mov['bra_filt_rect_norm'].iloc[start_prev:start],len(df_mov['bra'].iloc[start_prev:start]),pre_post_sample_emg))
            bra_fin.append(mct.signal.resample_by_interpolation(df_mov['bra_filt_rect_norm'].iloc[end:end_foll],len(df_mov['bra'].iloc[end:end_foll]),pre_post_sample_emg))
            
            # tlat
            tlat.append(mct.signal.resample_by_interpolation(df_mov['tlat_filt_rect_norm'].iloc[start:end],len(df_mov['tlat'].iloc[start:end]),1000))
            tlat_deb.append(mct.signal.resample_by_interpolation(df_mov['tlat_filt_rect_norm'].iloc[start_prev:start],len(df_mov['tlat'].iloc[start_prev:start]),pre_post_sample_emg))
            tlat_fin.append(mct.signal.resample_by_interpolation(df_mov['tlat_filt_rect_norm'].iloc[end:end_foll],len(df_mov['tlat'].iloc[end:end_foll]),pre_post_sample_emg))

            # tlng
            tlng.append(mct.signal.resample_by_interpolation(df_mov['tlng_filt_rect_norm'].iloc[start:end],len(df_mov['tlng'].iloc[start:end]),1000))
            tlng_deb.append(mct.signal.resample_by_interpolation(df_mov['tlng_filt_rect_norm'].iloc[start_prev:start],len(df_mov['tlng'].iloc[start_prev:start]),pre_post_sample_emg))
            tlng_fin.append(mct.signal.resample_by_interpolation(df_mov['tlng_filt_rect_norm'].iloc[end:end_foll],len(df_mov['tlng'].iloc[end:end_foll]),pre_post_sample_emg))


        t, bic_av, bic_born_inf, bic_born_sup = mean_profiles(np.array(bic).T,durations, errors='SEM')
        t_deb, bic_deb_av, bic_deb_born_inf, bic_deb_born_sup = mean_profiles(np.array(bic_deb).T,pre_post_time, errors='SEM')
        t_fin, bic_fin_av, bic_fin_born_inf, bic_fin_born_sup = mean_profiles(np.array(bic_fin).T,pre_post_time, errors='SEM')
        
        _, bra_av, bra_born_inf, bra_born_sup = mean_profiles(np.array(bra).T,durations, errors='SEM')
        _, bra_deb_av, bra_deb_born_inf, bra_deb_born_sup = mean_profiles(np.array(bra_deb).T,pre_post_time, errors='SEM')
        _, bra_fin_av, bra_fin_born_inf, bra_fin_born_sup = mean_profiles(np.array(bra_fin).T,pre_post_time, errors='SEM')
        
        _, tlat_av, tlat_born_inf, tlat_born_sup = mean_profiles(np.array(tlat).T,durations, errors='SEM')
        _, tlat_deb_av, tlat_deb_born_inf, tlat_deb_born_sup = mean_profiles(np.array(tlat_deb).T,pre_post_time, errors='SEM')
        _, tlat_fin_av, tlat_fin_born_inf, tlat_fin_born_sup = mean_profiles(np.array(tlat_fin).T,pre_post_time, errors='SEM')

        _, tlng_av, tlng_born_inf, tlng_born_sup = mean_profiles(np.array(tlng).T,durations, errors='SEM')
        _, tlng_deb_av, tlng_deb_born_inf, tlng_deb_born_sup = mean_profiles(np.array(tlng_deb).T,pre_post_time, errors='SEM')
        _, tlng_fin_av, tlng_fin_born_inf, tlng_fin_born_sup = mean_profiles(np.array(tlng_fin).T,pre_post_time, errors='SEM')
                    
        t = t+pre_post_time
        t_fin = t_fin+t[-1]

        # mono articulaire sur axes 4
        # bi-articulaires sur axes 3
        
        axes[3,desired_cond_order.get(cond)].plot(t,bic_av, c = 'tab:green', label = 'Biceps')
        axes[3,desired_cond_order.get(cond)].plot(t_deb,bic_deb_av,c = 'lightgreen')# colors.get(cond)
        axes[3,desired_cond_order.get(cond)].plot(t_fin,bic_fin_av,c = 'lightgreen')
        axes[3,desired_cond_order.get(cond)].fill_between(t,bic_born_inf, bic_born_sup,  color = 'darkseagreen')
        
        axes[4,desired_cond_order.get(cond)].plot(t,bra_av, c = 'tab:green', label = 'Brachio-radialis')
        axes[4,desired_cond_order.get(cond)].plot(t_deb,bra_deb_av,c = 'lightgreen')
        axes[4,desired_cond_order.get(cond)].plot(t_fin,bra_fin_av,c = 'lightgreen')
        axes[4,desired_cond_order.get(cond)].fill_between(t,bra_born_inf, bra_born_sup,  color = 'darkseagreen')
       
        
        axes[4,desired_cond_order.get(cond)].plot(t,-tlat_av, c = 'tab:red', label = 'Triceps lateral head')
        axes[4,desired_cond_order.get(cond)].plot(t_deb,-tlat_deb_av,c = 'peachpuff')
        axes[4,desired_cond_order.get(cond)].plot(t_fin,-tlat_fin_av,c = 'peachpuff')
        axes[4,desired_cond_order.get(cond)].fill_between(t,-tlat_born_inf, -tlat_born_sup,  color = 'darksalmon')

        axes[3,desired_cond_order.get(cond)].plot(t,-tlng_av, c = 'tab:red', label = 'Triceps long head')
        axes[3,desired_cond_order.get(cond)].plot(t_deb,-tlng_deb_av,c = 'peachpuff')
        axes[3,desired_cond_order.get(cond)].plot(t_fin,-tlng_fin_av,c = 'peachpuff')
        axes[3,desired_cond_order.get(cond)].fill_between(t,-tlng_born_inf, -tlng_born_sup,  color = 'darksalmon')

         # link y axes to keep same scale
        
        axes[4,desired_cond_order.get(cond)].get_shared_y_axes().join(axes[3,desired_cond_order.get(cond)], axes[4,desired_cond_order.get(cond)])
        cond_count +=1
        
    
    axes[0,0].set_ylabel(r'Pos. ($rad$)')
    axes[1,0].set_ylabel(r'Vel. ($rad.s^{-1}$)')
    axes[2,0].set_ylabel(r'Acc. ($rad.s^{-2}$)')
    axes[3,0].set_ylabel(r'EMG activity ($\%$ of max)')
    axes[4,0].set_ylabel(r'EMG activity ($\%$ of max)')
    # axes[5,0].set_ylabel(r'T.lat. ($mV$)')
    # axes[6,0].set_ylabel(r'T.lng. ($mV$)')

    axes[3,2].legend(loc = 'upper left')
    axes[4,2].legend(loc = 'upper left')
    
    
    inv_map = {v: k for k, v in desired_cond_order.items()}
    axes[0,0].set_title('{}'.format(inv_map.get(0)))
    axes[0,1].set_title('{}'.format(inv_map.get(1)))
    axes[0,2].set_title('{}'.format(inv_map.get(2)))
    
    fig.text(0.53, 0.02, r'Time ($s$)', ha='center')
    
    for axs in axes:
        for ax in axs:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlim(len(pos)/config.spec.get("sample_rate_rob"))
    #print(str([k for k,v in subj_map.items() if v == SUBJ]))
    fig.align_ylabels(axes[:, 0])
    # for axs in axes[3:5,:]:
    #     for ax in axs:
    #         ax.set_ylim(bottom = -0.005)
    #plt.tight_layout()
    return fig

def all_subj(data, param, ylabel = '', bottom = -0.16, top = 0.08, axhline = 0, per_block = True, each_subj_per_condition = True):
    palette = "deep"
    n_subj = len(data.reset_index().subject.unique())
    fig = plt.figure(dpi = 150, figsize = bt.cm2inch(15,6))

    if per_block:
        x = 'condition_by_blocks' # need this col in data set
        n_blocks = data.reset_index()['block'].unique()
    else:
        x = 'condition' # need this col in data set
        n_blocks = None
    
    if each_subj_per_condition:
        x_each_subj = 'condition'
        n_blocks_each_subj = None
        data_subj = data.groupby(['condition', 'subject']).mean().reset_index()

    ax2 = fig.add_subplot(1,2,2)
    ax1 = fig.add_subplot(1, 2, 1, sharey = ax2)
    sns.barplot(
        x=x,
        y=param,
        data = data.reset_index(),
        ax = ax1,
        color = 'grey',
        order = [
            '1G B1', '1G B2', '1G B3', '1G B4', '1G B5', '1G B6',
            '0G B1', '0G B2', '0G B3', '0G B4', '0G B5', '0G B6',
            '-1G B1', '-1G B2', '-1G B3', '-1G B4', '-1G B5', '-1G B6',
            ],
        edgecolor = ".2",
        errwidth=0.5,
        )
    ax1.set_xlabel('')
    mct.visual_tools.clean_axes(ax1)
    data = data.reset_index().rename(columns = {'subject' : 'Sujets'})
    data_subj = data_subj.reset_index().rename(columns = {'subject' : 'Sujets'})
    sns.stripplot(
        x=x_each_subj,
        y=param,
        hue= 'Sujets',
        data = data_subj,
        s = 3,
        jitter = False,
        palette  =sns.color_palette(palette, n_colors= n_subj),#"Dark2",
        ax = ax2,
        # legend = False,
        order = ['1G', '0G', '-1G'],
        )

    data_subj['condition'] = pd.Categorical(data_subj['condition'],
                                   categories=['1G', '0G', '-1G'],
                                   ordered=True)
    # remove legend from axis 'ax'
    ax2.legend_.remove()
    sns.lineplot(
        x=x_each_subj,
        y=param,
        hue= 'Sujets',
        data = data_subj,
        legend = False,
        palette  =sns.color_palette(palette, n_colors=n_subj),
        ax = ax2,
        sort = False,
        lw = 0.7,
        # order = ['1G', '0G', '-1G']
        )
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax1.set_ylabel(ylabel)
    # ax2.legend(loc='upper center', bbox_to_anchor=(1.3, 1.1),
    #           ncol=2,fontsize = 7)
    
    ax1.set_ylim(bottom=bottom, top = top)
    if not axhline == None:
        ax1.axhline(axhline, lw = 0.5, c = 'k' )
        ax2.axhline(axhline, lw = 0.5, c = 'k' )

    if n_blocks is not None:
        loc_v_lines = []
        if each_subj_per_condition:
            iterate_trought_axes = [ax1]
        else:
            iterate_trought_axes = [ax1, ax2]
        for ax in iterate_trought_axes:
            for n_cond in range(0, len(data.condition.unique())-1):
                loc_v_line = (n_cond+1)*np.max(n_blocks)-0.48
                ax.axvline(loc_v_line, lw = 0.7, c = 'k', ls = "--")
                loc_v_lines.append(loc_v_line)

            text_loc = (0 + loc_v_lines[0])/2  
            ax.text(text_loc, top-top/15, data['condition'].unique()[0])
            text_loc = (loc_v_lines[0]+loc_v_lines[1])/2  
            ax.text(text_loc, top-top/15, data['condition'].unique()[1])
            text_loc = (loc_v_lines[1]+ax.get_xlim()[1])/2  
            ax.text(text_loc, top-top/15, data['condition'].unique()[2])

            ax.get_xaxis().set_ticklabels([])

    
    for ax in [ax1, ax2]:
        ax.grid(axis = 'y', which = 'major', lw = 0.2)
    # for tick in ax1.get_xticklabels():
    #     tick.set_rotation(-45)
    # for tick in ax2.get_xticklabels():
    #     tick.set_rotation(-45)
    mct.visual_tools.clean_axes(ax2)

    plt.tight_layout()
    return fig


def err_fz(SUBJ, DIR, df):
    df_ = df.query("subject == @SUBJ & direction == @DIR")
    fig, axes = plt.subplots(5,3, figsize = bt.cm2inch(12,17), sharex=True, sharey='row', frameon = False, dpi = 100)
    pre_post_time = 0.5
    pre_post_sample = int(pre_post_time*config.spec.get("sample_rate_rob"))
    cond_count = 0
    desired_cond_order = {
        '1G':0,
        '0G':1,
        '-1G':2,
    }
    for cond, df_cond in df_.groupby('condition'):
        pos = []
        vel = []
        acc = []
        tau = []
        tau_theo = []
        fz = []
        fz_theo = []
        
        pos_deb = []
        pos_fin = []
        vel_deb = []
        vel_fin = []
        acc_deb = []
        acc_fin = []
        tau_deb = []
        tau_fin = []
        tau_theo_deb = []
        tau_theo_fin = []

        fz_deb = []
        fz_fin = []

        fz_theo_deb = []
        fz_theo_fin = []
        
        durations = []
        for mvt_id, movement_row in df_cond.groupby('movement'):
            with open(movement_row['file_name'].values[0], 'rb') as data_file:
                data_movement = pickle.load(data_file)
                df_mov = data_movement.get('df_mov')

            start = data_movement.get('bounds')[0]
            end = data_movement.get('bounds')[1]
            
            start_prev = np.clip(start -pre_post_sample,0, start) 
            end_foll = np.clip(end +pre_post_sample,0, len(df_mov)-1)
            
            
                       
            durations.append(movement_row['MD'].values[0])
            # Position
            pos.append(mct.signal.resample_by_interpolation(df_mov['pos'].iloc[start:end],len(df_mov['pos'].iloc[start:end]),1000))
            pos_deb.append(mct.signal.resample_by_interpolation(df_mov['pos'].iloc[start_prev:start],len(df_mov['pos'].iloc[start_prev:start]),pre_post_sample))
            pos_fin.append(mct.signal.resample_by_interpolation(df_mov['pos'].iloc[end:end_foll],len(df_mov['pos'].iloc[end:end_foll]),pre_post_sample))
            
            # Velocity
            vel.append(mct.signal.resample_by_interpolation(df_mov['vel'].iloc[start:end],len(df_mov['vel'].iloc[start:end]),1000))
            vel_deb.append(mct.signal.resample_by_interpolation(df_mov['vel'].iloc[start_prev:start],len(df_mov['vel'].iloc[start_prev:start]),pre_post_sample))
            vel_fin.append(mct.signal.resample_by_interpolation(df_mov['vel'].iloc[end:end_foll],len(df_mov['vel'].iloc[end:end_foll]),pre_post_sample))
            
            #Acceleration
            acc.append(mct.signal.resample_by_interpolation(df_mov['acc'].iloc[start:end],len(df_mov['acc'].iloc[start:end]),1000))
            acc_deb.append(mct.signal.resample_by_interpolation(df_mov['acc'].iloc[start_prev:start],len(df_mov['acc'].iloc[start_prev:start]),pre_post_sample))
            acc_fin.append(mct.signal.resample_by_interpolation(df_mov['acc'].iloc[end:end_foll],len(df_mov['acc'].iloc[end:end_foll]),pre_post_sample))

            #fz
            fz.append(mct.signal.resample_by_interpolation(df_mov['Fz'].iloc[start:end],len(df_mov['Fz'].iloc[start:end]),1000))
            fz_deb.append(mct.signal.resample_by_interpolation(df_mov['Fz'].iloc[start_prev:start],len(df_mov['Fz'].iloc[start_prev:start]),pre_post_sample))
            fz_fin.append(mct.signal.resample_by_interpolation(df_mov['Fz'].iloc[end:end_foll],len(df_mov['Fz'].iloc[end:end_foll]),pre_post_sample))

            #fz_theo
            fz_theo.append(mct.signal.resample_by_interpolation(df_mov['Fz_theo'].iloc[start:end],len(df_mov['Fz_theo'].iloc[start:end]),1000))
            fz_theo_deb.append(mct.signal.resample_by_interpolation(df_mov['Fz_theo'].iloc[start_prev:start],len(df_mov['Fz_theo'].iloc[start_prev:start]),pre_post_sample))
            fz_theo_fin.append(mct.signal.resample_by_interpolation(df_mov['Fz_theo'].iloc[end:end_foll],len(df_mov['Fz_theo'].iloc[end:end_foll]),pre_post_sample))

            #tau_i
            tau.append(mct.signal.resample_by_interpolation(df_mov['tau_i'].iloc[start:end],len(df_mov['tau_i'].iloc[start:end]),1000))
            tau_deb.append(mct.signal.resample_by_interpolation(df_mov['tau_i'].iloc[start_prev:start],len(df_mov['tau_i'].iloc[start_prev:start]),pre_post_sample))
            tau_fin.append(mct.signal.resample_by_interpolation(df_mov['tau_i'].iloc[end:end_foll],len(df_mov['tau_i'].iloc[end:end_foll]),pre_post_sample))

            #tau_i_theorique
            tau_theo.append(mct.signal.resample_by_interpolation(df_mov['tau_i_theo'].iloc[start:end],len(df_mov['tau_i_theo'].iloc[start:end]),1000))
            tau_theo_deb.append(mct.signal.resample_by_interpolation(df_mov['tau_i_theo'].iloc[start_prev:start],len(df_mov['tau_i_theo'].iloc[start_prev:start]),pre_post_sample))
            tau_theo_fin.append(mct.signal.resample_by_interpolation(df_mov['tau_i_theo'].iloc[end:end_foll],len(df_mov['tau_i_theo'].iloc[end:end_foll]),pre_post_sample))
            
            
        durations = np.array(durations).mean()
        t, pos_av, pos_born_inf, pos_born_sup = mean_profiles(np.array(pos).T,durations)
        t_deb, pos_deb_av, pos_deb_born_inf, pos_deb_born_sup = mean_profiles(np.array(pos_deb).T,pre_post_time)
        t_fin, pos_fin_av, pos_fin_born_inf, pos_fin_born_sup = mean_profiles(np.array(pos_fin).T,pre_post_time)
        
        _, vel_av, vel_born_inf, vel_born_sup = mean_profiles(np.array(vel).T,durations)
        _, vel_deb_av, vel_deb_born_inf, vel_deb_born_sup = mean_profiles(np.array(vel_deb).T,pre_post_time)
        _, vel_fin_av, vel_fin_born_inf, vel_fin_born_sup = mean_profiles(np.array(vel_fin).T,pre_post_time)
        
        _, acc_av, acc_born_inf, acc_born_sup = mean_profiles(np.array(acc).T,durations)
        _, acc_deb_av, acc_deb_born_inf, acc_deb_born_sup = mean_profiles(np.array(acc_deb).T,pre_post_time)
        _, acc_fin_av, acc_fin_born_inf, acc_fin_born_sup = mean_profiles(np.array(acc_fin).T,pre_post_time)

        _, fz_av, fz_born_inf, fz_born_sup = mean_profiles(np.array(fz).T,durations)
        _, fz_deb_av, fz_deb_born_inf, fz_deb_born_sup = mean_profiles(np.array(fz_deb).T,pre_post_time)
        _, fz_fin_av, fz_fin_born_inf, fz_fin_born_sup = mean_profiles(np.array(fz_fin).T,pre_post_time)
        
        _, fz_theo_av, fz_theo_born_inf, fz_theo_born_sup = mean_profiles(np.array(fz_theo).T,durations)
        _, fz_theo_deb_av, fz_theo_deb_born_inf, fz_theo_deb_born_sup = mean_profiles(np.array(fz_theo_deb).T,pre_post_time)
        _, fz_theo_fin_av, fz_theo_fin_born_inf, fz_theo_fin_born_sup = mean_profiles(np.array(fz_theo_fin).T,pre_post_time)

        _, tau_av, tau_born_inf, tau_born_sup = mean_profiles(np.array(tau).T,durations)
        _, tau_deb_av, tau_deb_born_inf, tau_deb_born_sup = mean_profiles(np.array(tau_deb).T,pre_post_time)
        _, tau_fin_av, tau_fin_born_inf, tau_fin_born_sup = mean_profiles(np.array(tau_fin).T,pre_post_time)

        _, tau_theo_av, tau_theo_born_inf, tau_theo_born_sup = mean_profiles(np.array(tau_theo).T,durations)
        _, tau_theo_deb_av, tau_theo_deb_born_inf, tau_theo_deb_born_sup = mean_profiles(np.array(tau_theo_deb).T,pre_post_time)
        _, tau_theo_fin_av, tau_theo_fin_born_inf, tau_theo_fin_born_sup = mean_profiles(np.array(tau_theo_fin).T,pre_post_time)
        
        err_fz_av = (fz_theo_av - fz_av)/fz_theo_av
        err_fz_born_sup = (fz_theo_born_sup - fz_born_sup)/fz_theo_born_sup
        err_fz_born_inf = (fz_theo_born_inf - fz_born_inf)/fz_theo_born_inf
        err_fz_deb_av = (fz_theo_deb_av - fz_deb_av)/fz_theo_deb_av
        err_fz_fin_av = (fz_theo_fin_av - fz_fin_av)/fz_theo_fin_av


        
                        
        t = t+pre_post_time
        t_fin = t_fin+t[-1]
        

        axes[0,desired_cond_order.get(cond)].plot(t,pos_av, c = 'k')
        axes[0,desired_cond_order.get(cond)].plot(t_deb,pos_deb_av,c = 'darkgrey')# colors.get(cond)
        axes[0,desired_cond_order.get(cond)].plot(t_fin,pos_fin_av,c = 'darkgrey')
        axes[0,desired_cond_order.get(cond)].fill_between(t,pos_born_inf, pos_born_sup,  color = 'lightsteelblue')
        
        axes[1,desired_cond_order.get(cond)].plot(t,vel_av, c = 'k')
        axes[1,desired_cond_order.get(cond)].plot(t_deb,vel_deb_av,c = 'darkgrey')
        axes[1,desired_cond_order.get(cond)].plot(t_fin,vel_fin_av,c = 'darkgrey')
        axes[1,desired_cond_order.get(cond)].fill_between(t,vel_born_inf, vel_born_sup,  color = 'lightsteelblue')
        
        
        axes[2,desired_cond_order.get(cond)].plot(t,acc_av, c = 'k')
        axes[2,desired_cond_order.get(cond)].plot(t_deb,acc_deb_av,c = 'darkgrey')
        axes[2,desired_cond_order.get(cond)].plot(t_fin,acc_fin_av,c = 'darkgrey')
        axes[2,desired_cond_order.get(cond)].fill_between(t,acc_born_inf, acc_born_sup,  color = 'lightsteelblue')

        axes[3,desired_cond_order.get(cond)].plot(t,fz_av, c = 'k')
        axes[3,desired_cond_order.get(cond)].plot(t_deb,fz_deb_av,c = 'darkgrey')
        axes[3,desired_cond_order.get(cond)].plot(t_fin,fz_fin_av,c = 'darkgrey')
        axes[3,desired_cond_order.get(cond)].fill_between(t,fz_born_inf, fz_born_sup,  color = 'lightsteelblue')

        axes[3,desired_cond_order.get(cond)].plot(t,fz_av, c = 'k')
        axes[3,desired_cond_order.get(cond)].plot(t_deb,fz_deb_av,c = 'darkgrey')
        axes[3,desired_cond_order.get(cond)].plot(t_fin,fz_fin_av,c = 'darkgrey')
        axes[3,desired_cond_order.get(cond)].fill_between(t,fz_born_inf, fz_born_sup,  color = 'lightsteelblue')

        axes[3,desired_cond_order.get(cond)].plot(t,fz_theo_av, c = 'tab:red')
        axes[3,desired_cond_order.get(cond)].plot(t_deb,fz_theo_deb_av,c = 'lightcoral')
        axes[3,desired_cond_order.get(cond)].plot(t_fin,fz_theo_fin_av,c = 'lightcoral')
        axes[3,desired_cond_order.get(cond)].fill_between(t,fz_theo_born_inf, fz_theo_born_sup,  color = 'mistyrose')

        axes[4,desired_cond_order.get(cond)].plot(t,err_fz_av, c = 'k')
        axes[4,desired_cond_order.get(cond)].plot(t_deb,err_fz_deb_av,c = 'darkgrey')
        axes[4,desired_cond_order.get(cond)].plot(t_fin,err_fz_fin_av,c = 'darkgrey')
        axes[4,desired_cond_order.get(cond)].fill_between(t,err_fz_born_inf, err_fz_born_sup,  color = 'lightsteelblue')

        # axes[4,desired_cond_order.get(cond)].plot(t,tau_theo_av, c = 'tab:red')
        # axes[4,desired_cond_order.get(cond)].plot(t_deb,tau_theo_deb_av,c = 'lightcoral')
        # axes[4,desired_cond_order.get(cond)].plot(t_fin,tau_theo_fin_av,c = 'lightcoral')
        # axes[4,desired_cond_order.get(cond)].fill_between(t,tau_theo_born_inf, tau_theo_born_sup,  color = 'mistyrose')
               
        
        cond_count +=1

        
    
    axes[0,0].set_ylabel(r'Pos. ($rad$)')
    axes[1,0].set_ylabel(r'Vel. ($rad.s^{-1}$)')
    axes[2,0].set_ylabel(r'Acc. ($rad.s^{-2}$)')
    axes[3,0].set_ylabel(r'Fz ($N$)')
    axes[4,0].set_ylabel(r'Fz error')
    # axes[3,0].set_ylabel(r'$\tau_{i}$ ($N.m$)')
    
    
#     axes[0,0].set_title(r'\textbf{Ctrl}')
#     axes[0,1].set_title(r'\textbf{FC}')
#     axes[0,2].set_title(r'\textbf{PCFr}')
#     axes[0,3].set_title(r'\textbf{CLPC}')
    inv_map = {v: k for k, v in desired_cond_order.items()}
    axes[0,0].set_title('{}'.format(inv_map.get(0)))
    axes[0,1].set_title('{}'.format(inv_map.get(1)))
    axes[0,2].set_title('{}'.format(inv_map.get(2)))
    
    fig.text(0.53, 0.02, r'Time ($s$)', ha='center')
    
    for axs in axes:
        for ax in axs:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlim(len(pos)/config.spec.get("sample_rate_rob"))
    #print(str([k for k,v in subj_map.items() if v == SUBJ]))
    fig.align_ylabels(axes[:, 0])
    # for axs in axes[3:5,:]:
    #     for ax in axs:
    #         ax.set_ylim(bottom = -0.005)
    #plt.tight_layout()
    return fig

def plot_adaptation(data, variable, title, ylabel = None, hline = 0.5, estimator = 'mean', all_mvt = True):
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
        lw = 0.5,
        estimator=estimator,
        ci=95
        )
    if all_mvt:

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
        # data=data.groupby("trials").mean().reset_index(),
        data=data.groupby("trials").median().reset_index(),
        s=10,
        alpha = 1,
        ax = ax1
    )
    # ax1.axvline(x = 24.5, lw = 1, color = 'k', ls = '--')
    if not hline is None:
        ax1.axhline(hline, lw = 0.2, c = 'k')
    # ax1.set_xlim(left = 0.8, right = 50.2)
    # ax1.text(25/2, ax1.get_ylim()[1], "Block 1")
    # ax1.text(25+25/2, ax1.get_ylim()[1], "Block 2")
    ax1.set_ylabel(ylabel)
    mct.visual_tools.clean_axes(ax1)

    return fig, ax1
