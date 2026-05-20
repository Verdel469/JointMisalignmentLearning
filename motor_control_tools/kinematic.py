# File name : kinematic_processing.py
# Author : Simon Bastide
# Encoding : utf-8
# Function : Tools for kinematic data processing
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import logging
import motor_control_tools as mct
from skspatial.objects import Points, Line
from motor_control_tools.signal import reversal_points
from scipy import integrate

def movement_bounds(position, sample_rate, velocity = None, treshold = 5, relativ_treshold = True):
    """Determine le début et la fin d'un mouvement. Le mouvement est considéré
     lorsque la vitesse du mouvement est supérieur au seuil (relatif ou absolu).
     Si le seuil est relatif (relativ_treshold = True), treshold doit être donné en % (5 par default)
     Si le seuil est absolue (relativ_treshold = False), treshold doit être donné dans la même unité que la vitesse (5 par default). 
     
     
     Notes : Il est préférable de donner en entré une position filtrée pour éviter que la vitesse
      calculée soit trop bruitée"""
    if velocity is None:
        velocity = np.diff(position)*sample_rate
        
    vel_abs = abs(velocity)
    if relativ_treshold:
        treshold_value = (treshold/100)*np.nanmax(vel_abs)
    else:
        treshold_value = treshold
    max_vel_loc = np.nanargmax(vel_abs)
    
    start_frame, end_frame = cut_velocity(vel_abs, max_vel_loc, treshold_value)  

    return start_frame, end_frame

def sub_movement_bounds(velocity, bounds, count_sub_movement = False, sub_movement_search_range = 200):
    """[summary]

    Args:
        velocity ([type]): [description]
        bounds (tuple): Bounds of the main movement

    Returns:
        [type]: [description]
    """
    if len(velocity)>bounds[1]+sub_movement_search_range:
        velocity_post_movement = velocity[bounds[1]:bounds[1]+sub_movement_search_range]
    else:
        velocity_post_movement = velocity[bounds[1]:]
    
    cross_0 = reversal_points(velocity_post_movement, distance_limit=20) #Find indexes where velocity cross the 0 line (reversal points)


    n_sub_movement = np.clip(len(cross_0)-1, a_min=0, a_max=None)
    # print("nb sub movement : " +  str(n_sub_movement))
        
    if len(cross_0) == 1: # Cas ou un seul reversal point trouvé # 
        start_sub = cross_0[0] + bounds[1]
        end_sub = len(velocity)
    elif len(cross_0) == 0: # si pas de reversal point = pas de sub mvt
        start_sub = bounds[1]
        end_sub = bounds[1]
    else: # Cas normal ou au moins 2 reversal points sont trouvés 
        # Les paramètres dont alors calculé sur le premier sub_movement
        start_sub = cross_0[0] + bounds[1]
        end_sub = cross_0[1] + bounds[1] #Start and end of the sub movement

    if count_sub_movement:
        return (start_sub, end_sub), n_sub_movement

    return start_sub, end_sub

def check_movement_bounds(position, velocity, sample_rate, treshold = 5, relativ_treshold = True):

    start, end = movement_bounds(
        position=position,
        sample_rate=sample_rate,
        velocity=velocity,
        treshold=treshold,
        relativ_treshold=relativ_treshold
    )
    start_sub, end_sub = sub_movement_bounds(
        velocity = velocity,
        bounds = (start, end)
    )

    vel_abs = abs(velocity)
    if relativ_treshold:
        treshold_value = (treshold/100)*np.nanmax(vel_abs)
    else:
        treshold_value = treshold
    max_vel_loc = np.nanargmax(vel_abs)

    plt.figure()
    plt.title("Position")
    plt.plot(position, label = 'Position', color = 'grey')
    plt.axvline(x=start, label = 'Start', color = 'brown')
    plt.axvline(x=end, label = 'End', color = 'brown')
    plt.axvline(x=start_sub, label = 'Start Sub', color = 'cyan')
    plt.axvline(x=end_sub, label = 'End Sub', color = 'cyan')
    plt.legend()

    plt.figure()
    plt.title("Vitesse")
    plt.plot(abs(velocity), label = 'vitesse', color = 'grey')
    #plt.plot(vel_abs, label = 'vitesse absolue')
    plt.axvline(x=max_vel_loc, label = 'Vitesse max', color = 'green')
    plt.axvline(x=start, label = 'Start', color = 'brown')
    plt.axvline(x=end, label = 'End', color = 'brown')
    plt.axvline(x=start_sub, label = 'Start Sub', color = 'cyan')
    plt.axvline(x=end_sub, label = 'End Sub', color = 'cyan')
    plt.axhline(y=treshold_value, label = 'seuil', color = 'red')
    
    plt.legend()
    plt.show()

def cut_velocity(vel_abs, max_vel_loc, treshold):

    starting_frame_find = False
    ending_frame_find = False
    

    for frame, vel_value in reversed(list(enumerate(vel_abs[:max_vel_loc]))):
        if (vel_value < treshold) and not starting_frame_find:
            start_frame = frame
            starting_frame_find = True
            break
                
    for frame, vel_value in (enumerate(vel_abs[max_vel_loc:])):
        if (vel_value < treshold) and not ending_frame_find:
            end_frame = max_vel_loc+frame
            ending_frame_find = True
            break
    
    if not starting_frame_find:
        start_frame = 1
        logging.warning("Début du mouvement non indentifié : fixé au début de l'enregistrement par défault")
    if not ending_frame_find:
        end_frame = len(vel_abs)
        logging.warning("Fin du mouvement non indentifié : fixé à la fin de l'enregistrement par défault")

    return start_frame, end_frame

def overshoot_on_position(pos_signal, end_of_movement, direction):
    """Take the position signal as input and return the amplitude of the overshoot(if present)

    Args:
        pos_signal (array): position signal

    Output:
        amplitude (float): Amplitude of the overshoot
    """
    searching_range = 50
    # overshoot calculer comme le max (ou min) de position atteint dans un range de -100+100 autour de la fin du mouvement
    born_for_searching_overshoot = [
        end_of_movement-searching_range,
        end_of_movement+searching_range,
    ]
    try:
        if born_for_searching_overshoot[1] > len(pos_signal)-end_of_movement:# si le range de recherche de l'overshoot est trop grand par rapport au signal restant aprés la
            # fin du mouvement. Permet d'eviter d'être out of index
            born_for_searching_overshoot[1] = len(pos_signal) -10 # on prend juste les 10 derniéres valeurs
        if direction == 'up':
            overshoot_amplitude = np.max(pos_signal[born_for_searching_overshoot[0]:born_for_searching_overshoot[1]]) - np.mean(pos_signal[born_for_searching_overshoot[1]:-1])# mean of ten last values considered as the plateau

        elif direction == 'down':
            overshoot_amplitude = np.mean(pos_signal[born_for_searching_overshoot[1]:-1]) - np.min(pos_signal[born_for_searching_overshoot[0]:born_for_searching_overshoot[1]])# mean of ten last values considered as the plateau

        # if the value is positive --> overshoot.
        # If negative --> undershoot
    except ValueError:
        overshoot_amplitude = None
        
    return {'overshoot' : overshoot_amplitude}

def isupward(max_velocity, n_mov = None):
    if n_mov is None:
        n_mov = 1
    if max_velocity > 0 and n_mov % 2 == 1:
        return True
    elif max_velocity > 0 and n_mov % 2 == 0:
        #logging.warning("Mouvement n° {} à vérifier".format(n_mov))
        return True
    elif max_velocity < 0 and n_mov % 2 == 1:
        #logging.warning("Mouvement n° {} à vérifier".format(n_mov))
        return False
    else:
        return False

def get_id_dict(subj_id = None, cond_id = None, block_id = None, mov_id = None):

     return {'subject':subj_id, 'condition': cond_id, 'block': block_id,\
         'movement': mov_id}

def get_position_params(position, sample_rate, extension = ""):

    duration = len(position)/sample_rate
    amplitude = position[-1]-position[0]

    return {'duration' + extension : duration, 'amplitude' + extension : amplitude}

def get_velocity_params(velocity, sample_rate, extension = ""):

    duration = len(velocity)/sample_rate
    mean_vel = np.mean(abs(velocity))
    max_vel_loc = np.argmax(abs(velocity))
    max_vel = velocity[max_vel_loc]
    time_to_peak_vel = max_vel_loc * 1/sample_rate
    

    return {'duration' + extension : duration, 'max_vel' + extension : max_vel, 'time_to_peak_vel' + extension : time_to_peak_vel,\
         'mean_vel' + extension : mean_vel}

def get_max_vel_loc(velocity):
    return np.argmax(abs(velocity))
    
def get_acceleration_params(acceleration, max_vel_loc, sample_rate, extension = ""):
    
    max_accel_loc = np.argmax(abs(acceleration[:max_vel_loc]))
    max_decel_loc = max_vel_loc + np.argmax(abs(acceleration[max_vel_loc:]))
    max_accel = acceleration[max_accel_loc]
    max_decel = acceleration[max_decel_loc]
    mean_accel = np.mean(acceleration[:max_vel_loc])
    mean_decel = np.mean(acceleration[max_vel_loc:])
    time_to_peak_acc = max_accel_loc * 1/sample_rate

    return {'max_accel' + extension : max_accel, 'time_to_peak_acc' + extension : time_to_peak_acc,\
         'mean_accel' + extension : mean_accel, 'max_decel' + extension : max_decel,\
             'mean_decel' + extension : mean_decel}

def comp_smoothness(posx, posy, posz, s_rate):
    """Compute smoothness as defined by Jarrasse & al. (2010)"""
    ## Compute components of velocity
    velx = mct.signal.diff_keep_length(posx, s_rate)
    vely = mct.signal.diff_keep_length(posy, s_rate)
    velz = mct.signal.diff_keep_length(posz, s_rate)
    ## Compute components of acceleration
    accx = mct.signal.diff_keep_length(velx, s_rate)
    accy = mct.signal.diff_keep_length(vely, s_rate)
    accz = mct.signal.diff_keep_length(velz, s_rate)
    ## Compute components of jerk
    jerkx = mct.signal.diff_keep_length(accx, s_rate)
    jerky = mct.signal.diff_keep_length(accy, s_rate)
    jerkz = mct.signal.diff_keep_length(accz, s_rate)
    ## COmpute squaresd jerk components
    jerkx2 = np.dot(jerkx,jerkx)
    jerky2 = np.dot(jerky,jerky)
    jerkz2 = np.dot(jerkz,jerkz)
    
    smoothness = (jerkx2 + jerky2 + jerkz2) / (len(posx) * s_rate)
    return smoothness


def params(position, posx, posy, posz, velocity = None, acceleration = None, sample_rate = 100, plot = True, suffix = ""):
    """Calcul des paramètres simples sur le mouvement. La position en entrée doit étre la
    position du mouvement effectif (utiliser movement_bounds avant pour selectionner le mouvement effectif).
    Il est préférable de donner la position filtrée en entrée. 
    Cette fonction retourne un dictionnaire conteannt les paramétres calculés."""

    if velocity is None:
        velocity = np.diff(position)*sample_rate
    if acceleration is None:
        acceleration = np.diff(velocity)*sample_rate


    try:
        duration = len(position)/sample_rate
        amplitude = position[-1]-position[0]
        mean_vel = np.mean(abs(velocity))
        peak_vel_loc = np.argmax(abs(velocity))
        peak_accel_loc = np.argmax(abs(acceleration[:peak_vel_loc]))
        peak_decel_loc = peak_vel_loc + np.argmax(abs(acceleration[peak_vel_loc:]))
        peak_vel = velocity[peak_vel_loc]
        peak_accel = acceleration[peak_accel_loc]
        peak_decel = acceleration[peak_decel_loc]
        mean_accel = np.mean(acceleration[:peak_vel_loc])
        mean_decel = np.mean(acceleration[peak_vel_loc:])

        time_to_peak_vel = peak_vel_loc * 1/sample_rate
        time_to_peak_acc = peak_accel_loc * 1/sample_rate
        time_to_peak_decel = peak_decel_loc * 1/sample_rate

        relativ_time_to_peak_vel = time_to_peak_vel/duration
        relativ_time_to_peak_acc = time_to_peak_acc/duration
        peak_vel_relativ_time_to_peak_acc = time_to_peak_acc/time_to_peak_vel
        vel_flat = peak_vel/mean_vel

        smoothness = comp_smoothness(posx, posy, posz, sample_rate)

        curve = compute_curve(posx,posz)

        if plot:
            time = np.linspace(0, duration, len(position))
            plt.figure
            plt.subplot(3, 1, 1)
            plt.title("Position")
            plt.plot(time, position)
            plt.subplot(3, 1, 2)
            plt.title("Vitesse")
            plt.plot(time, velocity)
            plt.scatter(time[peak_vel_loc], velocity[peak_vel_loc], label = 'Vitesse max', color = 'green', marker = '*')
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.title("Acceleration")
            plt.plot(time, acceleration)
            plt.scatter(time[peak_accel_loc], acceleration[peak_accel_loc], label = 'Acceleration max', color = 'green', marker = '*')
            plt.scatter(time[peak_decel_loc], acceleration[peak_decel_loc], label = 'Deceleration max', color = 'red', marker = '*')
            plt.xlabel("Time (s)")
            plt.legend()

            plt.tight_layout()
            plt.show()
    
    except ValueError:
        logging.warning("Unable to compute parameter in params()")
        duration = None
        amplitude = None
        mean_vel = None
        peak_vel = None
        peak_accel = None
        peak_decel = None
        mean_accel = None
        mean_decel = None
        time_to_peak_vel = None
        time_to_peak_acc = None
        time_to_peak_decel = None
        relativ_time_to_peak_vel = None
        relativ_time_to_peak_acc = None
        peak_vel_relativ_time_to_peak_acc = None
        vel_flat = None
        smoothness = None
        curve = None

    return ({
        'MD' + suffix : duration,
        'A' + suffix : amplitude,
        'PV' + suffix : peak_vel,
        'tPV' + suffix : time_to_peak_vel,
        'rtPV' + suffix : relativ_time_to_peak_vel,
        'mV' + suffix : mean_vel,
        'PA' + suffix : peak_accel, 
        'tPA' + suffix : time_to_peak_acc,
        'rtPA' + suffix : relativ_time_to_peak_acc,
        'rtPA/rtPV' + suffix : peak_vel_relativ_time_to_peak_acc,
        'mA'+ suffix :  mean_accel, 
        'PD'+ suffix : peak_decel,
        'tPD' + suffix : time_to_peak_decel, 
        'mD' + suffix : mean_decel,
        'PV/mV' + suffix : vel_flat,
        'smooth' + suffix: smoothness,
        'curve' + suffix: curve
        })

def compute_curve(posx,posz):
    """ Compute curvature in the sagittal plane. """
    
    l0 = posx[-1] - posx[0]
    l1 = posz[-1] - posz[0]
    line_vec = np.array([[l0], [l1]])
    succ_pts = np.array([posx - posx[0],posz - posz[0]])
    vect_prod = np.cross(succ_pts.T,line_vec.T)
    curve = max(abs(vect_prod))/np.linalg.norm(line_vec)
    return curve

def sub_movement_params(position, velocity, sample_rate):
    try:
        if len(position) == 0: #pas d'overshoot/sub_movement
            duration = 0
            amplitude = 0
            auc = 0
        else:
            duration =  len(position)/sample_rate
            # max_vel_loc = np.argmax(velocity)
            amplitude = abs(abs(position[-1])-abs(position[0]))
            auc = integrate.simps(abs(velocity), dx = 1/sample_rate) # area under curve

            if duration > 0.5:
                raise ValueError

    except ValueError:
        logging.warning("Unable to compute parameter in sub_movement_params() or submovement duration greater than 0.5")
        duration = None
        amplitude = None
        auc = None

    return ({
        'subMD' : duration,
        'subA' : amplitude,
        'subauc' : auc
    })


def unit_vector(vector):
    """ Returns the unit vector of a vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1_array, v2_array):
    """ Returns an array of angle in radians between vectors 'v1_array' and 'v2_array' """
    angle = list()
    if len(v1_array) == len(v2_array):
        for v1,v2 in zip(v1_array,v2_array):
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            angle.append(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    else:
        print("Arrays must have the same length ")
    return np.array(angle)

def tangential_velocity(pos, sample_rate):
    """Retroune la vitesse tangentielle du signal """

    if pos.shape[0] < pos.shape[1]:
        pos = pos.transpose()
    vel = np.diff(pos,axis = 0)
    vel = np.sqrt(np.sum(np.square(vel), axis = 1))*sample_rate
    return np.append(vel, vel[-1])

def travelled_distance(vel, sample_rate):
    """Retourne la distance parcourue en fonction du temps"""
    return np.concatenate(([0], np.cumsum(vel)))/sample_rate


def fit_line_direction(points):
    direction = list()
    for frame in range(0,len(points)):
        points_ = Points(np.reshape(np.array(points[frame,:]), (-1,3)))
        line_fit = Line.best_fit(points_)
        direction.append(np.array(line_fit.direction))
    return np.array(direction) 
