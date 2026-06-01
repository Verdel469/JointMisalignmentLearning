## General imports
import numpy as np
import pickle
## Local imports
import motor_control_tools as mct


def cut_vel_submovements(vel, cut = 0.05, rob = False, pv_h = 0):
    """
    This function cuts the velocity profile at 5% of its peak but
    accounts for possible submovements before and after the main
    bell-shaped component.
    """
    bounds = []
    ## Pre-processing
    vel_b_0 = vel[30:]
    vel_abs = np.abs(vel_b_0)
    if rob:
        centered_vel = vel_abs - cut*pv_h
    else:
        idx_max_vel = np.argmax(vel_abs)
        centered_vel = vel_abs - cut*vel_abs[idx_max_vel]

    ## Get bounds
    for i in range(0,len(centered_vel)):
        if centered_vel[i]>0:
            bounds.append(i + 30)
            break
    for i in range(0,len(centered_vel)):
        if centered_vel[-i]>0:
            bounds.append(len(centered_vel) - i + 30)
            break

    ## Return
    if not bounds:
        return [] # Handle cases where bounds cannot be found
    else:
        return bounds

def compare_bounds(bounds_a, bounds_b, both = False):
    """
    This function compares bounds obtained using the peak velocity of
    several joints to extract the overall begining and end of the movement.
    """
    if both:
        ## Extend shortest to max length
        len_a = bounds_a[1]-bounds_a[0]
        len_b = bounds_b[1]-bounds_b[0]
        if len_a < len_b:
            bounds_a[1] = bounds_a[0] + len_b
        else:
            bounds_b[1] = bounds_b[0] + len_a
        ## Return
        return bounds_a, bounds_b
    else:
        bounds = []
        ## Overall start
        if bounds_a[0] < bounds_b[0]:
            bounds.append(bounds_a[0])
        else:
            bounds.append(bounds_b[0])
        ## Overall end
        if bounds_a[1] < bounds_b[1]:
            bounds.append(bounds_b[1])
        else:
            bounds.append(bounds_a[1])
        ## Return
        return bounds
    
def extend_bounded_profiles(q_h, q_h_dot, q_r, q_r_dot, forces=False):
    """
    This function extends the shortest profiles so that human and
    robot data match in length.
    """
    if forces:
        forces_b = q_r
        for i in range(0, len(q_h)-len(forces_b)):
            forces_b = np.append(forces_b, forces_b[-1,:].reshape(1,-1), axis = 0)
        return forces_b
    else:
        if len(q_h) == len(q_r):
            return q_h,q_h_dot,q_r,q_r_dot
        elif len(q_h) < len(q_r):
            for i in range(0,len(q_r)-len(q_h)):
                q_h = np.append(q_h, q_h[-1:])
                q_h_dot = np.append(q_h_dot, q_h_dot[-1:])
            return q_h,q_h_dot,q_r,q_r_dot
        else:
            for i in range(0,len(q_h)-len(q_r)):
                q_r = np.append(q_r, q_r[-1])
                q_r_dot = np.append(q_r_dot, q_r_dot[-1])
            return q_h,q_h_dot,q_r,q_r_dot
    
def get_aligned_hr_joints_mvt(q_sh, q_sh_dot, q_el, q_el_dot, q_3, q_3_dot, q_4, q_4_dot, forces, cut = 0.05):
    """
    This function allows to align one human and robot joints movements on the begining of the movement.
    """
    ## Get human joints movement bounds
    bounds_sh = cut_vel_submovements(q_sh_dot, cut = cut)
    bounds_el = cut_vel_submovements(q_el_dot, cut = cut)
    if not (bounds_sh and bounds_el):
        return [], [], [], [], [], [], [], [], []
    bounds_human = compare_bounds(bounds_sh, bounds_el)
    # Get human joints peak velocities to align robot segmentation
    pv_sh = np.max(np.abs(q_sh_dot))
    pv_el = np.max(np.abs(q_el_dot))

    ## Get robot joints movement bounds
    bounds_q_3 = cut_vel_submovements(q_3_dot, cut = cut, rob = True, pv_h = pv_sh)
    bounds_q_4 = cut_vel_submovements(q_4_dot, cut = cut, rob = True, pv_h = pv_el)
    if not (bounds_q_3 and bounds_q_4):
        return [], [], [], [], [], [], [], [], []
    bounds_robot = compare_bounds(bounds_q_3, bounds_q_4)
    
    ## Get common bounds
    bounds_human, bounds_robot = compare_bounds(bounds_human, bounds_robot, both = True)
    #print("Bounds human: ", bounds_human)
    #print("Bounds robot: ", bounds_robot)
    
    ## Get bounded joints and force profiles
    # Human
    q_sh_b = q_sh[bounds_human[0]:bounds_human[1]]
    q_el_b = q_el[bounds_human[0]:bounds_human[1]]
    q_sh_dot_b = q_sh_dot[bounds_human[0]:bounds_human[1]]
    q_el_dot_b = q_el_dot[bounds_human[0]:bounds_human[1]]
    # Robot
    q_3_b = q_3[bounds_robot[0]:bounds_robot[1]]
    q_4_b = q_4[bounds_robot[0]:bounds_robot[1]]
    q_3_dot_b = q_3_dot[bounds_robot[0]:bounds_robot[1]]
    q_4_dot_b = q_4_dot[bounds_robot[0]:bounds_robot[1]]
    forces_b = forces[bounds_robot[0]:bounds_robot[1], :]
    #print("Bounded forces shape: ", np.shape(forces_b))
    
    ## Extend profiles at each joint
    q_sh_be, q_sh_dot_be, q_3_be, q_3_dot_be = extend_bounded_profiles(q_sh_b, q_sh_dot_b, q_3_b, q_3_dot_b)
    q_el_be, q_el_dot_be, q_4_be, q_4_dot_be = extend_bounded_profiles(q_el_b, q_el_dot_b, q_4_b, q_4_dot_b)
    ## Extend forces profiles
    if len(q_sh_be) > len(forces_b):
        forces_b = extend_bounded_profiles(q_sh_be, [], forces_b, [], forces = True)

    ## Return
    return q_sh_be, q_sh_dot_be, q_3_be, q_3_dot_be, q_el_be, q_el_dot_be, q_4_be, q_4_dot_be, forces_b

# Prepare empty matrices (IF NEEDED)
# q_sh = []
# q_el = []
# q_sh_dot = []
# q_el_dot = []
# q_3 = []
# q_4 = []
# q_3_dot = []
# q_4_dot = []
# q_sh_a = []
# q_el_a = []
# q_sh_dot_a = []
# q_el_dot_a = []
# q_3_a = []
# q_4_a = []
# q_3_dot_a = []
# q_4_dot_a = []

def get_all_aligned_hr_joints_mvts_one_subj(df_one_subj_h, df_one_subj_r, subj = "S1", cut = 0.05, forceLoc = "wrist"):
    """
    This function allows to loop on all movements of one subject to
    build the matrix then used for the training and evaluation of the
    robot-to-human mapping learning.
    """
    # Get anthropometric data
    arm_length = 0.3 #df_one_subj_h['arm_length'].mean().iloc[0]
    farm_length = 0.21 #df_one_subj_h['farm_length'].mean().iloc[0]
    str_subject = subj
    subject = int(str_subject[1:])
    print(subject)
    
    # Prepare the matrix for all movements storage
    all_mvts_one_subj = np.empty((1,24))
    for i in range(0, len(df_one_subj_h)):
        # Get file names
        humanfile = df_one_subj_h['file_name'].iloc[i]
        robfile = df_one_subj_r['file_name'].iloc[i]

        # Extract movement dicts
        with open(humanfile, 'rb') as hkin:
            dict_hkin = pickle.load(hkin)
        with open(robfile, 'rb') as rkin:
            dict_rkin = pickle.load(rkin)

        # Get human joints angles & vels
        hdf_mov = dict_hkin.get('df_mov')
        q_sh = hdf_mov['q_sh'].values
        q_el = hdf_mov['q_el'].values
        q_sh_dot = mct.signal.diff_keep_length(q_sh, 100, spec_t = False)
        q_el_dot = mct.signal.diff_keep_length(q_el, 100, spec_t = False)
        # Get robot joints angles & vels
        rdf_mov = dict_rkin.get('df_mov')
        q_3 = rdf_mov['pos_A'].values
        q_4 = rdf_mov['pos_FA'].values + np.pi/2
        q_3_dot = -mct.signal.diff_keep_length(q_3, 100, spec_t = False)
        q_4_dot = -mct.signal.diff_keep_length(q_4, 100, spec_t = False)
        # Get human-robot interaction forces
        if forceLoc == "both":
            fx_a = rdf_mov['Fx_A'].values
            fy_a = rdf_mov['Fy_A'].values
            fz_a = rdf_mov['Fz_A'].values
            tx_a = rdf_mov['Tx_A'].values
            ty_a = rdf_mov['Ty_A'].values
            tz_a = rdf_mov['Tz_A'].values
        fx_fa = rdf_mov['Fx_FA'].values
        fy_fa = rdf_mov['Fy_FA'].values
        fz_fa = rdf_mov['Fz_FA'].values
        tx_fa = rdf_mov['Tx_FA'].values
        ty_fa = rdf_mov['Ty_FA'].values
        tz_fa = rdf_mov['Tz_FA'].values
        # Concatenate forces in one matrix
        if forceLoc == "both":
            forces = np.concat((fx_a.reshape(len(fx_a),1), fy_a.reshape(len(fy_a),1), fz_a.reshape(len(fz_a),1),
                                tx_a.reshape(len(tx_a),1), ty_a.reshape(len(ty_a),1), tz_a.reshape(len(tz_a),1),
                                fx_fa.reshape(len(fx_fa),1), fy_fa.reshape(len(fy_fa),1), fz_fa.reshape(len(fz_fa),1),
                                tx_fa.reshape(len(tx_fa),1), ty_fa.reshape(len(ty_fa),1), tz_fa.reshape(len(tz_fa),1)), axis = 1)
        else:
            forces = np.concat((fx_fa.reshape(len(fx_fa),1), fy_fa.reshape(len(fy_fa),1), fz_fa.reshape(len(fz_fa),1),
                                tx_fa.reshape(len(tx_fa),1), ty_fa.reshape(len(ty_fa),1), tz_fa.reshape(len(tz_fa),1)), axis = 1)

        # Get aligned human-robot movement joints data
        q_sh_a, q_sh_dot_a, q_el_a, q_el_dot_a, q_3_a, q_3_dot_a, q_4_a, q_4_dot_a, forces_a = get_aligned_hr_joints_mvt(q_sh, q_sh_dot, q_el, q_el_dot, q_3, q_3_dot, q_4, q_4_dot, forces, cut = cut)
        
        if np.any(q_sh_a):
            # Generate matrix for the movement
            lmvt = len(q_sh_a)
            arm_vec = np.array([arm_length]*lmvt)
            farm_vec = np.array([farm_length]*lmvt)
            subj_vec = np.array([subj]*lmvt)
            mvt_vec = np.array([i]*lmvt)
            aligned_mvt = np.concat((q_3_a.reshape(lmvt,1), q_4_a.reshape(lmvt,1),
                                    q_3_dot_a.reshape(lmvt,1), q_4_dot_a.reshape(lmvt,1), forces_a,
                                    arm_vec.reshape(lmvt,1), farm_vec.reshape(lmvt,1),
                                    mvt_vec.reshape(lmvt,1), subj_vec.reshape(lmvt,1),
                                    q_sh_a.reshape(lmvt,1), q_sh_dot_a.reshape(lmvt,1),
                                    q_el_a.reshape(lmvt,1), q_el_dot_a.reshape(lmvt,1)), axis = 1)
            
            # Store movement in the subject's matrix
            all_mvts_one_subj = np.concat((all_mvts_one_subj, aligned_mvt), axis = 0)

    ## Return the matrix with all movements for 1 subject
    return all_mvts_one_subj
        

# def get_all_aligned_hr_joints_mvts_pop(df_pop, cut = 0.05):
#     """
#     This function allows to loop on all the movements of a dataset to
#     build the subject-by-subject matrices then used for the training and
#     evaluation of the robot-to-human mapping learning.
#     """