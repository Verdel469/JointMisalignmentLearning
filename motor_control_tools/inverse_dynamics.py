## Imports
# General
import numpy as np

# Local
import motor_control_tools.signal as mct_sig

def inverse_dynamics(anthropo,j_pos, j_vel = [], j_acc = [], duration = 1):
    """
    Compute human torques (2-dof arm) using inverse dynamics for a given
    joints trajectory.

    Args:
      - anthropo : dict              ; contains anthropometric data of a subject
      - j_pos    : 2xlen(batch) array; shoulder & elbow joints positions
      - j_vel    : 2xlen(batch) array; shoulder & elbow joints velocities
      - j_acc    : 2xlen(batch) array; shoulder & elbow joints accelerations
      - duration : 1x1 float         ; movement duration
    Outputs:
      - torques  : 2xlen(batch) array; estimated shoulder & elbow joints torques
    """
    ## Get velocity and acceleration if not provided
    if not j_vel:
        j_pos_filt = mct_sig.filter(j_pos, len(j_pos)/duration, low_pass = 5, order = 5)
        j_vel = mct_sig.diff_keep_length_duration(j_pos_filt, duration)
    if not j_acc:
        j_vel_filt = mct_sig.filter(j_vel, len(j_pos)/duration, low_pass = 5, order = 5)
        j_acc = mct_sig.diff_keep_length_duration(j_vel_filt, duration)

    ## Get anthropometrics
    # Upper-arm
    l_a  = anthropo.get("l_arm")
    lg_a = anthropo.get("lg_arm")
    m_a  = anthropo.get("m_arm")
    i_a  = anthropo.get("inertia_arm")
    # Forearm + hand
    lg_fa = anthropo.get("lg_forearm")
    m_fa  = anthropo.get("m_forearm")
    i_fa  = anthropo.get("inertia_forearm")

    ## Common dynamics
    g = 9.81 # [m/s²]
    mu_s = 0.05 # [Nm.s/rad] TO CHECK FROM VENTURE
    mu_e = 0.05 # [Nm.s/rad] IDEM

    ## Compute the different torque components
    # Inertia
    inertia_s = (i_a + i_fa + m_fa*(l_a**2 + 2*l_a*lg_fa*np.cos(j_pos[1,:])))*j_acc[0,:] + (i_fa + m_fa*l_a*lg_fa*np.cos(j_pos[1,:]))*j_acc[1,:]
    inertia_e = (i_fa + m_fa*l_a*lg_fa*np.cos(j_pos[1,:]))*j_acc[0,:] + i_fa*j_acc[1,:]
    # Coriolis
    coriolis_s = -m_fa * l_a * lg_fa * np.sin(j_pos[1,:]) * (j_vel[1,:]**2 + 2*j_vel[0,:]*j_vel[1,:])
    coriolis_e = m_fa * l_a * lg_fa * np.sin(j_pos[1,:]) * j_vel[0,:] * (2*j_vel[0,:] + j_vel[1,:])
    # Gravity
    gravity_s = g * m_a * lg_a * np.cos(j_pos[0,:]) + g * m_fa * (l_a*np.cos(j_pos[0,:]) + lg_fa*np.cos(j_pos[0,:] + j_pos[1,:]))
    gravity_e = g * m_fa * lg_fa * np.cos(j_pos[0,:] + j_pos[1,:])
    # Friction
    friction_s = mu_s*j_vel[0,:]
    friction_e = mu_e*j_vel[1,:]
    # Total torques
    tau_s = inertia_s + coriolis_s + gravity_s + friction_s
    tau_e = inertia_e + coriolis_e + gravity_e + friction_e

    # Return estimated torques
    torques = np.concatenate([tau_s, tau_e], axis = 0)
    return torques
