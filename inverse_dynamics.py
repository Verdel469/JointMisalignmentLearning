"""
This file defines functions to compute joints torques from trajectories.

Author: Dorian Verdel [d.verdel@imperial.ac.uk]
Created: 05/2026
Last modified: 07/2026
"""

## Imports
# General
import numpy as np

# Local
import motor_control_tools.signal as mct_sig

def inverse_dynamics(anthropo,j_pos, j_vel = np.array([]), j_acc = np.array([]), sRate = 100):
    """
    Compute human torques (2-dof arm) using inverse dynamics for a given
    joints trajectory.

    Args:
      - anthropo : dict           ; Contains anthropometric data of a subject
      - j_pos : 2xlen(batch) array; Shoulder & elbow joints positions
      - j_vel : 2xlen(batch) array; Shoulder & elbow joints velocities
      - j_acc : 2xlen(batch) array; Shoulder & elbow joints accelerations
      - sRate : 1x1 float         ; Sample rate [Hz]
    Outputs:
      - tau_s : 1xlen(batch) array; Estimated shoulder joint torques
      - tau_e : 1xlen(batch) array; Estimated elbow joint torques
    """
    ## Get velocity and acceleration if not provided
    if j_vel.size == 0:
        j_pos_filt = mct_sig.filter(j_pos, sRate, low_pass = 5, order = 5)
        j_vel = mct_sig.diff_keep_length(j_pos_filt, sRate)
    if j_acc.size == 0:
        j_vel_filt = mct_sig.filter(j_vel, sRate, low_pass = 5, order = 5)
        j_acc = mct_sig.diff_keep_length(j_vel_filt, sRate)

    ## Get anthropometrics
    # Upper-arm
    l_a  = anthropo.get('lengths').get('arm')
    lg_a = anthropo.get('coms').get('arm')
    m_a  = anthropo.get('masses').get('arm')
    i_a  = anthropo.get('inertias').get('arm')
    # Forearm + hand
    lg_fa = anthropo.get('coms').get('forearmHand')
    m_fa  = anthropo.get('masses').get('forearmHand')
    i_fa  = anthropo.get('inertias').get('forearmHand')

    ## Common dynamics
    g = 9.81 # [m/s²]
    mu_s = 0.05 # [Nm.s/rad] TO CHECK FROM VENTURE
    mu_e = 0.05 # [Nm.s/rad] IDEM

    ## Get joints kinematics
    qs   = j_pos[:,0]
    qe   = j_pos[:,1]
    dqs  = j_vel[:,0]
    dqe  = j_vel[:,1]
    ddqs = j_acc[:,0]
    ddqe = j_acc[:,1]

    ## Compute the different torque components
    # Inertia
    inertia_s = (i_a + i_fa + m_fa*(l_a**2 + 2*l_a*lg_fa*np.cos(qe)))*ddqs + (i_fa + m_fa*l_a*lg_fa*np.cos(qe))*ddqe
    inertia_e = (i_fa + m_fa*l_a*lg_fa*np.cos(qe))*ddqs + i_fa*ddqe
    # Coriolis
    coriolis_s = -m_fa * l_a * lg_fa * np.sin(qe) * (dqe**2 + 2*dqs*dqe)
    coriolis_e = m_fa * l_a * lg_fa * np.sin(qe) * dqs * (2*dqs + dqe)
    # Gravity
    gravity_s = g * m_a * lg_a * np.cos(qs) + g * m_fa * (l_a*np.cos(qs) + lg_fa*np.cos(qs + qe))
    gravity_e = g * m_fa * lg_fa * np.cos(qs + qe)
    # Friction
    friction_s = mu_s*dqs
    friction_e = mu_e*dqe
    # Total torques
    tau_s = inertia_s + coriolis_s + gravity_s + friction_s
    tau_e = inertia_e + coriolis_e + gravity_e + friction_e

    # Return estimated torques
    return tau_s, tau_e
