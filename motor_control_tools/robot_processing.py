import numpy as np
import os


def compute_tau_i(l, fz, Delta_q):
    """où $l$ est la distance entre le coude humain et l'attache,
    $fz$ est la force renvoyée sur l'axe z du capteur (repère capteur)
    et $\Delta{q}$ est le décalage angulaire moyen que nous identifions
    sur la plage de mouvement.

    Args:
        l ([type]): [description]
        fz ([type]): [description]
        Delta_q ([type]): [description]
    """
    # print(l*np.cos(Delta_q))
    return l*fz*np.cos(Delta_q)

def compute_theoritical_fz(m,q):
    """Compute theoritical Fz, the interaction force record by the effort sensor on the robot

    Args:
        m (float): Mass to compensate by the robot (generally the mass of the human forearm with a factor 0, 1 or 2)
        l (float): Length of the human forearm (disctance between elbow and the fixation of the splint)
        q (float): Angle of the human elbow

    Returns:
        float: Theroritical Fz
    """
    g = 9.81
    return -m*g*np.cos(q)

def compute_g_felt(Fz, m, q):
    g_felt = Fz/(m*np.cos(q))
    return 1-(g_felt/9.81)


