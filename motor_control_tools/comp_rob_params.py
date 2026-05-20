"""This script computes ABLE-based movement parameters."""

# Author : Dorian Verdel
# Date created : 01/2023

## Imports
import config
import numpy as np
import logging

## Compute robot parameters
def rob_params(pos, vel, acc, fx, fy, fz, tx, ty, tz,
               it_time, s = "", sample_rate = config.spec.get("sample_rate_rob")):

    """Main function computing robot parameters."""
    
    try:
        ## Compute general parameters
        dur  = np.sum(it_time)
        amp  = abs(pos[0] - pos[-1])
        mvel = amp / dur

        ## Compute velocities' local parameters
        pvel    = max(abs(vel))
        tPV     = np.sum(it_time[0:np.argmax(abs(vel))])
        rtPV    = tPV / dur
        flatvel = pvel / mvel

        ## Compute accelerations' local parameters
        maxacc     = max(acc)
        minacc     = min(acc)
        tmaxacc    = np.sum(it_time[0:np.argmax(acc)])
        tminacc    = np.sum(it_time[0:np.argmin(acc)])
        rtmaxacc   = tmaxacc / dur
        rtminacc   = tminacc / dur
        posacc_dur = np.sum(it_time[np.where(acc > 0)])
        negacc_dur = np.sum(it_time[np.where(acc < 0)])

        ## Compute maximum interaction efforts
        mFx = max(abs(fx))
        mFy = max(abs(fy))
        mFz = max(abs(fz))
        mTx = max(abs(tx))
        mTy = max(abs(ty))
        mTz = max(abs(tz))

        ## Compute mean interaction efforts
        avFx = np.mean(abs(fx))
        avFy = np.mean(abs(fy))
        avFz = np.mean(abs(fz))
        avTx = np.mean(abs(tx))
        avTy = np.mean(abs(ty))
        avTz = np.mean(abs(tz))

        ## Compute mean norm of efforts
        mNorm = max(np.sqrt(abs(fx)**2 + abs(fy)**2 + abs(fz)**2))

        ## Compute mean norm of efforts
        avNorm = np.mean(np.sqrt(abs(fx)**2 + abs(fy)**2 + abs(fz)**2))

        ## Compute work of Fz forces
        if s == 'FA':
            f_vel_prod = np.multiply(abs(fz), vel) * 0.3
            int_fvelprod = np.trapz(f_vel_prod, x = it_time)
            work = np.sum(int_fvelprod)
        else:
            f_vel_prod = np.multiply(abs(fz), vel) * 0.2 # Arm moment to check
            int_fvelprod = np.trapz(f_vel_prod, x = it_time)
            work = np.sum(int_fvelprod)

        return_dict = {
            'MD'          : dur     , 'Amp'     + s: amp       , 'mVel'    + s: mvel      , 'PVel'     + s: pvel,
            'tPVel'    + s: tPV     , 'rtPVel'  + s: rtPV      , 'flatVel' + s: flatvel   , 'maxAcc'   + s: maxacc,
            'minAcc'   + s: minacc  , 'tMaxAcc' + s: tmaxacc   , 'tMinAcc' + s: tminacc   , 'rtMaxAcc' + s: rtmaxacc,
            'rtMinAcc' + s: rtminacc, 'posDAcc' + s: posacc_dur, 'negDAcc' + s: negacc_dur, 'mFx'      + s: mFx,
            'mFy'      + s: mFy     , 'mFz'     + s: mFz       , 'mTx'     + s: mTx       , 'mTy'      + s: mTy,
            'mTz'      + s: mTz     , 'avFx'    + s: avFx      , 'avFy'    + s: avFy      , 'avFz'     + s: avFz,
            'avTx'     + s: avTx    , 'avTy'    + s: avTy      , 'avTz'    + s: avTz      , 'work'    + s: work
            }
    except ValueError:
        logging.warning("Unable to compute parameter in params()")
        return_dict = {
            'MD'          : None, 'Amp'     + s: None, 'mVel'    + s: None, 'PVel'     + s: None,
            'tPVel'    + s: None, 'rtPVel'  + s: None, 'flatVel' + s: None, 'maxAcc'   + s: None,
            'minAcc'   + s: None, 'tMaxAcc' + s: None, 'tMinAcc' + s: None, 'rtMaxAcc' + s: None,
            'rtMinAcc' + s: None, 'posDAcc' + s: None, 'negDAcc' + s: None, 'mFx'      + s: None,
            'mFy'      + s: None, 'mFz'     + s: None, 'mTx'     + s: None, 'mTy'      + s: None,
            'mTz'      + s: None, 'avFx'    + s: None, 'avFy'    + s: None, 'avFz'     + s: None,
            'avTx'     + s: None, 'avTy'    + s: None, 'avTz'    + s: None, 'work'     + s: None
            }
    return return_dict

    
