"""
Helper functions for transforms.
"""

import numpy as np

from scipy.spatial.transform import Rotation
from figs.dynamics.external_forces import ExternalForces
from figs.utilities.polynomial_helper import get_M, get_nt

def fo_to_xu(fo:np.ndarray,m:float,kt:float,
             fext:np.ndarray,
             n_mtr:int=4)  -> np.ndarray:
    """
    Converts a flat output vector to a state vector and body-rate command. Returns
    just x if quad is None.

    Args:
        - fo:       Flat output array.
        - m:        Mass of the quadcopter.
        - kt:       Total motor thrust coefficient.
        - fext:     External forces vector.
        - n_mtr:    Number of motors

    Returns:
        - xu:    State vector and control input.
    """
    
    # Unpack flat output
    pt = fo[0:3,0]
    vt = fo[0:3,1]
    at = fo[0:3,2]
    jt = fo[0:3,3]

    psit  = fo[3,0]
    psidt = fo[3,1]

    # Compute Gravity
    gt = np.array([0.00,0.00,9.81])

    # Compute Thrust
    alpha = at-gt-(fext/m)

    # Compute Intermediate Frame xy
    xct = np.array([ np.cos(psit), np.sin(psit), 0.0 ])
    yct = np.array([-np.sin(psit), np.cos(psit), 0.0 ])
    
    # Compute Orientation
    xbt = np.cross(alpha,yct)/np.linalg.norm(np.cross(alpha,yct))
    ybt = np.cross(xbt,alpha)/np.linalg.norm(np.cross(xbt,alpha))
    zbt = np.cross(xbt,ybt)
    
    Rt = np.hstack((xbt.reshape(3,1), ybt.reshape(3,1), zbt.reshape(3,1)))
    qt = Rotation.from_matrix(Rt).as_quat()

    # Compute Thrust Variables
    c = zbt.T@alpha
    uf = m*c/(n_mtr*kt)

    # Compute Angular Velocity
    B1 = c
    D1 = xbt.T@jt
    A2 = c
    D2 = -ybt.T@jt
    B3 = -yct.T@zbt
    C3 = np.linalg.norm(np.cross(yct,zbt))
    D3 = psidt*(xct.T@xbt)

    wxt = (B1*C3*D2)/(A2*(B1*C3))
    wyt = (C3*D1)/(B1*C3)
    wzt = ((B1*D3)-(B3*D1))/(B1*C3)

    wt = np.array([wxt,wyt,wzt])
    
    # Compute Body-Rate Command if Quadcopter is defined
    ut = np.hstack((uf,wt))

    # Stack
    xu = np.hstack((pt,vt,qt,ut))

    return xu

def TpCP_to_TsFO(Tp:np.ndarray,CP:np.ndarray,
                 hz:int=20,Nfo:int=4,Ndr:int=4) -> tuple[np.ndarray,np.ndarray]:
    """
    Converts a trajectory spline (defined by Tp,CP) to a sequence of trajectory
    segment times and flat outputs.

    Args:
        - Tp:  Trajectory segment times.
        - CP:  Control points.
        - hz:  Control loop frequency.
        - Nfo: Number of flat outputs.
        - Ndr: Number of derivatives.

    Returns:
        - Ts:  Trajectory time sequence.
        - FO:  Flat outputs.
    """

    # Initialize output variables
    Nt = int((Tp[-1]-Tp[0])*hz+1)
    Ts = np.linspace(Tp[0],Tp[-1],Nt)
    FO = np.zeros((Nt,Nfo,Ndr))

    # Compute flat outputs
    idx = 0
    for k in range(Nt):
        tk = Tp[0]+k/hz

        if tk > Tp[idx+1] and idx < len(Tp)-2:
            idx += 1

        t0,tf = Tp[idx],Tp[idx+1]
        CPk = CP[idx,:,:]
        
        FO[k,:,:] = CP_to_fo(tk-t0,tf-t0,CPk,Nfo,Ndr)

    return Ts,FO

def TsFO_to_tXU(Ts:np.ndarray,FO:np.ndarray,
                m:float,kt:float,
                Fext:ExternalForces|None,
                n_mtr:int=4,ndim:int=15) -> np.ndarray:
    """
    Converts a sequence of trajectory sequence times and flat outputs to a state
    vector and control input rollout.

    Args:
        - Ts:       Trajectory time sequence.
        - FO:       Flat outputs.
        - m:        Mass of the quadcopter.
        - kt:       Total motor thrust coefficient.
        - Fext:     External forces object.
        - n_mtr:    Number of motors
        - ndim:     Number of dimensions in the state vector.

    Returns:
        - tXU:      State vector and control input rollout.
    """
    
    # Initialize output variables
    N = FO.shape[0]
    tXU = np.zeros((ndim,N))

    # Compute flat outputs
    for k in range(N):
        # Compute External Forces (if any)
        if Fext is None:
            fext = np.zeros(3)
        else:
            pv = FO[0:3,0,:].flatten()
            fext = Fext.get_forces(pv)

        # Compute state vector and control input
        xu = fo_to_xu(FO[k,:,:],m,kt,fext,n_mtr)

        # Store in output variable
        tXU[0,k] = Ts[k]
        tXU[1:,k] = xu

    return tXU

def kf_to_TsFO(kf:dict[str,dict],Nfo:int=4,Ndr:int=4) -> tuple[np.ndarray,np.ndarray]:
    """
    Converts a waypoint trajectory (defined by WPs) to a sequence of trajectory
    segment times and flat outputs.
    Args:
        - kf:  Waypoint trajectory.
        - Nfo: Number of flat outputs.
        - Ndr: Number of derivatives.

    Returns:
        - Ts:  Trajectory time sequence.
        - FO:  Flat outputs.
    """


    Ts = np.array([kf["t"]])    
    FO = np.zeros((1,Nfo,Ndr))
    for i in range(Nfo):
        fo = kf["fo"][i]
        for j in range(len(fo)):
            FO[0,i,j] = fo[j]

    return Ts,FO

def CP_to_fo(tk:float,tf:float,CP:np.ndarray,
             Nfo:int,Ndr:int) -> np.ndarray:
    """
    Converts a trajectory spline (defined by Tp,CP) to a flat output.

    Args:
        - tk:   Segment current time.
        - tf:   Segment final time.
        - CP:   Control points.
        - Nfo:  Number of flat outputs.
        - Ndr:  Number of derivatives.

    Returns:
        - fo:   Flat output vector.
    """

    Ncp = CP.shape[1]
    M = get_M(Ncp)

    fo = np.zeros((Nfo,Ndr))
    for i in range(Ndr):
        nt = get_nt(tk/tf,i,Ncp)
        fo[:,i] = (CP@M@nt) / (tf**i)

    return fo

def xv_to_T(xcr:np.ndarray) -> np.ndarray:
    """
    Converts a state vector to a transfrom matrix.

    Args:
        - xcr:    State vector.

    Returns:
        - Tcr:    Pose matrix.
    """
    Tcr = np.eye(4)
    Tcr[0:3,0:3] = Rotation.from_quat(xcr[6:10]).as_matrix()
    Tcr[0:3,3] = xcr[0:3]

    return Tcr