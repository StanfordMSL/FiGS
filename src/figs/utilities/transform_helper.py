"""
Helper functions for transforms.
"""

import numpy as np
import figs.utilities.polynomial_helper as ph

from scipy.spatial.transform import Rotation
from figs.dynamics.external_forces import ExternalForces

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

def x_to_fo(x:np.ndarray,Nfo:int=4,Ndr:int=5) -> np.ndarray:
    """
    Converts a state vector to flat output vector.

    Args:
        x:      State vector
        Ndr:    Number of derivatives.
        Nfo:    Number of flat outputs.

    Returns:
        fo:     Flat outputs.
    """


    # Initialize output
    fo = np.full((Nfo,Ndr),np.nan)

    # Compute yaw term
    Rk = Rotation.from_quat(x[6:10]).as_matrix()
    psi = np.arctan2(Rk[1,0], Rk[0,0])

    # Compute linear terms
    fo[0:3,0] = x[0:3]
    fo[0:3,1] = x[3:6]
    fo[3,0]  = psi

    return fo

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

    # Get some useful constants
    _,Nfo,Nco = CP.shape
    dT = np.diff(Tp)

    # Initialize output variable
    Nt = int((Tp[-1]-Tp[0])*hz+1)
    Ts = np.linspace(Tp[0],Tp[-1],Nt)
    FO = np.zeros((Nt,Nfo,Ndr))

    # Generate the CP to flat output mapping
    M = ph.generate_M(Nco)

    # Compute flat outputs
    for i in range(Nt):
        idx,tsm,dt = ph.get_segment_vars(Ts[i],dT)
        Ai = ph.generate_As([tsm],dt,Nco,Ndr)[0]

        for j in range(Nfo):
            CPj = CP[idx,j,:]
            FO[i,j,:] = Ai@M@CPj

    return Ts,FO

def TsFO_to_tXU(Ts:np.ndarray,FO:np.ndarray,
                m:float,kt:float,
                fext:ExternalForces|None,
                n_mtr:int=4,ndim:int=15) -> np.ndarray:
    """
    Converts a sequence of trajectory sequence times and flat outputs to a state
    vector and control input rollout.

    Args:
        - Ts:       Trajectory time sequence.
        - FO:       Flat outputs.
        - m:        Mass of the quadcopter.
        - kt:       Total motor thrust coefficient.
        - fext:     External forces object.
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
        if fext is None:
            fk = np.zeros(3)
        else:
            pv = FO[0:3,0,:].flatten()
            fk = fext.get_forces(pv)

        # Compute state vector and control input
        xu = fo_to_xu(FO[k,:,:],m,kt,fk,n_mtr)

        # Store in output variable
        tXU[0,k] = Ts[k]
        tXU[1:,k] = xu

    return tXU

def TpCP_to_tXU(Tp:np.ndarray,CP:np.ndarray,
                hz:int=20,m:float=1.0,kt:float=1.0,
                fext:ExternalForces|None=None,
                n_mtr:int=4,ndim:int=15) -> np.ndarray:
    """
    Converts a trajectory spline (defined by Tp,CP) to a state vector and control
    input rollout.

    Args:
        - Tp:       Trajectory segment times.
        - CP:       Control points.
        - hz:       Control loop frequency.
        - m:        Mass of the quadcopter.
        - kt:       Total motor thrust coefficient.
        - fext:     External forces object.
        - n_mtr:    Number of motors
        - ndim:     Number of dimensions in the state vector.

    Returns:
        - tXU:      State vector and control input rollout.
    """

    Ts,FO = TpCP_to_TsFO(Tp,CP,hz)

    tXU = TsFO_to_tXU(Ts,FO,m,kt,fext,n_mtr,ndim)

    return tXU

def KF_to_TpFO(KF:dict,Ndr:int|None) -> tuple[np.ndarray,np.ndarray]:
    """
    Extract the time and flat output values from the trajectory. Automatically
    pads unstated flat outputs with NaN values.

    Args:
        KF:     Dictionary containing the course configuration.

    Returns:
        Tp: Time points.
        FO: Flat output frames.
    """

    # Some useful internal variables
    Nkf = len(KF)
    kf0 = next(iter(KF.values()))["fo"]
    Nfo = len(kf0)

    # Initialize output variables
    Tp = np.zeros(Nkf)

    if Ndr is None:
        # Use FO lists
        FO = [[[] for _ in range(Nfo)] for _ in range(Nkf)]
    else:
        # Check if Ndr is valid
        for kf in KF.values():
            for fo in kf["fo"]:
                if len(fo) > Ndr:
                    raise ValueError("Flat output derivative exceeds proposed Ndr")
        
        # Use FO arrays
        FO = np.full((Nkf,Nfo,Ndr),np.nan)
    
    # Extract time and flat output values
    for i,kf in enumerate(KF.values()):
        Tkf,FOkf = kf["t"],kf["fo"]
        Tp[i] = Tkf

        for j in range(Nfo):
            fokf = FOkf[j]

            for k,fo in enumerate(fokf):
                if isinstance(fo,float):
                    val = fo
                elif fo == None:
                    val = np.nan

                if Ndr is None:
                    FO[i][j].append(val)
                else:
                    FO[i,j,k] = val

    return Tp,FO

def dTPn_to_FO(Ts:np.ndarray,dT:np.ndarray,Pn:np.ndarray,Ndr:int=4) -> tuple[np.ndarray,np.ndarray]:
    """
    Converts a polynomial matrix to a sequence of trajectory segment times and
    flat outputs.

    Args:
        - Ts:  Times along trajectory.
        - dT:  Time intervals.
        - Pn:  Polynomial matrix.
        - Ndr: Number of derivatives.

    Returns:
        - FO:  Flat outputs.
    """

    # Get some useful constants
    Nfo,_,Nco = Pn.shape
    Nt = len(Ts)

    # Initialize output variable
    FO = np.zeros((Nt,Nfo,Ndr))

    # Compute flat outputs
    for i,t in enumerate(Ts):
        idx,tsm,dt = ph.get_segment_vars(t,dT)

        Ai = ph.generate_As([tsm],dt,Nco,Ndr)[0]
        for j in range(Nfo):
            FO[i,j,:] = Ai@Pn[j,idx,:]

    return FO

def dTPn_to_TsFO(dT:np.ndarray,Pn:np.ndarray,Ndr:int,hz:int) -> tuple[np.ndarray,np.ndarray]:
    """
    Get the desired time and flat output values. If hz is None,
    return the keyframe values.

    Args:
        dT:  Time intervals.
        Pn:  Polynomial coefficients.
        Ndr: Number of derivatives.
        hz:  Sampling frequency.

    Returns:
        Ts:  Time values.
        FO:  Flat output values.
    """

    # Generate the time and flat output values
    Tkf = np.hstack((0.0,np.cumsum(dT)))
    Ndt = int(np.ceil(Tkf[-1]*hz))+1

    Ts = np.linspace(0.0,(Ndt-1)/hz,Ndt)
    FO = dTPn_to_FO(Ts,dT,Pn,Ndr)

    return Ts,FO

def dTPn_to_tXU(Ts:np.ndarray,dT:np.ndarray,Pn:np.ndarray,
                m:float,kt:float,
                Fex:ExternalForces|None=None,
                n_mtr:int=4,ndim:int=15) -> np.ndarray:
    """
    Converts a polynomial matrix to a state vector and control input rollout.

    Args:
        - Ts:       Times along trajectory.
        - dT:       Time intervals.
        - Pn:       Polynomial matrix.
        - m:        Mass of the quadcopter.
        - kt:       Total motor thrust coefficient.
        - Fex:      External forces object.
        - n_mtr:    Number of motors
        - ndim:     Number of dimensions in the state vector.

    Returns:
        - tXU:      State vector and control input rollout.
    """

    FO = dTPn_to_FO(Ts,dT,Pn)
    tXU = TsFO_to_tXU(Ts,FO,m,kt,Fex,n_mtr,ndim)

    return tXU

def x_to_T(xcr:np.ndarray) -> np.ndarray:
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