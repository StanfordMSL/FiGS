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

def xu_to_fo(xu:np.ndarray,m:float,kt:float,
            fext:np.ndarray,
            n_mtr:int=4,Nfo:int=4,Ndr:int=3,
            use_list:bool=False) -> np.ndarray:
    """
    Converts a state vector to approximation of flat output vector.

    Args:
        xu:         State vector
        m:          Mass of the quadcopter.
        kt:         Total motor thrust coefficient.
        fext:       External forces vector.
        n_mtr:      Number of motors
        Nfo:        Number of flat outputs.
        Ndr:        Number of derivatives.
        use_list:   Use list instead of numpy array.

    Returns:
        fo:         Flat outputs.
    """

    # Unpack state vector
    pk = xu[0:3]
    vk = xu[3:6]
    qk = xu[6:10]
    uk = xu[10:14]

    # Define Gravity
    gt = np.array([0.00,0.00,9.81])

    # Initialize output
    fo = np.zeros((Nfo,Ndr))

    # Compute position terms
    fo[0:3,0] = pk
    fo[0:3,1] = vk

    # Compute acceleration terms
    Rk = Rotation.from_quat(qk).as_matrix()
    xbt,ybt,zbt = Rk[:,0],Rk[:,1],Rk[:,2]
    c = (uk[0]*(n_mtr*kt))/m

    fo[0:3,2] = c*zbt+gt+(fext/m)

    # Compute yaw term
    psi = np.arctan2(Rk[1,0], Rk[0,0])

    fo[3,0]  = psi

    # Compute yaw rate term
    xct = np.array([np.cos(psi), np.sin(psi), 0])     # Intermediate frame x vector
    yct = np.array([-np.sin(psi), np.cos(psi), 0])    # Intermediate frame y vector
    
    B1 = c
    B3 = -yct.T@zbt
    C3 = np.linalg.norm(np.cross(yct,zbt))
    D1 = uk[2]*(B1*C3)/C3
    D3 = (uk[3]*(B1*C3)+(B3*D1))/B1

    psid = D3/(xct.T@xbt)

    fo[3,1] = psid

    # Convert to list if required
    if use_list:
        fo[3,2] = None
        fo = fo.tolist()
        fo = [[x for x in sublist if ~np.isnan(x)] for sublist in fo]

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
    
    # Initialize output variable
    Nt = int((Tp[-1]-Tp[0])*hz+1)
    Ts = np.linspace(Tp[0],Tp[-1],Nt)
    FO = np.zeros((Nt,Nfo,Ndr))

    # Generate A matrices
    As = ph.generate_As(Ts,Nco,Ndr)

    # Generate the CP to flat output mapping
    M = ph.generate_M(Nco)

    # Compute flat outputs
    for i in range(Nt):
        Ai = As[Ts[i]]
        idx = ph.get_segment_index(Ts[i],Tp)
        for j in range(Nfo):
            CPj = CP[idx,j,:]
            FO[i,j,:] = Ai@M@CPj

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

def TpCP_to_tXU(Tp:np.ndarray,CP:np.ndarray,
                hz:int=20,m:float=1.0,kt:float=1.0,
                Fext:ExternalForces|None=None,
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
        - Fext:     External forces object.
        - n_mtr:    Number of motors
        - ndim:     Number of dimensions in the state vector.

    Returns:
        - tXU:      State vector and control input rollout.
    """

    Ts,FO = TpCP_to_TsFO(Tp,CP,hz)

    tXU = TsFO_to_tXU(Ts,FO,m,kt,Fext,n_mtr,ndim)

    return tXU

def KF_to_TpFO(KF:dict,Ndr:int,impose:bool=False) -> tuple[np.ndarray,np.ndarray]:
    """
    Extract the time and flat output values from the trajectory. Automatically
    pads unstated flat outputs with NaN values.

    Args:
        KF:     Dictionary containing the course configuration.
        impose: Impose continuity on undefined flat outputs.

    Returns:
        Tp: Time points.
        FO: Flat output frames.
    """

    # Some useful internal variables
    Nkf = len(KF)
    kf0 = next(iter(KF.values()))["fo"]
    Nfo = len(kf0)

    # Check if number of flat output derivatives is reasonable
    for kf in KF.values():
        for fo in kf["fo"]:
            if len(fo) > Ndr:
                raise ValueError("Flat output derivative exceeds proposed Ndr")
    
    # Condition undefined flat outputs
    if impose:
        limit_fill,stage_fill = 0.0,np.nan
    else:
        limit_fill,stage_fill = "no","no"
    #TODO replace no with a float type.

    # Initialize output variables
    Tp = np.zeros(Nkf)
    FO = np.full((Nkf,Nfo,Ndr),stage_fill)
    FO[[0,-1],:,:] = limit_fill

    # Extract time and flat output values
    for i,kf in enumerate(KF.values()):
        Tkf,FOkf = kf["t"],kf["fo"]
        Tp[i] = Tkf

        for j in range(Nfo):
            fokf = FOkf[j]

            for k,fo in enumerate(fokf):
                if isinstance(fo,float):
                    FO[i,j,k] = fo
                elif fo == None:
                    FO[i,j,k] = np.nan

    return Tp,FO

def TpPn_to_CP(Tp:np.ndarray,Pn:np.ndarray) -> np.ndarray:
    """
    Converts a polynomial matrix to control points.

    Args:
        Tp: Time points.
        Pn: Polynomial matrix.

    Returns:
        CP: Control points.
    """

    # Get some useful constants
    Nfo,Nsm,Nco = Pn.shape
    
    # Initialize output variable
    CP = np.zeros((Nsm,Nfo,Nco))
    for i in range(Nsm):
        Ti = np.linspace(Tp[i],Tp[i+1],Nco)
        As = ph.generate_As(Ti,Nco)

        for j in range(Nco):
            Aj = As[Ti[j]]
            for k in range(Nfo):
                CP[i,k,j] = Aj@Pn[k,i,:]
    print('ha',np.around(CP,3))
    return CP

def TpPn_to_TsFO(Tp:np.ndarray,Pn:np.ndarray,
                 hz:int=20,Ndr:int=4) -> tuple[np.ndarray,np.ndarray]:
    """
    Converts a polynomial matrix to a sequence of trajectory segment times and
    flat outputs.

    Args:
        - Tp:  Trajectory segment times.
        - Pn:  Polynomial matrix.
        - hz:  Control loop frequency.
        - Ndr: Number of derivatives.

    Returns:
        - Ts:  Trajectory time sequence.
        - FO:  Flat outputs.
    """

    # Get some useful constants
    Nfo,_,Nco = Pn.shape
    Nt = int((Tp[-1]-Tp[0])*hz+1)
    
    # Initialize output variable
    Ts = np.linspace(Tp[0],Tp[-1],Nt)
    FO = np.zeros((Nt,Nfo,Ndr))

    # Compute flat outputs
    for i in range(Nt):
        idx0 = ph.get_segment_index(Ts[i],Tp)
        idx1 = idx0+1
        t0,t1 = Tp[idx0],Tp[idx1]

        Ai = ph.generate_As([Ts[i]],t0,t1,Nco,Ndr)[0]
        for j in range(Nfo):
            FO[i,j,:] = Ai@Pn[j,idx0,:]

    idx = np.where(Ts < Tp[1])[0][-1]+1
    print(Ts[  0],'\n',np.around(FO[  0,:,0:4],2))
    print(Ts[idx],'\n',np.around(FO[idx,:,0:4],2))
    print(Ts[ -1],'\n',np.around(FO[ -1,:,0:4],2))
    return Ts,FO

def TpPn_to_tXU(Tp:np.ndarray,Pn:np.ndarray,
                hz:int=20,m:float=1.0,kt:float=1.0,
                Fext:ExternalForces|None=None,
                n_mtr:int=4,ndim:int=15) -> np.ndarray:
    """
    Converts a polynomial matrix to a state vector and control input rollout.

    Args:
        - Tp:       Trajectory segment times.
        - Pn:       Polynomial matrix.
        - hz:       Control loop frequency.
        - m:        Mass of the quadcopter.
        - kt:       Total motor thrust coefficient.
        - Fext:     External forces object.
        - n_mtr:    Number of motors
        - ndim:     Number of dimensions in the state vector.

    Returns:
        - tXU:      State vector and control input rollout.
    """

    # Convert polynomial matrix to trajectory segment times and flat outputs
    Ts,FO = TpPn_to_TsFO(Tp,Pn,hz)

    # Convert trajectory segment times and flat outputs to state vector and
    # control input rollout
    tXU = TsFO_to_tXU(Ts,FO,m,kt,Fext,n_mtr,ndim)

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