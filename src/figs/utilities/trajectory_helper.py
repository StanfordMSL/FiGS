"""
Helper functions for trajectory data.
"""

import numpy as np
import math

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

def get_M(Ncp:int) -> np.ndarray:
    """
    Generates the M matrix for polynomial interpolation.

    Args:
        - Ncp:    Number of control points.

    Returns:
        - M:      Polynomial interpolation matrix.
    """

    M = np.zeros((Ncp,Ncp))
    for i in range(Ncp):
        ci = (1/(Ncp-1))*i
        for j in range(Ncp):
            M[i,j] = ci**j
    M = np.linalg.inv(M).T

    return M

def get_nt(rt:float,ndr:int,Ncp:int) -> np.ndarray:  
    """
    Generates the normalized time vector based on desired derivative order.

    Args:
        - rt:   Segment time ratio
        - ndr:  Derivative order.
        - Ncp:  Number of control points.

    Returns:
        - nt:   Normalized time vector.
    """

    nt = np.zeros(Ncp)
    for i in range(ndr,Ncp):
        c = math.factorial(i)/math.factorial(i-ndr)
        nt[i] = c*rt**(i-ndr)
    
    return nt

def obedient_quaternion(qcr:np.ndarray,qrf:np.ndarray) -> np.ndarray:
    """
    Ensure that the quaternion is well-behaved (unit norm and closest to reference).
    
    Args:
        - qcr:    Current quaternion.
        - qrf:    Previous quaternion.

    Returns:
        - qcr:     Closest quaternion to reference.
    """
    qcr = qcr/np.linalg.norm(qcr)

    if np.dot(qcr,qrf) < 0:
        qcr = -qcr

    return qcr

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

# def ts_to_xu(tcr:float,Tp:float,CP:np.ndarray,m:float,kt:float,
#              Fext:ExternalForces|None,
#              n_mtr:int=4) -> np.ndarray:
#     """
#     Converts a trajectory spline (defined by tf,CP) to a state vector and control input.
#     Returns just x if quad is None.

#     Args:
#         - tcr:      Current segment time.
#         - Tp:       Trajectory segment final time.
#         - CP:       Control points.
#         - m:        Mass of the quadcopter.
#         - kt:       Total motor thrust coefficient.
#         - Fext:     External forces object.
#         - n_mtr:    Number of motors

#     Returns:
#         xu:    State vector and control input.
#     """
#     # Convert to Flat Output
#     fo = ts_to_fo(tcr,Tp,CP)

#     # Compute External Forces (if any)
#     if Fext is None:
#         fext = np.zeros(3)
#     else:
#         pv = fo[0:3,0:2].flatten()
#         fext = Fext.get_forces(pv)

#     return fo_to_xu(fo,m,kt,fext,n_mtr)

# def TS_to_xu(tcr:float,Tps:np.ndarray,CPs:np.ndarray,m:float,kt:float,
#              Fext:ExternalForces|None,
#              n_mtr:int=4) -> np.ndarray:
#     """
#     Extracts the state and input from a sequence of trajectory splines (defined by
#     Tps,CPs). Returns just x if quad is None.

#     Args:
#         - tcr:      Current segment time.
#         - Tps:      Trajectory segment times.
#         - CPs:      Trajectory control points.
#         - m:        Mass of the quadcopter.
#         - kt:       Total motor thrust coefficient.
#         - Fext:     External forces object.
#         - n_mtr:    Number of motors

#     Returns:
#         xu:    State vector and control input.
#     """
#     idx = np.max(np.where(Tps < tcr)[0])
    
#     if idx == len(Tps)-1:
#         tcr = Tps[-1]
#         t0,tf = Tps[-2],Tps[-1]
#         CPk = CPs[-1,:,:]
#     else:
#         t0,tf = Tps[idx],Tps[idx+1]
#         CPk = CPs[idx,:,:]

#     xu = ts_to_xu(tcr-t0,tf-t0,CPk,m,kt,Fext,n_mtr)

#     return xu

# def TS_to_tXU(hz:int,Tps:np.ndarray,CPs:np.ndarray,m:float,kt:float,
#               Fext:ExternalForces|None,
#               n_mtr:int=4) -> np.ndarray:
#     """
#     Converts a sequence of trajectory splines (defined by Tps,CPs) to a trajectory
#     rollout. Returns just tX if quad is None.

#     Args:
#         - hz:       Control loop frequency.
#         - Tps:      Trajectory segment times.
#         - CPs:      Trajectory control points.
#         - m:        Mass of the quadcopter.
#         - kt:       Motor thrust coefficient.
#         - Fext:     External forces object.
#         - n_mtr:    Number of motors

#     Returns:
#         - tXU:  State vector and control input rollout.
#     """
#     Nt = int((Tps[-1]-Tps[0])*hz+1)

#     idx = 0
#     for k in range(Nt):
#         tk = Tps[0]+k/hz

#         if tk > Tps[idx+1] and idx < len(Tps)-2:
#             idx += 1

#         t0,tf = Tps[idx],Tps[idx+1]
#         CPk = CPs[idx,:,:]
#         xu = ts_to_xu(tk-t0,tf-t0,CPk,m,kt,Fext,n_mtr)

#         if k == 0:
#             ntxu = len(xu)+1
#             tXU = np.zeros((ntxu,Nt))
#         else:
#             xu[6:10] = obedient_quaternion(xu[6:10],tXU[7:11,k-1])
                
#         tXU[0,k] = tk
#         tXU[1:,k] = xu

#     return tXU







# def RO_to_tXU(RO:tuple[np.ndarray,np.ndarray,np.ndarray]) -> np.ndarray:
    """
    Converts a tuple of rollouts to a state vector and control input rollout.

    Args:
        - RO:    Rollout tuple (Tro,Xro,Uro).

    Returns:
        - tXU:   State vector and control input rollout.
    """
    # Unpack the tuple
    Tro,Xro,Uro = RO

    # Stack the arrays
    if Uro.shape[1] != Xro.shape[1]:
        Uro = np.hstack((Uro,Uro[:,-1].reshape(-1,1)))

    tXU = np.vstack((Tro,Xro,Uro))

    return tXU