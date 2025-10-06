"""
Helper functions for sous vide.
"""

import numpy as np
import figs.utilities.transform_helper as th
from scipy.spatial.transform import Rotation as R

def compute_prms(frame:dict[str,np.ndarray,str|int|float]) -> list:
    """
    Computes the frame parameters (mass, thrust coefficient, normalized thrust gain).

    Args:
        frame:  Frame configuration.

    Returns:
        params: Frame parameters (mass, thrust coefficient, normalized thrust gain).
    """

    # Unpack variables
    n_mtr = frame["number_of_rotors"]
    m_fr = frame["mass"]
    k_fr = frame["motor_thrust_coeff"]

    # Compute mass normalized thrust
    c_fr = (k_fr*n_mtr)/m_fr

    # Params
    params = [m_fr,k_fr,c_fr]

    return params

def compute_Wrs(Xro:np.ndarray,Uro:np.ndarray,Wro:np.ndarray,
             frame:dict[str,np.ndarray,str|int|float],
             bframe:dict[str,np.ndarray,str|int|float]) -> np.ndarray:
    """
    Computes the resultant force/torques acting on the frame.

    Args:
        Xro:    State vector.
        Uro:    Control input vector.
        Wro:    External wrench vector.
        frame:  Frame configuration.
        bframe: Base frame configuration.

    Returns:
        Wrs:   Resultant forces array.
    """

    # Some useful constants
    g = np.array([0,0,9.81])        # Gravity vector
    zb = np.array([0.0,0.0,1.0])    # Z-axis unit vector

    # Unpack variables
    Ndt = Uro.shape[0]
    n_mtr = frame["number_of_rotors"]
    m_fr,m_bs = frame["mass"],bframe["mass"]
    k_fr,k_bs = frame["motor_thrust_coeff"],bframe["motor_thrust_coeff"]
    
    # Compute the resultant forces
    Wrs = np.zeros_like(Wro)
    for i in range(Ndt):
        # Unpack data
        xcr = Xro[i,:]
        ucr = Uro[i,:]
        fcr = Wro[i,0:3]

        # Compute rotation matrix
        Rb2w = R.from_quat(xcr[6:10]).as_matrix()

        # Compute forces
        f_dgv = (m_fr-m_bs)*g                   # Difference from gravity
        f_dth = (k_fr-k_bs)*n_mtr*ucr[0]*zb     # Difference from thrust

        Wrs[i,0:3] = f_dgv + Rb2w@f_dth + fcr
        Wrs[i,3:6] = Wro[i,3:6]                 # Copy the torque vector

    return Wrs

def compute_FOro(Tro:np.ndarray,Xro:np.ndarray,Uro:np.ndarray,
               Wro:np.ndarray,frame:dict[str,np.ndarray,str|int|float]) -> np.ndarray:
    """
    Computes the flat output sequence given a trajectory rollout

    Args:
        Tro:    Time vector.
        Xro:    State vector.
        Uro:    Control input vector.
        Wro:    Force/Torque vector.
        frame:  Frame configuration.
    
    Returns:
        FO:    Flat output rollout sequence.
        
    """

    # Unpack variables
    tXU = np.hstack((Tro[:-1].reshape((-1,1)),Xro[:-1,:],Uro))
    Fro = Wro[:,0:3]            # Extract only the force part
    
    # Compute the flat output
    _,FO = th.tXU_to_TsFO(tXU,Fro,frame)

    return FO

def compute_Lro(Tro:np.ndarray,Xro:np.ndarray,Uro:np.ndarray,
               pt_w:np.ndarray,frame:dict[str,np.ndarray,str|int|float]) -> np.ndarray:
    """
    Computes the localization sequence given a trajectory rollout

    Args:
        Tro:    Time vector.
        Xro:    State vector.
        Uro:    Control input vector.
        pt_w:   Target position in world frame.
        frame:  Frame configuration.
    
    Returns:
        Lro:    Localization rollout sequence.
    """

    # Extract some useful variables
    Tc2b = np.array(frame['camera_to_body_transform'])
    pp_b = np.array(frame["probe_position"])
    camera = frame['camera']
    fx,fy = camera["fx"],camera["fy"]
    nW,nH = camera["width"]/2,camera["height"]/2
    Nro = Tro.shape[0]

    for i in range(Nro):
        pb_w = Xro[i,0:3]
        vb_w = Xro[i,3:6]
        qx,qy,qz,qw = Xro[i,6:10]
        wx,wy,wz = Uro[i,1:4]

        # Precompute some useful variables
        Tb2c = np.linalg.inv(Tc2b)

        # Extract transforms
        Rb2c = Tb2c[:3,:3]
        pb_c = Tb2c[0:3,3]

        # Compute skew symmetric matrix
        Wb = np.array([
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0]
        ])
        
        # Compute rotation matrix from quaternion
        Rb2w = np.array([
            [1.0-2.0*(qy**2+qz**2), 2.0*(qx*qy-qw*qz), 2.0*(qx*qz+qw*qy)],
            [2.0*(qx*qy+qw*qz), 1.0-2*(qx**2+qz**2), 2.0*(qy*qz-qw*qx)],
            [2.0*(qx*qz-qw*qy), 2.0*(qy*qz+qw*qx), 1.0-2.0*(qx**2+qy**2)]]
        )
        Rw2b = Rb2w.T

        # Body frame velocities
        vb_b = Rw2b@vb_w
        
        # Compute camera frame states
        rt_w = pt_w - pb_w                  # Target in world frame
        pt_b = Rw2b@(rt_w)                  # Target in body frame
        pt_p = pt_b - pp_b                  # Target in probe frame
        pt_c = Rb2c@pt_b + pb_c             # Target in camera frame
    
        # Compute orientation metrics
        head = np.arctan2(2*(qw*qz + qx*qy), 1-2*(qy*qy + qz*qz))
        azim = np.arctan2(pt_c[0],pt_c[2])
        elev = np.arctan2(pt_c[1],pt_c[2])
        u_n = fx*pt_c[0]/(nW*(pt_c[2]+1e-5))
        v_n = fy*pt_c[1]/(nH*(pt_c[2]+1e-5))

        # Populate the localization array
        lro = np.hstack((
                pb_w,
                pt_p,
                vb_b,
                head,
                u_n,v_n,
                azim,elev
            ))
        
        if i == 0:
            Lro = np.zeros((Nro,lro.shape[0]))

        Lro[i,:] = lro

        return Lro