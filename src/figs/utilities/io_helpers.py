"""
Helper functions for sous vide.
"""

import numpy as np
import figs.utilities.transform_helper as th
from scipy.spatial.transform import Rotation as R
from figs.visualize import rich_visuals as rv
def compute_params(frame:dict[str,np.ndarray,str|int|float]) -> list:
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

def compute_resultants(Xro:np.ndarray,Uro:np.ndarray,Wro:np.ndarray,
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

def compute_flatoutputs(Tro:np.ndarray,Xro:np.ndarray,Uro:np.ndarray,
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

def compute_object_data(Tro:np.ndarray,Xro:np.ndarray,
                         obj:dict[str,dict[str,list[float|int]]],
                         frame:dict[str,np.ndarray,str|int|float],
                         Ndt:int=4,nMag:float=3.14,
                         rsW:int=256,rsH:int=256,
                         crW:int=224,crH:int=224) -> np.ndarray:
    """
    Computes the id, localization and bounding box sequence given a trajectory rollout

    Args:
        Tro:    Time vector.
        Xro:    State vector.
        obj:    Object to track.
        frame:  Frame configuration.
        Ndt:    Number of localization/bounding box dimensions.
        nMag:   Normalization magnitude.
        rsW:    Resized image width.
        rsH:    Resized image height.
        crW:    Cropped image width.
        crH:    Cropped image height.
    
    Returns:
        Ido:    Object IDs sequence.
        Lro:    Localization sequence.
        Bro:    Bounding box sequence.
    """

    # Extract some useful variables
    Tc2b = np.array(frame['camera_to_body_transform'])
    camera = frame['camera']
    fx,fy = camera["fx"],camera["fy"]
    cx,cy = camera["cx"],camera["cy"]
    nW,nH = camera["width"],camera["height"]
    Nro = Tro.shape[0]

    # Precompute some useful variables
    Tb2c = np.linalg.inv(Tc2b)
    Rb2c = Tb2c[:3,:3]
    pb_c = Tb2c[0:3,3].reshape((-1,1))

    # Get object and bounding box positions in world frame
    pobj_w = np.array(obj["position"]).reshape((-1,1))
    pbbx_w = (np.array(obj["position"]) + np.array(obj["boundary"])).T
    o_id = obj["id"]

    # Compute bounding boxes
    Ido = np.zeros(Nro)
    Lro = np.zeros((Nro,Ndt))
    Bro = np.zeros((Nro,Ndt))
    for i in range(Nro):
        # Extract drone pose
        pb_w = Xro[i,0:3].reshape((-1,1))
        qx,qy,qz,qw = Xro[i,6:10]

        # Compute rotation matrix from quaternion
        Rb2w = np.array([
            [1.0-2.0*(qy**2+qz**2), 2.0*(qx*qy-qw*qz), 2.0*(qx*qz+qw*qy)],
            [2.0*(qx*qy+qw*qz), 1.0-2*(qx**2+qz**2), 2.0*(qy*qz-qw*qx)],
            [2.0*(qx*qz-qw*qy), 2.0*(qy*qz+qw*qx), 1.0-2.0*(qx**2+qy**2)]]
        )
        Rw2b = Rb2w.T
        Rw2c = Rb2c@Rw2b
        
        # Determine relative object position in camera frame
        pob_w = np.hstack((pobj_w,pbbx_w))  # Object + bounding box (3x5)
        rob_w = pob_w - pb_w                # Object + bounding box in world frame (3x5)
        rob_c = Rw2c@rob_w+pb_c             # Object + bounding box in camera frame (3x5)

        # Compute orientation metrics and normalize
        rd_ob = np.linalg.norm(rob_c)
        hd_ob = np.arctan2(2*(qw*qz + qx*qy), 1-2*(qy*qy + qz*qz))
        az_ob = np.arctan2(rob_c[0,0],rob_c[2,0])
        el_ob = np.arctan2(rob_c[1,0],rob_c[2,0])

        rd_n,hd_n = rd_ob/nMag,hd_ob/nMag
        az_n,el_n = az_ob/nMag,el_ob/nMag

        # Compute the pixel coordinates and pixel dimensions of bounding box
        Uob = fx*rob_c[0,:]/(rob_c[2,:]+1e-5)+cx
        Vob = fy*rob_c[1,:]/(rob_c[2,:]+1e-5)+cy
        
        # Pass through feature extraction normalization
        rsUob,rsVob = Uob * (rsW / nW), Vob * (rsH / nH)
        crUob,crVob = rsUob - (rsW - crW) / 2, rsVob - (rsH - crH) / 2
        
        # Normalize to [0,1] for locNet
        nUob,nVob = crUob / crW, crVob / crH
        
        up_n,vp_n = nUob[0], nVob[0]
        wp_n,hp_n = np.max(nUob) - np.min(nUob), np.max(nVob) - np.min(nVob)
    
        wp_n = np.clip(wp_n,0.0,1.0)
        hp_n = np.clip(hp_n,0.0,1.0)

        # Check and store if object is visible
        if ((0.0 <= up_n <= 1.0) and (0.0 <= vp_n <= 1.0) and (rob_c[2,0] > 0.0)):
            Ido[i] = o_id
            Lro[i,:] = np.array([rd_n,hd_n,az_n,el_n])
            Bro[i,:] = np.array([up_n,vp_n,wp_n,hp_n])

    return Ido,Lro,Bro