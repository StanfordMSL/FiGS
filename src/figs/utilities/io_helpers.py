"""
Helper functions for sous vide.
"""

import numpy as np
import figs.utilities.transform_helper as th
from scipy.spatial.transform import Rotation as R

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

def compute_detectors(Tro:np.ndarray,Xro:np.ndarray,Obj:dict[str,dict[str,list[float|int]]],frame:dict[str,np.ndarray,str|int|float]) -> np.ndarray:
    """
    Computes the detection sequence given a trajectory rollout

    Args:
        Tro:    Time vector.
        Xro:    State vector.
        Obj:    Objects to track.
        frame:  Frame configuration.
    
    Returns:
        Bro:    Detection sequence.
    """

    # Extract some useful variables
    Npt,Nbb = 3,4
    Tc2b = np.array(frame['camera_to_body_transform'])
    camera = frame['camera']
    fx,fy = camera["fx"],camera["fy"]
    cx,cy = camera["cx"],camera["cy"]
    nW,nH = camera["width"],camera["height"]
    Nro = Tro.shape[0]
    Nobj = len(Obj)

    # Precompute some useful variables
    Tb2c = np.linalg.inv(Tc2b)

    Rb2c = Tb2c[:3,:3]
    pb_c = Tb2c[0:3,3].reshape((-1,1))

    # Get object and bounding box positions
    Pobjs = np.zeros((Nobj,Npt))
    Pbbxs = np.zeros((Nobj,Nbb,Npt))
    for i,obj in enumerate(Obj.values()):
        Pobjs[i,:] = np.array(obj["position"])
        Pbbxs[i,:,:] = Pobjs[i,:] + np.array(obj["boundary"])

    # Compute bounding boxes
    Bro = np.zeros((Nro,Nobj,1+Nbb))
    for i in range(Nro):
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
        
        # Compute bounding boxes
        bro = np.zeros_like(Bro[i,:,:])
        for j in range(Nobj):
            # Determine object and bounding box in camera frame
            pobj_w = Pobjs[j,:].reshape((-1,1)) # Object position (3x1)
            pbbx_w = Pbbxs[j,:,:].T             # Object bounding box (3x4)
            pob_w = np.hstack((pobj_w,pbbx_w))  # Object + bounding box (3x5)

            rob_w = pob_w - pb_w                # Object + bounding box in world frame (3x5)
            rob_c = Rw2c@rob_w+pb_c             # Object + bounding box in camera frame (3x5)
  
            # Compute the pixel coordinates
            UVob = np.zeros((2,5))
            UVob[0,:] = fx*rob_c[0,:]/(rob_c[2,:]+1e-5)+cx
            UVob[1,:] = fy*rob_c[1,:]/(rob_c[2,:]+1e-5)+cy

            # Normalize and account for center crop
            offset,scale = 16/256,224/256

            nUVob = np.zeros((2,5))
            nUVob[0,:] = UVob[0,:]/nW
            nUVob[1,:] = UVob[1,:]/nH
            nUVob = (nUVob - offset)/scale

            # Check if object is visible
            if 0.0 <= nUVob[0,0] <= 1.0 and 0.0 <= nUVob[1,0] <= 1.0 and rob_c[2,0] > 0.0:
                bro[j,0] = 1.0                                    # Object is visible
                bro[j,1] = nUVob[0,0]                             # Object center u
                bro[j,2] = nUVob[1,0]                             # Object center v
                bro[j,3] = np.max(nUVob[0,:])-np.min(nUVob[0,:])  # Object width
                bro[j,4] = np.max(nUVob[1,:])-np.min(nUVob[1,:])  # Object height
            
        # Store the bounding boxes
        Bro[i,:,:] = bro

    return Bro

def compute_localization(Tro:np.ndarray,Xro:np.ndarray,Obj:dict[str,dict[str,list[float|int]]],frame:dict[str,np.ndarray,str|int|float]) -> np.ndarray:
    """
    Computes the localization sequence given a trajectory rollout

    Args:
        Tro:    Time vector.
        Xro:    State vector.
        Obj:    Objects to track.
        frame:  Frame configuration.
    
    Returns:
        Lro:    Localization sequence.
    """

    # Extract some useful variables
    Npt,Nbb = 3,4
    Tc2b = np.array(frame['camera_to_body_transform'])
    Nro = Tro.shape[0]
    Nobj = len(Obj)

    # Precompute some useful variables
    Tb2c = np.linalg.inv(Tc2b)

    Rb2c = Tb2c[:3,:3]
    pb_c = Tb2c[0:3,3]

    # Get object and bounding box positions
    Pobjs = np.zeros((Nobj,Npt))
    for j,obj in enumerate(Obj.values()):
        Pobjs[j,:] = np.array(obj["position"])

    # Compute bounding boxes
    Lro = np.zeros((Nro,Nobj,Nbb))
    for i in range(Nro):
        pb_w = Xro[i,0:3]
        qx,qy,qz,qw = Xro[i,6:10]

        # Compute rotation matrix from quaternion
        Rb2w = np.array([
            [1.0-2.0*(qy**2+qz**2), 2.0*(qx*qy-qw*qz), 2.0*(qx*qz+qw*qy)],
            [2.0*(qx*qy+qw*qz), 1.0-2*(qx**2+qz**2), 2.0*(qy*qz-qw*qx)],
            [2.0*(qx*qz-qw*qy), 2.0*(qy*qz+qw*qx), 1.0-2.0*(qx**2+qy**2)]]
        )
        Rw2b = Rb2w.T
        Rw2c = Rb2c@Rw2b
        
        # Compute bounding boxes
        lro = np.zeros_like(Lro[i,:,:])
        for j in range(Nobj):
            # Compute camera frame states
            rob_w = Pobjs[j,:] - pb_w           # Object in world frame (3,)
            rob_c = Rw2c@rob_w+pb_c             # Object in camera frame (3,)
            
            # Compute orientation metrics
            rdis = np.linalg.norm(rob_c)
            head = np.arctan2(2*(qw*qz + qx*qy), 1-2*(qy*qy + qz*qz))
            azim = np.arctan2(rob_c[0],rob_c[2])
            elev = np.arctan2(rob_c[1],rob_c[2])

            # Check if object is visible
            nMag = 3.14
            lro[j,0] = rdis/nMag
            lro[j,1] = head/nMag
            lro[j,2] = azim/nMag
            lro[j,3] = elev/nMag

        # Store the bounding boxes
        Lro[i,:,:] = lro

    return Lro