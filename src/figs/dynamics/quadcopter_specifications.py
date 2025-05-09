import numpy as np
from typing import Dict,Union

def generate_specifications(
        drn_prms:Dict[str,Union[float,np.ndarray]],
        name:str='the_shepherd',nx:int=10,nu:int=4
        ) -> Dict["str",Union[str,int,float,np.ndarray]]:
    """
    Generate a dictionary with the full drone specifications. Some drone specifications are derived
    from input parameters but are queried frequently. To save computation time, they are precomputed
    and stored in the dictionary.
    
    Args:
        drn_prms:       Dictionary containing the drone parameters.
        name:           Name of the quadcopter.
        nx:             Number of states in the system.
        nu:             Number of inputs in the system.

    Variable Inputs:
        - mass: Mass of the quadcopter
        - massless_inertia: Inertia of the quadcopter without mass
        - arm_front: Length of the front arm
        - arm_back: Length of the back arm
        - motor_thrust_coeff: Coefficient for thrust
        - motor_torque_coeff: Coefficient for torque
        - number_of_rotors: Number of rotors
        - camera_to_body_transform: Transformation matrix from camera to body
        - camera: Camera specifications (fx, fy, cx, cy)
        - camera["fx"]: Focal length in x direction
        - camera["fy"]: Focal length in y direction
        - camera["cx"]: Optical center in x direction
        - camera["cy"]: Optical center in y direction
        - camera["height"]: Height of the camera
        - camera["width"]: Width of the camera
        - camera["channels"]: Number of channels in the camera
    
    Drone Specifications:
        - m: Mass of the quadcopter
        - I: Inertia matrix of the quadcopter
        - lf: Length of the front arm
        - lb: Length of the back arm
        - kt: Thrust coefficient
        - kq: Torque coefficient
        - g: Gravitational acceleration
        - nx: Number of states in the system
        - nu: Number of inputs in the system
        - Nrtr: Number of rotors
        - Tc2b: Transformation matrix from camera to body
        - K: Camera intrinsic matrix
        - camera: Camera specifications (height, width, channels, fx, fy, cx, cy)
        - rgb_dim: Dimensions of the RGB image
        - dpt_dim: Dimensions of the depth image
        - Iinv: Inverse inertia matrix of the quadcopter
        - fMw: Force to motor weight matrix
        - wMf: Motor to force weight matrix
        - kt_sum: Total thrust coefficient
        - name: Name of the quadcopter
    
    """

    # Unpack the params dictionary ===========================================
    m,Impp = drn_prms["mass"],drn_prms["massless_inertia"]
    lf,lb = drn_prms["arm_front"],drn_prms["arm_back"]
    kt,kq = drn_prms["motor_thrust_coeff"],drn_prms["motor_torque_coeff"]
    Nrtr = drn_prms["number_of_rotors"]
    Tc2b = drn_prms["camera_to_body_transform"]
    camera = drn_prms["camera"]
    fx,fy = camera["fx"], camera["fy"]
    cx,cy = camera["cx"], camera["cy"]
    height,width,channels = camera["height"], camera["width"], camera["channels"]

    # Initialize the dictionary
    quad = {}
    
    # Variable Quadcopter Constants ==========================================

    # F=ma, T=Ia Variables
    quad["m"],quad["I"] = m,m*np.diag(Impp)
    quad["lf"] = np.array(lf)
    quad["lb"] = np.array(lb)
    quad["kt"],quad["kq"] = kt, kq
    quad["g"] = 9.81
    
    # Model Constants
    quad["nx"],quad["nu"] = nx,nu
    quad["Nrtr"] = Nrtr
    quad["Tc2b"] = np.array(Tc2b)
    quad["K"] = np.array([
        [ fx, 0.0,  cx],
        [0.0,  fy,  cy],
        [0.0, 0.0, 1.0]
    ])

    quad["camera"] = camera
    quad["rgb_dim"] = (height,width,channels)
    quad["dpt_dim"] = (height,width,1)

    # Derive Quadcopter Constants
    fMw = kt*np.array([
            [  -1.0,   -1.0,   -1.0,    -1.0],
            [-lf[1],  lf[1],  lb[1],  -lb[1]],
            [ lf[0], -lb[0],  lf[0],  -lb[0]],
            [ kt/kq,  kt/kq, -kt/kq, -kt/kq]])
    
    quad["Iinv"] = np.diag(1/(m*np.array(Impp)))
    quad["fMw"]  = fMw
    quad["wMf"]  = np.linalg.inv(fMw)
    quad["kt_sum"] = kt*Nrtr

    # name
    quad["name"] = name
    
    return quad