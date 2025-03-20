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

    Variable Constants:
        - m: Mass of the quadcopter (kg)
        - Impp: Massless Inertia tensor of the quadcopter (m^2)
        - lf: [x,y] distance from the center of mass to the front motors
        - lb: [x,y] distance from the center of mass to the back motors
        - fn: Normalized motor force gain
        - tG: Motor torque gain (after normalizing by fn)
    
    Fixed Constants:
        - nx_fs: Number of states for the full state model
        - nu_fs: Number of inputs for the full state model
        - nx_br: Number of states for the body rate model
        - nu_br: Number of inputs for the body rate model
        - nu_va: Number of inputs for the vehicle attitude model
        - n_mtr: Number of motors
        - lbu: Lower bound on the inputs
        - ubu: Upper bound on the inputs
        - tf: Time horizon for the MPC
        - hz: Frequency of the MPC
        - Qk: Stagewise State weight matrix for the MPC
        - Rk: Stagewise Input weight matrix for the MPC
        - QN: Terminal State weight matrix for the MPC
        - Ws: Search weights for the MPC (to get xv_ds)

    Derived Constants:
        - Iinv: Inverse of the inertia tensor
        - fMw: Matrix to convert from forces to moments
        - wMf: Matrix to convert from moments to forces
        - tn: Total normalized thrust

    Misc:
        - name: Name of the quadcopter

    The default values are for the Iris used in the Gazebo SITL simulation.
    
    """

    # Unpack the params dictionary ===========================================
    m,Impp = drn_prms["mass"],drn_prms["massless_inertia"]
    lf,lb = drn_prms["arm_front"],drn_prms["arm_back"]
    fn,tG = drn_prms["force_normalized"],drn_prms["torque_gain"]
    n_rtr = drn_prms["number_of_rotors"]
    T_c2b = drn_prms["camera_to_body_transform"]
    camera = drn_prms["camera"]

    # Initialize the dictionary
    quad = {}
    
    # Variable Quadcopter Constants ==========================================

    # F=ma, T=Ia Variables
    quad["m"],quad["I"] = m,m*np.diag(Impp)
    quad["lf"] = np.array(lf)
    quad["lb"] = np.array(lb)
    quad["fn"],quad["tg"] = fn, tG

    # Model Constants
    quad["nx"],quad["nu"] = nx,nu
    quad["n_rtr"] = n_rtr
    quad["T_c2b"] = np.array(T_c2b)
    quad["camera"] = camera

    # Derive Quadcopter Constants
    fMw = fn*np.array([
            [   -1.0,   -1.0,   -1.0,   -1.0],
            [ -lf[1],  lf[1],  lb[1], -lb[1]],
            [  lf[0], -lb[0],  lf[0], -lb[0]],
            [     tG,     tG,    -tG,    -tG]])
    
    quad["Iinv"] = np.diag(1/(m*np.array(Impp)))
    quad["fMw"] = fMw
    quad["wMf"] = np.linalg.inv(fMw)
    quad["tn"] = fn*n_rtr

    # name
    quad["name"] = name
    
    return quad