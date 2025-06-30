import numpy as np
import matplotlib.pyplot as plt
import figs.utilities.transform_helper as th

from scipy.spatial.transform import Rotation as R
from typing import Union,Tuple,List

def plot_tXU_3D(tXUs:Union[np.ndarray,List[np.ndarray]],
                     n_fr:int=None) -> None:
    """
    Plot the spatial trajectory.

    Args:
        - tXUs: List of tXU arrays or a single tXU array.
        - n_fr: Number of steps between plots of the quadcopter frame.

    """

    # Some useful constants
    traj_colors:List[str]=["red","green","blue","orange","purple","brown","pink","gray","olive","cyan"]

    # Capture case where only one tXU is passed
    if isinstance(tXUs, np.ndarray):
        tXUs = [tXUs]
    
    # Set the plot limits
    tXU_lim = get_plot_limits(tXUs)
    
    # Initialize World Frame Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(tXU_lim[1,:])
    ax.set_ylim(tXU_lim[2,:])
    ax.set_zlim(tXU_lim[3,:])
    ax.set_box_aspect(tXU_lim[1:4,1]-tXU_lim[1:4,0])  # aspect ratio is 1:1:1 in data space

    ax.invert_zaxis()
    ax.invert_yaxis()

    # Rollout the world frame trajectory
    for idx,tXU in enumerate(tXUs):
        # Plot the world frame trajectory
        ax.plot(tXU[1,:], tXU[2,:], tXU[3,:],color=traj_colors[idx%len(traj_colors)],alpha=0.5)             # spline

        # Plot Initial and Final
        quad_frame(tXU[1:14,0],ax)
        quad_frame(tXU[1:14,-1],ax)
            
        if n_fr is not None:
            for i in range(n_fr,tXU.shape[1],n_fr):
                quad_frame(tXU[1:14,i],ax)

    plt.show(block=False)

def plot_tXU_time(tXUs:Union[np.ndarray,List[np.ndarray]]) -> None:
    """
    Plot the time trajectory.

    Args:
        - tXUs: List of tXU arrays or a single tXU array.
    """

    # Some useful constants
    ylabels_pv = [["$p_x$","$p_y$","$p_z$"],["$v_x$","$v_y$","$v_z$"]]
    ylabels_qw = [["$q_x$","$q_y$","$q_z$","$q_w$"],["f_n",r"$\omega_x$",r"$\omega_y$",r"$\omega_z$"]]

    # Capture case where only one tXU is passed
    if isinstance(tXUs, np.ndarray):
        tXUs = [tXUs]

    # Set the plot limits
    tXU_lim = get_plot_limits(tXUs)

    # Plot Positions and Velocities
    fig, axs = plt.subplots(3, 2, figsize=(10, 4))

    for i in range(6):
        col,row = divmod(i,3)
        for tXU in tXUs:
            axs[row,col].plot(tXU[0,:],tXU[i+1,:],alpha=0.5)
        
        axs[row,col].set_xlim([tXU_lim[0,0],tXU_lim[0,1]])
        axs[row,col].set_ylim([tXU_lim[i+1,0],tXU_lim[i+1,1]])
        axs[row,col].set_ylabel(ylabels_pv[col][row])

    axs[0, 0].set_title('Position')
    axs[0, 1].set_title('Velocity')
    
    plt.tight_layout()
    plt.show(block=False)

    # Plot Orientation and Body Rates
    fig, axs = plt.subplots(4, 2, figsize=(10, 4))

    for i in range(8):
        col,row = divmod(i,4)
        for tXU in tXUs:
            axs[row,col].plot(tXU[0,:],tXU[i+7,:],alpha=0.5)

        axs[row,col].set_xlim([tXU_lim[0,0],tXU_lim[0,1]])
        axs[row,col].set_ylim([tXU_lim[i+7,0],tXU_lim[i+7,1]])
        axs[row,col].set_ylabel(ylabels_qw[col][row])

        if i == 4:
            axs[row,col].invert_yaxis()

    axs[0, 0].set_title('Orientation')
    axs[0, 1].set_title('Body Rates')
    
    plt.tight_layout()
    plt.show(block=False)

def get_plot_limits(tXUs:List[np.ndarray],use_aesthetics:bool=True,
                    pz_aesth:np.ndarray=np.array([0.0,-2.0]),
                    pxy_min_aesth:float=2.0,
                    pq_aesth:np.ndarray=np.array([-1.0,1.0]),
                    dx_aesth:float=0.2) -> np.ndarray:
    """
    Get the plot limits for the trajectory. An aesthetic option is available
    that sets px and py to a minimum of -2m <-> 2m, locks pz to 0m <-> -2m, and sets
    a margin on all states for better visualization.

    Args:
        - tXUs: List of tXU arrays or a single tXU array.
        - use_aesthetics: Use aesthetic limits.
        - pz_aesth: Aesthetic limits for pz.
        - pxy_min_aesth: Aesthetic limits for px and py.
        - pq_aesth: Aesthetic limits for q.
        - dx_aesth: Aesthetic margin for all states.

    Returns:
        - tXU_lim: Plot limits.

    """

    # Initialize the plot limits
    Ndim = tXUs[0].shape[0]
    tXU_max = [[] for _ in range(Ndim)]
    tXU_min = [[] for _ in range(Ndim)] 

    # Determine the plot limits
    for tXU in tXUs:
        for i in range(Ndim):
            tXU_min[i] = np.append(tXU_min[i],np.min(tXU[i,:]))
            tXU_max[i] = np.append(tXU_max[i],np.max(tXU[i,:]))
        
    tXU_lim = np.zeros((Ndim,2))
    for i in range(Ndim):
        tXU_lim[i,0] = np.floor(np.min(tXU_min[i]))
        tXU_lim[i,1] = np.ceil(np.max(tXU_max[i]))

    if use_aesthetics:
        # Limit the plot to a minimum of -2m <-> 2m for px and py
        for i in range(2):
            tXU_lim[i+1,0] = np.min([tXU_lim[i+1,0],-pxy_min_aesth])
            tXU_lim[i+1,1] = np.max([tXU_lim[i+1,1],pxy_min_aesth])
        
        # Lock pz to 0m <-> -2m
        tXU_lim[3,:] = pz_aesth

        # Lock q to -1 <-> 1
        tXU_lim[7:11,0] = pq_aesth[0]
        tXU_lim[7:11,1] = pq_aesth[1]

        # Set a margin on all states for better visualization
        tXU_lim[1:,0] -= dx_aesth
        tXU_lim[1:,1] += dx_aesth

    return tXU_lim

def quad_frame(x:np.ndarray,ax:plt.Axes,scale:float=1.0,
               dims:np.ndarray=np.diag([0.6,0.6,-0.2])) -> None:
    """
    Plot a quadcopter frame in 3D.

    Args:
        - x:     State vector.
        - ax:    Axes object.
        - scale: Scale factor.
        - dims:  Quadcopter dimensions.


    """

    frame_body = scale*dims
    frame_labels = ["red","green","blue"]
    pos  = x[0:3]
    quat = x[6:10]
    
    for j in range(0,3):
        Rj = R.from_quat(quat).as_matrix()
        arm = Rj@frame_body[j,:]

        frame = np.zeros((3,2))
        if (j == 2):
            frame[:,0] = pos
        else:
            frame[:,0] = pos - arm

        frame[:,1] = pos + arm

        ax.plot(frame[0,:],frame[1,:],frame[2,:], frame_labels[j],label='_nolegend_')
