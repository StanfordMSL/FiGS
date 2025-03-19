import os
import shutil
import json
import torch
import numpy as np
import figs.utilities.trajectory_helper as th

from pathlib import Path
from typing import Type,Union,Tuple,Dict,List
from acados_template import AcadosSimSolver, AcadosSim
from figs.control.base_controller import BaseController
from figs.dynamics.fext_model import export_fext_ode_model
from figs.dynamics.model_specifications import generate_specifications
from figs.render.gsplat import GSplat

class Simulator:
    """
    Class to simulation in FiGS
    """

    def __init__(self,
                 scene_name:str,rollout_name:str='baseline',
                 frame_name:Union[None,str]=None,
                 configs_path:Path=None,gsplats_path:Path=None) -> None:
        """
        The FiGS simulator simulates flying in a Gaussian Splat by using an ACADOS integrator
        (solver) to rollout a trajectory in a Gaussian Splat (gsplat) according to a control
        policy (policy) and simulation configuration (conFiG).

        For efficiency, the gsplat and conFiG are tied to individual Simulator objects. The
        solver and policy can be swapped out during runtime. This allows us to abstract away
        the JSON-based configuration and C backend of ACADOS. Note that every time the solver
        gets updated, the conFiG must also be uploaded with new drone specifications.

        Args:
            - scene_name:       Name of the scene to load.
            - rollout_name:     Rollout config to load.
            - frame_name:       Name of the frame to load (None if not instantiating with a frame).
            - configs_path:     Path to the directory containing the JSON files.
            - gsplats_path:     Path to the directory containing the gsplats.

        Attributes:
            - gsplat:           Gaussian Splat of the scene.
            - conFiG:           Dictionary holding simulation configurations (frequency, noise, delay and drone specs).
            - solver:           An ACADOS integrator for the drone dynamics.
            - policy:           Policy to control the drone (an ACADOS OCP based MPC by default).
            - configs_path:     Path to the configuration directory.
            - workspace_path:   Path to the gsplat directory.
        """

        # Set the configuration directory
        if configs_path is None:
            self.configs_path = Path(__file__).parent.parent.parent.parent/'configs'
        else:
            self.configs_path = configs_path

        # Set the gsplat directory
        if gsplats_path is None:
            self.workspace_path = Path(__file__).parent.parent.parent.parent/'gsplats'/'workspace'
        else:
            self.workspace_path = gsplats_path/'workspace'

        # Instantiate empty attributes
        self.gsplat = None
        self.conFiG = {"rollout":{},"drone":{}}
        self.solver = None

        # Load the attributes
        self.load_scene(scene_name)
        self.load_rollout(rollout_name)

        if frame_name is not None:
            self.load_frame(frame_name)

    def load_scene(self, scene_name:str):
        """
        Loads/Updates the gsplat attribute given a scene name.

        Args:
            - scene_name:     Name of the scene to load.
        """

        # Get current and workspace directories
        curr_path,work_path = Path(os.getcwd()),self.workspace_path

        # Find the GSplat configuration
        search_path = work_path/'outputs'/scene_name
        yaml_configs = list(search_path.rglob("*.yml"))
    
        if len(yaml_configs) == 0:
            raise ValueError(f"The search path '{search_path}' did not return any configurations.")
        elif len(yaml_configs) > 1:
            raise ValueError(f"The search path '{search_path}' returned multiple configurations. Please specify a unique configuration within the directory.")
        else:
            gsplat_config = {"name":scene_name,"path":yaml_configs[0]}

        # Load GSplat (from the workspace directory to avoid path issues)
        os.chdir(work_path)
        gsplat = GSplat(gsplat_config)
        os.chdir(curr_path)

        # Update attribute(s)
        self.gsplat = gsplat

    def load_rollout(self, rollout:Union[str,dict]):
        """
        Loads/Updates the conFiG attribute given a rollout name.

        Args:
            - rollout:   Type of rollout to load.
        """

        # Check if rollout is a string or dictionary
        if isinstance(rollout,str):
            # Load the rollout config
            json_config = self.configs_path/"rollout"/(rollout+".json")

            if not json_config.exists():
                raise ValueError(f"The json file '{json_config}' does not exist.")
            else:
                # Load the json configuration
                with open(json_config) as file:
                    rollout_config = json.load(file)
        else:
            rollout_config = rollout

        # Update attribute(s)
        self.conFiG["rollout"] = rollout_config
        
    def load_frame(self, frame:Union[str,dict]):
        """
        Loads the solver attribute.

        Args:
            - frame_name:     Name of the frame to load.
        """
        
        # Check if rollout is a string or dictionary
        if isinstance(frame,str):
            # Load the frame config
            json_config = self.configs_path/"frame"/(frame+".json")

            if not json_config.exists():
                raise ValueError(f"The json file '{json_config}' does not exist.")
            else:
                # Load the json configuration
                with open(json_config) as file:
                    frame_config = json.load(file)
        else:
            frame_config = frame

        # Clear previous solver
        del self.solver
        
        # Some useful intermediate variables
        drn_spec = generate_specifications(frame_config)
        sim_json = 'figs_sim_solver.json'

        # Generate the simulator
        sim = AcadosSim()
        sim.model = export_fext_ode_model(drn_spec["m"],drn_spec["tn"])  
        sim.solver_options.T = 1/self.conFiG["rollout"]["frequency"]
        sim.solver_options.integrator_type = 'IRK'

        solver = AcadosSimSolver(sim, json_file=sim_json, verbose=False)

        # Clean up the ACADOS generation files
        os.remove(os.path.join(os.getcwd(),sim_json))
        shutil.rmtree(sim.code_export_directory)
        
        # Update attribute(s)
        self.solver = solver
        self.conFiG["drone"] = drn_spec
    
    def simulate(self,policy:BaseController,
                 t0:float,tf:int,x0:np.ndarray,obj:Union[None,np.ndarray]=None
                 ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """
        Simulates the flight.

        Args:
            - t0:   Initial time.
            - tf:   Final time.
            - x0:   Initial state.
            - obj:  Objective to use for the simulation.
        """

        # Check if frame is loaded
        if self.solver is None:
            raise ValueError("Frame has not been loaded. Please load a frame before simulating.")

        # Drone Variables
        nx,nu = self.conFiG["drone"]["nx"],self.conFiG["drone"]["nu"]
        cam_cfg = self.conFiG["drone"]["camera"]
        height,width,channels = cam_cfg["height"],cam_cfg["width"],cam_cfg["channels"]
        T_c2b = self.conFiG["drone"]["T_c2b"]

        # Base Rollout Variables
        hz_sim = self.conFiG["rollout"]["frequency"]
        t_dly = self.conFiG["rollout"]["delay"]

        # External Forces Rollout Variables
        Fext_con:dict = self.conFiG["rollout"]["external_forces"]

        Fext = []
        if Fext_con is not None:
            for fext_con in Fext_con.values():
                Fext.append({
                    "lower": np.array(fext_con["lower"]),
                    "upper": np.array(fext_con["upper"]),
                    "mean": np.array(fext_con["mean"]),
                    "std": np.array(fext_con["std"])
                })
        
        # Noise Rollout Variables
        model_noise = self.conFiG["rollout"]["noise"]["model"]
        sensor_noise = self.conFiG["rollout"]["noise"]["sensor"]

        if model_noise is None:
            mu_md_s,std_md_s = np.zeros(nx),np.zeros(nx)
        else:
            mu_md_s = np.array(model_noise["mean"])
            std_md_s = np.array(model_noise["std"])

        if sensor_noise is None:
            mu_sn,std_sn = np.zeros(nx),np.zeros(nx)
        else:
            mu_sn = np.array(sensor_noise["mean"])
            std_sn = np.array(sensor_noise["std"])

        # Fusion Rollout Variables
        fuse_con = self.conFiG["rollout"]["fusion"]

        use_fusion = False if fuse_con is None else True
        Wf_md = np.diag(fuse_con) if fuse_con else np.eye(nx)
        Wf_sn = np.eye(nx)-Wf_md
        
        # Derived Variables
        n_sim2ctl = int(hz_sim/policy.hz)       # Number of simulation steps per control step
        mu_md = mu_md_s*(1/n_sim2ctl)           # Scale model mean noise to control rate
        std_md = std_md_s*(1/n_sim2ctl)         # Scale model std noise to control rate
        dt = np.round(tf-t0)                    # Total time
        Nsim = int(dt*hz_sim)                   # Number of simulation steps
        Nctl = int(dt*policy.hz)                # Number of control steps
        n_delay = int(t_dly*hz_sim)             # Number of steps for input delay

        # Diagnostics Variables
        Tsol = np.zeros((4,Nctl))
        
        # Transient Variables
        xcr,xpr,xsn = x0.copy(),x0.copy(),x0.copy()
        ucm = np.array([-self.conFiG["drone"]['m']/self.conFiG["drone"]['tn'],0.0,0.0,0.0])
        udl = np.hstack((ucm.reshape(-1,1),ucm.reshape(-1,1)))
        zcr = {key: torch.zeros(policy.Nznn[key]) for key in policy.Nznn.keys()}

        # Instantiate camera object
        camera = self.gsplat.generate_output_camera(cam_cfg)

        # Trajectory Rollout Variables
        Tro,Xro,Uro = np.zeros(Nctl+1),np.zeros((nx,Nctl+1)),np.zeros((nu,Nctl))
        Iro = np.zeros((Nctl,height,width,channels),dtype=np.uint8)
        Xro[:,0] = x0
        Fro = np.zeros((3,Nctl))

        # Rollout
        for i in range(Nsim):
            # Get current time and state
            tcr = t0+i/hz_sim

            # Control
            if i % n_sim2ctl == 0:
                # Get current image
                Tb2w = th.xv_to_T(xcr)
                T_c2w = Tb2w@T_c2b
                icr = self.gsplat.render_rgb(camera,T_c2w)

                # Add sensor noise and syncronize estimated state
                if use_fusion:
                    xsn += np.random.normal(loc=mu_sn,scale=std_sn)
                    xsn = Wf_sn@xsn + Wf_md@xcr
                else:
                    xsn = xcr + np.random.normal(loc=mu_sn,scale=std_sn)
                xsn[6:10] = th.obedient_quaternion(xsn[6:10],xpr[6:10])

                # Generate controller command
                ucm,zcr,tsol = policy.control(tcr,xsn,ucm,obj,icr,zcr)

                # Update delay buffer
                udl[:,0] = udl[:,1]
                udl[:,1] = ucm

            # Extract delayed command
            uin = udl[:,0] if i%n_sim2ctl < n_delay else udl[:,1]

            # Add external forces
            ufe = np.zeros(3)
            for fext in Fext:
                if np.all((xcr[0:3] <= fext["upper"]) & (xcr[0:3] >= fext["lower"])):
                    force = np.random.normal(loc=fext["mean"],scale=fext["std"])
                    ufe += force

            ucr = np.hstack((uin,ufe))

            # Simulate both estimated and actual states
            xcr = self.solver.simulate(x=xcr,u=ucr)
            if use_fusion:
                xsn = self.solver.simulate(x=xsn,u=ucr)

            # Add model noise
            xcr = xcr + np.random.normal(loc=mu_md,scale=std_md)
            xcr[6:10] = th.obedient_quaternion(xcr[6:10],xpr[6:10])

            # Update previous state
            xpr = xcr
            
            # Store values
            if i % n_sim2ctl == 0:
                k = i//n_sim2ctl

                Iro[k,:,:,:] = icr
                Tro[k] = tcr
                Xro[:,k+1] = xcr
                Uro[:,k] = ucm
                Fro[:,k] = ufe
                Tsol[:,k] = tsol

        # Log final time
        Tro[Nctl] = t0+Nsim/hz_sim

        return Tro,Xro,Uro,Iro,Fro,Tsol