import os
import shutil
import numpy as np
import figs.utilities.config_helper as ch
import figs.utilities.transform_helper as th
import figs.utilities.orientation_helper as oh
import figs.dynamics.quadcopter_rate_model as qrm
import figs.dynamics.quadcopter_specifications as qs

from acados_template import AcadosSimSolver, AcadosSim
from figs.dynamics.external_forces import ExternalForces
from figs.control.base_controller import BaseController
from figs.render.gsplat import GSplat

class Simulator:
    """
    Class to simulation in FiGS
    """

    def __init__(self,gsplat:str|GSplat,method:str|dict,frame:None|str|dict=None,forces:None|dict=None) -> None:
        """
        The FiGS simulator simulates flying in a Gaussian Splat by using an ACADOS integrator
        (solver) to rollout a trajectory in a Gaussian Splat (gsplat) in the presence of a set
        of external forces (forces) and according to a control policy (policy) and simulation
        configuration (conFiG).

        Args:
            - gsplat:   GSplat.
            - method:   Method config.
            - frame:    Frame config (None if not instantiating with a frame).
            - forces:   Forces config (None if no external forces).

        Attributes:
            - gsplat:   Gaussian Splat of the scene.
            - conFiG:   Dictionary holding simulation configurations.
            - solver:   An ACADOS integrator for the drone dynamics.
        """

        # Check if gsplat is a string or GSplat object
        if isinstance(gsplat, str):
            gsplat = ch.get_gsplat(gsplat)
        
        # Check if rollout is a string or dictionary
        if isinstance(method, str):
            method = ch.get_config(method, "methods")
        rollout = method["rollout"]

        # Check if frame is a string or dictionary
        if isinstance(frame, str):
            frame = ch.get_config(frame, "frames")

        # Instantiate the dynamics solver
        sim_json = 'figs_sim_solver.json'

        sim = AcadosSim()
        sim.model = qrm.export_model()
        sim.parameter_values = np.zeros(sim.model.p.shape)
        sim.solver_options.T = 1/rollout["frequency"]
        sim.solver_options.integrator_type = 'IRK'

        # Instantiate attributes
        self.gsplat = gsplat
        self.conFiG = {
            "rollout": rollout,
            "frame": frame,
            "forces": forces,
            }
        self.solver = AcadosSimSolver(sim, json_file=sim_json, verbose=False)

        # Clean up the ACADOS generation files
        os.remove(os.path.join(os.getcwd(),sim_json))
        shutil.rmtree(sim.code_export_directory)

    def update_frame(self, frame_config:str|dict):
        """
        Loads/Updates the conFiG attribute given a rollout name.

        Args:
            - frame_config: Configuration dictionary.
        """
        # Check if frame_config is a string or dictionary
        if isinstance(frame_config, str):
            frame_config = ch.get_config(frame_config, "frames")

        # Update attribute(s)
        self.conFiG["frame"] = frame_config

    def update_forces(self, forces_config:dict):
        """
        Loads/Updates the conFiG attribute given a rollout name.

        Args:
            - forces_config: Configuration dictionary.
        """

        # Update attribute(s)
        self.conFiG["forces"] = forces_config

    def simulate(self,policy:BaseController,
                 t0:float,tf:int,x0:np.ndarray
                 ) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,list[dict]]:
        """
        Simulates the flight.

        Args:
            - t0:   Initial time.
            - tf:   Final time.
            - x0:   Initial state.
            - obj:  Objective to use for the simulation.

        Returns:
            - Tro:  Time vector.
            - Xro:  State vector.
            - Uro:  Control input vector.
            - Fro:  Resultant force vector.
            - Rgb:  RGB image vector.
            - Dpt:  Depth image vector.
            - Aux:  Auxiliary data vector.
        """
        # Wrench dimensions
        nw = 6                  # Force + Torque

        # Load configs
        Rout = self.conFiG["rollout"]
        Spec = qs.generate_specifications(self.conFiG["frame"])
        fex = ExternalForces(self.conFiG["forces"])

        # Drone Variables
        nx,nu = Spec["nx"],Spec["nu"]
        m,kt = Spec["m"],Spec["kt"]
        g,Nrtr = Spec["g"],Spec["Nrtr"]
        Tc2b = Spec["Tc2b"]
        rgb_dim,dpt_dim = Spec["rgb_dim"],Spec["dpt_dim"]
        camera = self.gsplat.generate_output_camera(Spec["camera"])

        # Base Rollout Variables
        hz_sim = Rout["frequency"]
        
        # Noise Rollout Variables
        model_noise = Rout["noise"]["model"]
        sensor_noise = Rout["noise"]["sensor"]

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

        # Derived Variables
        n_sim2ctl = int(hz_sim/policy.hz)       # Number of simulation steps per control step
        mu_md = mu_md_s*(1/n_sim2ctl)           # Scale model mean noise to control rate
        std_md = std_md_s*(1/n_sim2ctl)         # Scale model std noise to control rate
        dt = np.round(tf-t0,5)                  # Total time
        Nsim = int(dt*hz_sim)                   # Number of simulation steps
        Nctl = int(dt*policy.hz)                # Number of control steps

        # Trajectory Rollout Variables
        Tro,Xro,Uro = np.zeros((Nctl+1)),np.zeros((Nctl+1,nx)),np.zeros((Nctl,nu))
        Wro = np.zeros((Nctl,nw))
        Rgb = np.zeros(((Nctl,) + rgb_dim),dtype=np.uint8)
        Dpt = np.zeros(((Nctl,) + dpt_dim),dtype=np.uint8)
        Tsol = np.zeros((Nctl,))                # Time to solution for each control step

        # Transient Variables
        xcr,xpr,xsn = x0.copy(),x0.copy(),x0.copy()
        ucr = np.array([-(m*g)/(Nrtr*kt),0.0,0.0,0.0])

        # Rollout Loop
        tau_cr = np.zeros(3)                            # Current torque (unmodeled due to body rate dynamics)
        for i in range(Nsim):
            # Get current variables
            tcr = t0+i/hz_sim                           # Current time
            fcr = fex.get_forces(xcr[0:6], noisy=True)  # Current forces
            pcr = np.hstack((m,kt,fcr))                 # Parameters for the dynamics solver            
            fts = np.hstack((fcr,tau_cr))               # Sensed force/torque vector

            # Control Loop
            if i % n_sim2ctl == 0:
                # Get current images
                Tb2w = th.x_to_T(xcr)
                Tc2w = Tb2w@Tc2b
                rgb,dpt = self.gsplat.render_rgb(camera,Tc2w)

                # Add sensor noise and syncronize estimated state
                xsn = xcr + np.random.normal(loc=mu_sn,scale=std_sn)
                xsn[6:10] = oh.obedient_quaternion(xsn[6:10],xpr[6:10])

                # Generate controller command
                ucr,tsol = policy.control(tcr,xsn,ucr,rgb,dpt,fts)

                # Log data
                k = i//n_sim2ctl
                Tro[k],Xro[k,:],Uro[k,:] = tcr,xcr,ucr
                Wro[k,0:3] = fcr
                Rgb[k,:,:,:],Dpt[k,:,:,:] = rgb,dpt
                Tsol[k] = sum(tsol.values())

            # Update previous state
            xpr = xcr

            # Simulate both estimated and actual states
            xcr = self.solver.simulate(x=xcr,u=ucr,p=pcr)

            # Add model noise
            xcr = xcr + np.random.normal(loc=mu_md,scale=std_md)
            xcr[6:10] = oh.obedient_quaternion(xcr[6:10],xpr[6:10])

        # Log entry
        Tro[Nctl] = t0+Nsim/hz_sim
        Xro[Nctl,:] = xcr

        return Tro,Xro,Uro,Wro,Rgb,Dpt,Tsol