import os
import shutil
import torch
import numpy as np
import figs.utilities.trajectory_helper as th
import figs.dynamics.quadcopter_model as qm
import figs.dynamics.quadcopter_specifications as qs

from acados_template import AcadosSimSolver, AcadosSim
from figs.dynamics.external_forces import ExternalForces
from figs.control.base_controller import BaseController
from figs.render.gsplat import GSplat

class Simulator:
    """
    Class to simulation in FiGS
    """

    def __init__(self,gsplat:GSplat,rollout:dict,frame:None|dict=None,forces:None|dict=None) -> None:
        """
        The FiGS simulator simulates flying in a Gaussian Splat by using an ACADOS integrator
        (solver) to rollout a trajectory in a Gaussian Splat (gsplat) in the presence of a set
        of external forces (forces) and according to a control policy (policy) and simulation
        configuration (conFiG).

        Args:
            - gsplat:      GSplat.
            - rollout:     Rollout config.
            - frame:       Frame config (None if not instantiating with a frame).
            - forces:      Forces config (None if no external forces).

        Attributes:
            - gsplat:           Gaussian Splat of the scene.
            - conFiG:           Dictionary holding simulation configurations.
            - solver:           An ACADOS integrator for the drone dynamics.
        """

        # Instantiate the dynamics solver
        sim_json = 'figs_sim_solver.json'

        sim = AcadosSim()
        sim.model = qm.export_model()
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

    def update_frame(self, frame_config:dict):
        """
        Loads/Updates the conFiG attribute given a rollout name.

        Args:
            - frame_config: Configuration dictionary.
        """

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
                 t0:float,tf:int,x0:np.ndarray,obj:None|np.ndarray=None
                 ) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """
        Simulates the flight.

        Args:
            - t0:   Initial time.
            - tf:   Final time.
            - x0:   Initial state.
            - obj:  Objective to use for the simulation.
        """

        # Load configs
        Rout = self.conFiG["rollout"]
        Spec = qs.generate_specifications(self.conFiG["frame"])
        Fext = ExternalForces(self.conFiG["forces"])

        # Drone Variables
        nx,nu = Spec["nx"],Spec["nu"]
        m,kt = Spec["m"],Spec["kt"]
        Tc2b = Spec["Tc2b"]
        height,width = Spec["camera"]["height"],Spec["camera"]["width"]
        channels = Spec["camera"]["channels"]
        camera = self.gsplat.generate_output_camera(Spec["camera"])

        # Base Rollout Variables
        hz_sim = Rout["frequency"]
        t_dly = Rout["delay"]
        
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

        # Fusion Rollout Variables
        fuse_con = Rout["fusion"]

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
        ucm = np.array([-m/kt,0.0,0.0,0.0])
        udl = np.hstack((ucm.reshape(-1,1),ucm.reshape(-1,1)))
        zcr = {key: torch.zeros(policy.Nznn[key]) for key in policy.Nznn.keys()}

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
                Tc2w = Tb2w@Tc2b
                icr = self.gsplat.render_rgb(camera,Tc2w)

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
            ucr = udl[:,0] if i%n_sim2ctl < n_delay else udl[:,1]

            # Add external forces
            ufe = Fext.get_forces(xcr, noisy=True)
            pcr = np.hstack((m,kt,ufe))

            # Simulate both estimated and actual states
            xcr = self.solver.simulate(x=xcr,u=ucr,p=pcr)
            if use_fusion:
                xsn = self.solver.simulate(x=xsn,u=ucr,p=pcr)

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