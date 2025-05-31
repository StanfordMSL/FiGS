import time
import shutil
import os
import numpy as np
import scipy.linalg
import figs.utilities.config_helper as ch
import figs.utilities.transform_helper as th
import figs.dynamics.quadcopter_model as qm

from pathlib import Path
from casadi import vertcat
from acados_template import AcadosOcp, AcadosOcpSolver
from figs.control.base_controller import BaseController
from figs.dynamics.external_forces import ExternalForces
from figs.tsplines.min_time_snap import MinTimeSnap

class VehicleRateMPC(BaseController):
    def __init__(self,
                 policy:str|dict,course:str|dict,frame:str|dict=None,
                 use_RTI:bool=False,
                 name:str="vrmpc") -> None:
        
        """
        Constructor for the VehicleRateMPC class.
        
        Args:
            - policy:       Config Dict of the policy.
            - course:       Config Dict of the course.
            - frame:        Config Dict of the (drone) frame.
            - use_RTI:      Use RTI flag.
            - name:         Name of the controller.

        Variables:
            -  name:         Name of the controller.
            -  hz:           Frequency of the controller.
            -  Nx:          Number of states.
            -  Nu:          Number of inputs.
            -  Tsd:         Desired trajectory.
            -  FOd:         Desired forces.
            -  tXUd:        Desired trajectory in the state space.
            -  fex:         External forces.
            -  p:           Parameters of the model.
            -  Qk:          State cost matrix.
            -  Rk:          Input cost matrix.
            -  QN:          Terminal cost matrix.
            -  lbu:         Lower bounds on the inputs.
            -  ubu:         Upper bounds on the inputs.
            -  Ws:          State cost weights.
            -  use_RTI:     Use RTI flag.
            -  solver:      Acados solver.

        """
        # =====================================================================
        # Check configs
        # =====================================================================

        # Check if policy is a string or dictionary
        if isinstance(policy, str):
            policy = ch.get_config(policy, "pilots")

        # Check if course is a string or dictionary
        if isinstance(course, str):
            course = ch.get_config(course, "courses")

        # Check if frame is a string or dictionary
        if isinstance(frame, str):
            frame = ch.get_config(frame, "frames")

        # =====================================================================
        # Extract parameters
        # =====================================================================
        
        # Initialize the BaseController
        super().__init__()

        # Policy Parameters
        plan,track = policy["plan"],policy["track"]

        kT,use_l2_time = plan["kT"],plan["use_l2_time"]
        hz,Nhn = track["hz"],track["horizon"]
        Qk,Rk,QN = np.diag(track["Qk"]),np.diag(track["Rk"]),np.diag(track["QN"])
        Ws = np.diag(track["Ws"])
        lbu,ubu = np.array(track["bounds"]["lower"]),np.array(track["bounds"]["upper"])

        # Course Parameters
        WPs_cfg,Fs_cfg = course["waypoints"],course["forces"]
        
        # Frame Parameters
        if frame is None:
            m,kt = 1.0,7.0
        else:
            m,kt = frame["mass"],frame["motor_thrust_coeff"]
        p = np.hstack((m,kt,np.zeros(3)))

        # Some useful constants
        nx,nu = Qk.shape[0], Rk.shape[0]
        ny,ny_e = nx+nu,nx
        
        # Get initial solution
        mts = MinTimeSnap(WPs_cfg,hz,kT,use_l2_time)
        fex = ExternalForces(Fs_cfg)

        Tsd,FOd = mts.get_desired_trajectory()
        tXUd = th.TsFO_to_tXU(Tsd,FOd,m,kt,fex)

        # =====================================================================
        # Setup Acados Variables
        # =====================================================================

        # Initialize Acados OCP
        ocp = AcadosOcp()

        ocp.model = qm.export_model()        
        ocp.parameter_values = np.zeros(ocp.model.p.shape)
        ocp.model.cost_y_expr = vertcat(ocp.model.x, ocp.model.u)
        ocp.model.cost_y_expr_e = ocp.model.x

        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'

        ocp.cost.W = scipy.linalg.block_diag(Qk,Rk)
        ocp.cost.W_e = QN
        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((ny_e, ))

        ocp.constraints.x0 = tXUd[0,1:11]
        ocp.constraints.lbu = lbu
        ocp.constraints.ubu = ubu
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        # Initialize Acados Solver
        ocp.solver_options.N_horizon = Nhn
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'EXACT'
        ocp.solver_options.integrator_type = 'IRK'
        ocp.solver_options.sim_method_newton_iter = 10

        if use_RTI:
            ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        else:
            ocp.solver_options.nlp_solver_type = 'SQP'

        ocp.solver_options.qp_solver_cond_N = Nhn
        ocp.solver_options.tf = Nhn/hz
        ocp.solver_options.qp_solver_warm_start = 1

        solver = AcadosOcpSolver(ocp,verbose=False)
        
        # Clear the generated code
        os.remove(os.path.join(os.getcwd(),"acados_ocp.json"))
        shutil.rmtree(ocp.code_export_directory)

        # =====================================================================
        # Controller Variables
        # =====================================================================

        # ---------------------------------------------------------------------
        # Necessary Variables for Base Controller -----------------------------
        self.name = name
        self.hz = hz

        # ---------------------------------------------------------------------
        # Controller Specific Variables
        self.Nx,self.Nu = nx,nu
        self.Tsd,self.FOd = Tsd,FOd
        self.tXUd,self.fex = tXUd,fex
        self.p = p
        self.Qk,self.Rk,self.QN = Qk,Rk,QN
        self.lbu,self.ubu = lbu,ubu
        self.Ws = Ws
        self.use_RTI = use_RTI
        self.solver = solver

        # =====================================================================
        # Warm start the solver
        # =====================================================================

        for _ in range(5):
            self.control(0.0,tXUd[0,1:11],None,None,None,None)

    def update_frame(self,frame:str|dict) -> None:
        """
        Method to update the frame related variables of the controller.
        
        Args:
            - frame: Config Dict of the (drone) frame.

        """

        # Check if frame is a string or dictionary
        if isinstance(frame, str):
            frame = ch.get_config(frame, "frames")
        
        # Update the frame
        m,kt = frame["mass"],frame["motor_thrust_coeff"]
        tXUd = th.TsFO_to_tXU(self.Tsd,self.FOd,m,kt,self.fex)

        self.p[0],self.p[1] = m,kt
        self.tXUd = tXUd

    def control(self,tcr:float,xcr:np.ndarray,upr:np.ndarray=None,
                rgb:np.ndarray=None,dpt:np.ndarray=None,
                fcr:np.ndarray=np.array([0.0,0.0,0.0,0.0,0.0,0.0])
    ) -> tuple[np.ndarray, dict[str,float]]:
        """
        Method to compute the control input for the VehicleRateMPC controller. We use the standard input arguments
        format with the unused arguments set to None. Likewise, we use the standard output format with the unused
        outputs set to None.

        Args:
            - tcr: Current time.
            - xcr: Current state.
            - upr: Previous control input (unused).
            - rgb: RGB image (unused).
            - dpt: Depth image (unused).
            - fcr: Current force.

        Returns:
            - ucr:  Control input.
            - tsol: Solve times dictionary with keys "setup_ocp" and "solve_ocp".
        """

        # Unpack inputs
        _ = upr,rgb,dpt,fcr

        # Start timer
        t0 = time.time()

        # Get desired trajectory
        ydes = self.get_ydes(tcr,xcr)
        
        # Get external forces
        self.p[2:5] = self.fex.get_forces(xcr[0:6])

        # Set desired trajectory
        for i in range(self.solver.acados_ocp.dims.N):
            self.solver.cost_set(i, "yref", ydes[i,:])
            self.solver.set(i,'x',ydes[i,0:10])
            self.solver.set(i,'u',ydes[i,10:])
            self.solver.set(i,'p',self.p)

        self.solver.cost_set(self.solver.acados_ocp.dims.N, "yref", ydes[-1,0:10])
        self.solver.set(self.solver.acados_ocp.dims.N,'x',ydes[-1,0:10])
        
        # Solve OCP
        t1 = time.time()
        if self.use_RTI:
            # preparation phase
            self.solver.options_set('rti_phase', 1)
            status = self.solver.solve()

            # set initial state
            self.solver.set(0, "lbx", xcr)
            self.solver.set(0, "ubx", xcr)

            # feedback phase
            self.solver.options_set('rti_phase', 2)
            status = self.solver.solve()

            ucr = self.solver.get(0, "u")
        else:
            # Solve ocp and get next control input
            try:
                ucr = self.solver.solve_for_x0(x0_bar=xcr)
            except:
                print("Warning: VehicleRateMPC failed to solve OCP. Using previous input.")
                ucr = self.solver.get(0, "u")
        t2 = time.time()

        # Compute solve times
        tsol = {"setup_ocp":t1-t0,
                "solve_ocp":t2-t1
                }

        return ucr,tsol

    def get_ydes(self,tcr:float,xcr:np.ndarray) -> np.ndarray:
        """
        Method to get the section of the desired trajectory at the current time.

        Args:
            - tcr: Time at the current control step.
            - xcr: States at the current control step.

        Returns:
            - ydes:   Desired trajectory section at the current time.

        """

        # Unpack some stuff
        hz,ns = self.hz,int(self.hz)
        tXUd = self.tXUd
        Ndes = tXUd.shape[0]
        Ws = self.Ws

        # Use time to get estimated index and section out the search space
        idx_est = int(hz*tcr)
        ks0,ksf = idx_est-ns,idx_est+ns
        ks0,ksf = np.clip(ks0,0,Ndes-1),np.clip(ksf,0,Ndes)
        
        Xi = tXUd[ks0:ksf,1:11]
        
        # Find the closest point in the search space (weighted)
        dXi = Xi-xcr
        J_dXi = np.array([x.T@Ws@x for x in dXi])
        idx0 = ks0 + np.argmin(J_dXi)
        idxf = idx0 + self.solver.acados_ocp.dims.N+1

        # Pad if idxf is greater than the last index
        if idxf < Ndes:
            ydes = tXUd[idx0:idxf,1:]
        else:
            ydes = tXUd[idx0:,1:]
            ypad = np.tile(ydes[-1:,:],(idxf-tXUd.shape[0],1))
            ydes = np.vstack((ydes,ypad))
            
        return ydes
    
    def reset_memory(self,x0:np.ndarray,u0:np.ndarray=None,
                     fts0=None,pch0=None) -> None:
        """
        Method to reset the memory of the controller. This method is called
        at the beginning of each trajectory rollout to reset the controller's
        internal state and prepare it for a new trajectory.

        VehicleRateMPC does not have any internal state to reset, so this
        method is a no-op. However, it is included to maintain the interface
        with the BaseController class and to allow for future extensions
        where internal state might be added.
        
        Args:
            - x0: Initial state.
            - u0: Initial control input (unused).
            - fts0: Initial forces (unused).
            - pch0: Initial perturbations (unused).
        """
        # Unpack unused variables
        _ = x0,u0,fts0,pch0
        
        pass