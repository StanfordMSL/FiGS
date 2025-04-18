import time
import shutil
import os
import numpy as np
import scipy.linalg
import figs.utilities.transform_helper as th
import figs.dynamics.quadcopter_model as qm

from pathlib import Path
from copy import deepcopy
from casadi import vertcat
from acados_template import AcadosOcp, AcadosOcpSolver
from figs.control.base_controller import BaseController
from figs.dynamics.external_forces import ExternalForces
from figs.tsplines.min_snap import MinSnap

class VehicleRateMPC(BaseController):
    def __init__(self,
                 policy:dict,course:dict,frame:dict=None,
                 use_RTI:bool=False,
                 name:str="vrmpc",
                 configs_path:Path=None,
                 solver_json:str='figs_ocp_solver.json',) -> None:
        
        """
        Constructor for the VehicleRateMPC class.
        
        Args:
            - policy:       Config Dict of the policy.
            - course:       Config Dict of the course.
            - frame:        Config Dict of the (drone) frame.
            - use_RTI:      Use RTI flag.
            - name:         Name of the controller.
            - configs_path: Path to the directory containing the JSON files.
            - solver_json:  Name of the solver JSON file.

        Variables:
            - hz:              Controller frequency.
            - nzcr:            Feature vector size (if controller uses learned feedback. Set to None if not used).

            - Nx:              Number of states.
            - Nu:              Number of inputs.
            - p:               Model Parameters. (mass,thrust,fx,fy,fz)
            - Tpd:             Ideal TSpline time vector.
            - CPd:             Ideal TSpline control points.
            - tXUd:            Desired trajectory.
            - Fext:            External forces.
            - Qk:              State cost.
            - Rk:              Input cost.
            - QN:              Final state cost.
            - lbu:             Lower bound on inputs.
            - ubu:             Upper bound on inputs.
            - Ws:              Search cost.
            - ns:              Number of states to consider.
            - use_RTI:         Use RTI flag.
            - model:           Model of the system.
            - solver:          Solver object.
            - code_export_path: Path to the generated code.

        """

        # =====================================================================
        # Extract parameters
        # =====================================================================
        
        # Initialize the BaseController
        super().__init__(configs_path)

        # (MPC) Policy Parameters
        hz,Nhn,Ws = policy["hz"],policy["horizon"],np.diag(policy["Ws"])
        Qk,Rk,QN = np.diag(policy["Qk"]),np.diag(policy["Rk"]),np.diag(policy["QN"])
        lbu,ubu = np.array(policy["bounds"]["lower"]),np.array(policy["bounds"]["upper"])

        nx,nu = len(policy["Qk"]), len(policy["Rk"])
        ns = int(hz/5)
        ny,ny_e = nx+nu,nx

        # Course Parameters
        WPs = course["waypoints"]
        Fcfg = course["forces"]
        
        # Frame Parameters
        if frame is None:       # Use placeholder values
            m,kt = 1.0,7.0
        else:
            m,kt = frame["mass"],frame["motor_thrust_coeff"]
        p = np.hstack((m,kt,np.zeros(3)))
        
        # Get initial solution (padded)
        # WPs = self.pad_trajectory(WPs,Nhn,hz)
        ms = MinSnap(WPs,hz)
        Tsd,FOd = ms.get_ideal(hz)

        Fex = ExternalForces(Fcfg)

        tXUd = th.TsFO_to_tXU(Tsd,FOd,m,kt,Fex)

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

        ocp.constraints.x0 = tXUd[1:11,0]
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

        solver = AcadosOcpSolver(ocp,json_file=solver_json,verbose=False)
        
        # Clear the generated code
        os.remove(os.path.join(os.getcwd(),solver_json))
        shutil.rmtree(ocp.code_export_directory)

        # =====================================================================
        # Controller Variables
        # =====================================================================

        # ---------------------------------------------------------------------
        # Necessary Variables for Base Controller -----------------------------
        self.name = name
        self.hz = hz
        self.Nznn = {}
        self.nhy = 0

        # ---------------------------------------------------------------------
        # Controller Specific Variables
        self.Nx,self.Nu = nx,nu
        self.Tsd,self.FOd = Tsd,FOd
        self.tXUd,self.Fex = tXUd,Fex
        self.p = p
        self.Qk,self.Rk,self.QN = Qk,Rk,QN
        self.lbu,self.ubu = lbu,ubu
        self.Ws = Ws
        self.ns = ns
        self.use_RTI = use_RTI
        self.solver = solver

        # =====================================================================
        # Warm start the solver
        # =====================================================================

        for _ in range(5):
            self.control(0.0,tXUd[1:11,0])

    def update_frame(self,frame:dict) -> None:
        """
        Method to update the frame related variables of the controller.
        
        Args:
            - frame: Config Dict of the (drone) frame.

        """
        m,kt = frame["mass"],frame["motor_thrust_coeff"]
        
        tXUd = th.TsFO_to_tXU(self.Tsd,self.FOd,m,kt,self.Fex)

        self.p[0],self.p[1] = m,kt
        self.tXUd = tXUd

    def control(self,
                tcr:float,xcr:np.ndarray,
                upr:np.ndarray=None,
                obj:np.ndarray=None,
                icr:None=None,zcr:None=None) -> tuple[
                    np.ndarray,None,np.ndarray]:
        
        """
        Method to compute the control input for the VehicleRateMPC controller. We use the standard input arguments
        format with the unused arguments set to None. Likewise, we use the standard output format with the unused
        outputs set to None.

        Args:
            - tcr: Time at the current control step.
            - xcr: States at the current control step.
            - upr: Previous control step inputs (unused).
            - obj: Objective vector (unused).
            - icr: Image at the current control step (unused).
            - zcr: Feature vector at current control step (unused).

        Returns:
            - ucr:  Control input.
            - zcr:  Output feature vector (unused).
            - tsol: Time taken to solve components [setup ocp, solve ocp, unused, unused].

        """
        # Unused arguments
        _ = upr,obj,icr,zcr

        # Start timer
        t0 = time.time()

        # Get desired trajectory
        ydes = self.get_ydes(tcr,xcr)

        # Get external forces
        self.p[2:5] = self.Fex.get_forces(xcr[0:6])

        # Set desired trajectory
        for i in range(self.solver.acados_ocp.dims.N):
            self.solver.cost_set(i, "yref", ydes[:,i])
            self.solver.set(i,'x',ydes[0:10,i])
            self.solver.set(i,'u',ydes[10:,i])
            self.solver.set(i,'p',self.p)

        self.solver.cost_set(self.solver.acados_ocp.dims.N, "yref", ydes[0:10,-1])
        self.solver.set(self.solver.acados_ocp.dims.N,'x',ydes[0:10,-1])
        
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

            ucc = self.solver.get(0, "u")
        else:
            # Solve ocp and get next control input
            try:
                ucc = self.solver.solve_for_x0(x0_bar=xcr)
            except:
                print("Warning: VehicleRateMPC failed to solve OCP. Using previous input.")
                ucc = self.solver.get(0, "u")
        t2 = time.time()

        # Compute timer values
        tsol = np.array([t1-t0,t2-t1,0.0,0.0])

        return ucc,None,tsol

    def pad_trajectory(self,WPs:dict,Nhn:int,hz_ctl:float) -> dict:
        """
        Method to pad the trajectory with the final waypoint so that the MPC horizon is satisfied at the end of the trajectory.

        Args:
            - WPs:   Dictionary containing the flat output waypoints.
            - Nhn:        Prediction horizon.
            - hz_ctl:     Controller frequency.

        Returns:
            - WPs_pd: Padded flat output waypoints.

        """

        # Get final waypoint
        kff = list(WPs["keyframes"])[-1]
        
        # Pad trajectory
        t_pd = WPs["keyframes"][kff]["t"]+(Nhn/hz_ctl)
        fo_pd = np.array(WPs["keyframes"][kff]["fo"])[:,0:3].tolist()

        WPs_pd = deepcopy(WPs)
        WPs_pd["keyframes"]["fof"] = {"t":t_pd,"fo":fo_pd}

        return WPs_pd

    def get_ydes(self,tcr:float,xcr:np.ndarray) -> np.ndarray:
        """
        Method to get the section of the desired trajectory at the current time.

        Args:
            - tcr: Time at the current control step.
            - xcr: States at the current control step.

        Returns:
            - ydes:   Desired trajectory section at the current time.

        """
        # Get relevant portion of trajectory
        idx_i = int(self.hz*tcr)
        Nhn_lim = self.tXUd.shape[1]-self.solver.acados_ocp.dims.N-1
        ks0 = np.clip(idx_i-self.ns,0,Nhn_lim-1)
        ksf = np.clip(idx_i+self.ns,0,Nhn_lim)
        Xi = self.tXUd[1:11,ks0:ksf]
        
        # Find index of nearest state
        dXi = Xi-xcr.reshape(-1,1)
        wl2_dXi = np.array([x.T@self.Ws@x for x in dXi.T])
        idx0 = ks0 + np.argmin(wl2_dXi)
        idxf = idx0 + self.solver.acados_ocp.dims.N+1

        # Pad if idxf is greater than the last index
        if idxf < self.tXUd.shape[1]:
            ydes = self.tXUd[1:,idx0:idxf]
        else:
            print("Warning: VehicleRateMPC.get_ydes() padding trajectory. Increase your padding horizon.")
            ydes = self.tXUd[1:,idx0:]
            ydes = np.hstack((ydes,np.tile(ydes[:,-1:],(1,idxf-self.tXUd.shape[1]))))

        return ydes