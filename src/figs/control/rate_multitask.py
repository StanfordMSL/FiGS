import time
import shutil
import os
import numpy as np
from typing import Literal
from enum import Enum

import figs.utilities.config_helper as ch
import figs.utilities.transform_helper as th
import figs.dynamics.quadcopter_rate_model as qrm
import figs.control.state_machines as sm

from casadi import Function
from acados_template import AcadosOcp, AcadosOcpSolver
from figs.control.base_controller import BaseController
from figs.dynamics.external_forces import ExternalForces
from figs.visualize import rich_visuals as rv

class RateMultitask(BaseController):
    def __init__(self,
                 policy:str|dict,course:str|dict,frame:str|dict,
                 mode:Literal["RESET","APPROACH","FINETUNE","ALL"]="ALL",
                 use_RTI:bool=False,
                 name:str="rate_multitask",
                 debug:bool=False) -> None:

        """
        Constructor for the RateMultitask class.

        Args:
            - policy:       Config Dict of the policy.
            - course:       Config Dict of the course.
            - frame:        Config Dict of the (drone) frame.
            - use_RTI:      Use RTI flag.
            - name:         Name of the controller.

        Variables:
            -  name:        Name of the controller.
            -  hz:          Control frequency.
            -  Nx:          Number of states.
            -  Nu:          Number of inputs.
            -  m:           Mass of the frame.
            -  kt:          Thrust coefficient of the frame.
            -  uhov:        Hover command.
            -  tXUd:        Desired trajectory.
            -  fex:         External forces model.
            -  forces:      Current external forces.
            -  mode_00:     Initial mode of the controller.
            -  md_lock:     Mode lock flag.
            -  mode_cr:     Current mode of the controller.
            -  mode_t0:     Time of last mode switch.
            -  fb_count:    Fallback counter.
            -  fb_limit:    Fallback limit.
            -  fb_fail:     Fallback fail flag.
            -  states:      States dictionary.
            -  K:           Camera intrinsic matrix.
            -  solver:      Acados OCP solver.
            -  get_y_expr:  Casadi function to extract stagewise cost terms.
            -  get_y_expr_e:Casadi function to extract terminal cost terms.
            -  debug:       Debug flag.
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
        track = policy["track"]

        hz,Nhn = policy["track"]["hz"],policy["track"]["horizon"]
        states:dict[str,list|dict[str,dict]] = policy["track"]["states"]
        fb_limit = policy["track"]["tolerances"]["fallback_limit"]

        lbu,ubu = np.array(track["bounds"]["lower"]),np.array(track["bounds"]["upper"])

        # Frame Parameters
        m,kt = frame["mass"],frame["motor_thrust_coeff"]
        g = frame["gravity_vector"][2]
        nx,nu = frame["nx"],frame["nu"]
        nmtr = frame["number_of_rotors"]

        # Course Parameters
        WPs_cfg,Fs_cfg = course["waypoints"],course["forces"]
        fo0 = np.array(WPs_cfg["keyframes"]["fo0"]["fo"])
        fof = np.array(WPs_cfg["keyframes"]["fo1"]["fo"])
        t0,tf = WPs_cfg["keyframes"]["fo0"]["t"],WPs_cfg["keyframes"]["fo1"]["t"]

        # Desired Variables
        fex = ExternalForces(Fs_cfg)                    # External Forces
        x0 = th.fo_to_xu(fo0,m,kt,np.zeros(3))[0:10]    # Initial state (assuming no external forces)
        xf = th.fo_to_xu(fof,m,kt,np.zeros(3))[0:10]    # Final state (assuming no external forces)
        uhov = np.array([-m*g/(nmtr*kt),0.0,0.0,0.0])   # Hover command

        txu0 = np.hstack((t0,x0,uhov))                  # Initial trajectory point
        txuf = np.hstack((tf,xf,uhov))                  # Final trajectory point
        tXUd = np.vstack((txu0,txuf))                   # Complete desired trajectory

        # Mode Lock
        if mode == "ALL":
            md_lock = False
            mode_00,mode_0f = sm.bMode.RESET,sm.bMode.FINETUNE
            mode_cr = sm.bMode.RESET
        else:
            md_lock = True
            mode_00,mode_0f = sm.bMode[mode],sm.bMode[mode]
            mode_cr = sm.bMode[mode]
        
        # =====================================================================
        # Setup Non-Acados Variables
        # =====================================================================

        # Necessary Variables for Base Controller -----------------------------
        self.name = name
        self.hz = hz
        
        # Frame Changing Variables (affected by changes in m and kt) ----------
        self.m,self.kt = m,kt
        self.uhov = uhov
        self.tXUd = tXUd
        self.fof = fof
        
        # Controller Specific Variables ---------------------------------------
        self.debug = debug
        self.Nx,self.Nu = nx,nu
        self.fex = fex
        self.forces = np.zeros(3)
        self.mode_00,self.mode_0f,self.md_lock = mode_00,mode_0f,md_lock
        self.mode_cr,self.mode_t0 = mode_cr,t0
        self.fb_count = 0
        self.fb_limit = fb_limit
        self.fb_fail = False
        self.states = states
        self.K = np.array([
            [frame["camera"]["fx"], 0, frame["camera"]["cx"]],
            [0, frame["camera"]["fy"], frame["camera"]["cy"]],
            [0, 0, 1]
        ])
                
        # =====================================================================
        # Setup Acados Variables
        # =====================================================================

        # Extract cost terms
        W,We,yref,yref_e = self.get_cost_terms()

        # Initialize Acados OCP
        ocp = AcadosOcp()

        ocp.model = qrm.export_model()   
        ocp.parameter_values = np.zeros(ocp.model.p.shape)

        y_expr, y_expr_e = qrm.extract_homing_variables(ocp.model,xf[0:3],frame)
        ocp.model.cost_y_expr = y_expr
        ocp.model.cost_y_expr_e = y_expr_e

        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'

        ocp.cost.W,ocp.cost.W_e = W,We
        ocp.cost.yref,ocp.cost.yref_e = yref,yref_e

        ocp.constraints.x0 = x0
        ocp.constraints.lbu = lbu
        ocp.constraints.ubu = ubu
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        # Initialize Acados Solver
        ocp.solver_options.N_horizon = Nhn
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'EXACT'
        # ocp.solver_options.sim_method_newton_iter = 10

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

        self.solver = solver

        # Controller Specific Functions --------------------------------------
        self.get_y_expr = Function("get_y_expr",[ocp.model.x, ocp.model.u],[ocp.model.cost_y_expr])
        self.get_y_expr_e = Function("get_y_expr_e",[ocp.model.x],[ocp.model.cost_y_expr_e])
        
        # =====================================================================
        # Warm start the solver
        # =====================================================================
        
        self.reset_controller()

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
            - upr: Pyyrevious control input (unused).
            - rgb: RGB image (unused).
            - dpt: Depth image (unused).
            - fcr: Current force.

        Returns:
            - ucr:  Control input.
            - mcr:  Current mode.
            - tsol: Solve times dictionary with keys "setup_ocp" and "solve_ocp".
        """

        t0 = time.time()

        # Assign unused inputs
        _ = rgb,dpt,upr

        # Update the state machine
        if self.md_lock is False:
            self.update_state_machine(tcr,xcr,fcr)
        else:
            self.mode_cr = self.mode_00
        self.update_ocp()

        t1 = time.time()
        
        # Solve the OCP
        try:
            ucr = self.solver.solve_for_x0(xcr,print_stats_on_failure=False)
        except:
            if self.md_lock is False:
                # First Fallback Strategy
                self.reset_controller(sm.bMode.RESET,tcr,hard_reset=False)
                self.update_ocp()

                try:
                    ucr = self.solver.solve_for_x0(xcr,print_stats_on_failure=False)
                except:
                    # Second Fallback Strategy
                    self.fb_count += 1
                    ucr = self.uhov

                    if self.fb_count >= self.fb_limit:
                        self.fb_fail = True

                if self.debug:
                    if self.fb_fail:
                        print("Solver failed main fallback strategy at time:",tcr,". Using hover command.")
                    else:
                        print("Solver recovered via fallback strategy at time:",tcr,". Switching to",self.mode_cr.name,"mode.")
            else:
                # Fallback Strategy for Mode Locked Controllers
                self.fb_count += 1
                ucr = self.uhov

                if self.fb_count >= self.fb_limit:
                    self.fb_fail = True

                if self.debug:
                    if self.fb_fail:
                        print("Solver failed main fallback strategy at time:",tcr,". Using hover command.")

        t2 = time.time()

        # Assemble remainder of outputs
        mcr = self.mode_cr.value
        tsol = {"setup_ocp":t1-t0,
                "solve_ocp":t2-t1
                }

        return ucr,mcr,tsol

    def reset_controller(self,mode:sm.bMode=None,tcr:float=None,hard_reset:bool=True) -> None:
        """
        Method to reset the solver of the controller.

        Args:
            - hard_reset: If True, resets the fail_flag to False.
        """

        # Assign default values
        if mode is None:
            mode = self.mode_00
        if tcr is None:
            tcr = 0.0

        # Reset the mode variables
        self.mode_cr = mode
        self.mode_t0 = tcr

        # Reset the solver
        for i in range(self.solver.acados_ocp.dims.N):
            lam = self.solver.get(i,"lam")
            pi = self.solver.get(i,"pi")

            self.solver.set(i,"lam",0.0*lam)
            self.solver.set(i,"pi",0.0*pi)
            self.solver.set(i,"x",self.tXUd[0,1:11])
            self.solver.set(i,"u",self.uhov)

        self.solver.set(self.solver.acados_ocp.dims.N,"x",self.tXUd[0,1:11])

        if hard_reset:
            self.fb_fail = False
            self.fb_count = 0

        if self.debug:
            print("Mode set to:",self.mode_cr.name)
        
    def update_frame(self,frame:str|dict) -> None:
        """
        Method to update the frame related variables of the controller.
        
        Args:
            - frame: Config Dict of the (drone) frame.

        """

        # Check if frame is a string or dictionary
        if isinstance(frame, str):
            frame = ch.get_config(frame, "frames")
        
        # Extract/compute relevant parameters
        m,kt = frame["mass"],frame["motor_thrust_coeff"]
        g = frame["gravity_vector"][2]
        nmtr = frame["number_of_rotors"]

        uhov = np.array([-m*g/(nmtr*kt),0.0,0.0,0.0])           # Hover command

        # Update the relevant variables
        self.m,self.kt = m,kt
        self.uhov = uhov
        self.tXUd[0,11:15] = uhov
        self.tXUd[1,11:15] = uhov   

    def update_state_machine(self,tcr:float,xcr:np.ndarray,fcr:np.ndarray) -> int:
        """
        Method to compute the current mode of the controller based on the current state.
        
        Args:
            tcr: Current time.
            xcr: Current state.

        Returns:
            None    

        """
                
        # Extract conditions and tolerances
        mode_cr = self.mode_cr
        conditions = self.get_condition_values(tcr, xcr, fcr)
        tolerances_dict = self.states[mode_cr.name]["tolerances"]

        # Update state machine
        mode_upd = mode_cr
        for mode_name, tolerances in tolerances_dict.items():
            if sm.check_conditions(conditions, tolerances):
                mode_upd = sm.bMode[mode_name]
                break
        self.mode_cr = mode_upd
        
        if self.debug and mode_cr != mode_upd:
            print("Switching to", self.mode_cr.name, "Mode at time:", tcr)

        # Update forces
        self.forces = fcr[0:3]

    def update_ocp(self) -> None:
        """
        Method to update the OCP of the controller based on the current time and state.
        """

        # Get the relevant cost weights
        W,We,yref,yref_e = self.get_cost_terms()
        # Assemble the parameter variable
        p = np.hstack((self.m,self.kt,self.forces))

        # Update the OCP solver with the new weights and parameters
        for i in range(self.solver.acados_ocp.dims.N):
            self.solver.cost_set(i,"W",W)
            self.solver.set(i,'yref',yref)
            self.solver.set(i,'p',p)

        self.solver.cost_set(self.solver.acados_ocp.dims.N,"W",We)
        self.solver.set(self.solver.acados_ocp.dims.N,'yref',yref_e)
        self.solver.set(self.solver.acados_ocp.dims.N,'p',p)

    def get_cost_terms(self) -> tuple[np.ndarray,np.ndarray]:
        """
        Method to get the current terms of the controller.

        Args:
            - weights: Weights dictionary. If None, uses the current mode's weights.

        Returns:
            - W:  Current stagewise weights.
            - We: Current terminal weights.
            - yref: Current stagewise cost reference.
            - yref_e: Current terminal cost reference.
        """

        # Extract State Variables
        state:dict[str,dict[str,list]] = self.states[self.mode_cr.name]
        stagewise,terminal,offset = state["stagewise"],state["terminal"], state["offset"]

        # Assemble the weights
        W,We = [],[]
        for value in stagewise.values():
            W.append(value)

        for value in terminal.values():
            We.append(value)

        W,We = np.diag(np.hstack(W)), np.diag(np.hstack(We))
        # Assemble the references
        pt_p_ds = np.array(offset)
        vb_b_ds = np.zeros(3)                           # Desired body velocity
        head_ds = self.fof[3,0]                         # Desired heading
        uv_n_ds = np.zeros(2)                           # Desired normalized pixel coordinates
        ratt_ds = np.zeros(2)                           # Desired normalized pixel coordinates
        datt_ds = np.zeros(2)                           # Desired relative attitude
        u_br_ds = self.uhov                             # Desired relative attitude rate

        yref = np.hstack((pt_p_ds,vb_b_ds,head_ds,uv_n_ds,ratt_ds,datt_ds,u_br_ds))     # Stagewise Cost Reference
        yref_e = np.hstack((pt_p_ds,vb_b_ds,head_ds,uv_n_ds,ratt_ds))                   # Terminal Cost Reference
        
        return W,We,yref,yref_e
    
    def get_cost_values(self,xcr:np.ndarray,ucr:np.ndarray|None=None) -> dict[str,np.ndarray]:
        """
        Method to compute the current stagewise cost terms.

        Args:
            - xcr: Current state.
            - ucr: Current input.

        Returns:
            - ydict: Current stagewise cost terms.
        """
        # Extract the relevant cost terms
        if ucr is None:
            y_expr:np.ndarray = self.get_y_expr_e(xcr).full().flatten()
            weights:dict[str,list] = self.states["RESET"]["terminal"]
        else:
            y_expr:np.ndarray = self.get_y_expr(xcr,ucr).full().flatten()
            weights:dict[str,list] = self.states["RESET"]["stagewise"]

        # Assemble the cost values dictionary
        ydict,i = {},0
        for k,v in weights.items():
            size = len(v)
            ydict[k] = y_expr[i:i+size]
            i += size

        return ydict

    def get_condition_values(self,tcr:float,xcr:np.ndarray,fcr:np.ndarray) -> dict[str,np.ndarray]:
        """
        Method to compute the current condition terms.

        Args:
            - xcr: Current state.

        Returns:
            - cdict: Current condition terms.
        """

        # Extract the relevant cost terms
        ydict = self.get_cost_values(xcr)                   # Current cost states

        conditions = {
            "distance": np.linalg.norm(ydict["pt_p"]),      # Probe to target distance
            "npixels": np.linalg.norm(ydict["uv_n"]),       # Normalized pixel coordinates
            "duration": tcr-self.mode_t0,                   # Duration in current mode
            "forces": np.linalg.norm(fcr),                  # Current external forces
            "azim_bnds": np.abs(ydict["ratt"][0]),          # Azimuth angle
        }

        return conditions