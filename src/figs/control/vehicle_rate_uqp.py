
import time
import numpy as np
import figs.utilities.transform_helper as th

from pathlib import Path
from figs.control.base_controller import BaseController
from figs.dynamics.external_forces import ExternalForces
from figs.tsplines.min_time_snap import MinTimeSnap

class VehicleRateUQP(BaseController):
    def __init__(self,
                 policy:dict,course:dict,frame:dict=None,
                 name:str="vruqp",
                 configs_path:Path=None) -> None:
        
        """
        Constructor for the VehicleRateFQP class (Unconstrained Quadratic Program).
        
        Args:
            - policy:       Config Dict of the policy.
            - course:       Config Dict of the course.
            - frame:        Config Dict of the (drone) frame.
            - use_RTI:      Use RTI flag.
            - name:         Name of the controller.
            - configs_path: Path to the directory containing the JSON files.
            - solver_json:  Name of the solver JSON file.
        """

        # Initialize the BaseController
        super().__init__(configs_path)

        # Controller Parameters
        kT = policy["kT"]
        hz = policy["hz"]

        # Course Parameters
        WPs = course["waypoints"]
        Fcfg = course["forces"]

        # Frame Parameters
        if frame is None:       # Use placeholder values
            print("VehicleRateMPC initialized with placeholder frame parameters.")
            m,kt = 1.0,7.0
        else:
            m,kt = frame["mass"],frame["motor_thrust_coeff"]

        # Get initial solution
        mts = MinTimeSnap(WPs,kT,hz)
        Tsd,FOd = mts.get_ideal(hz)

        Fex = ExternalForces(Fcfg)

        tXUd = th.TsFO_to_tXU(Tsd,FOd,m,kt,Fex)

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
        self.mts = mts
        self.Tsd,self.FOd = Tsd,FOd
        self.tXUd = tXUd
        self.m,self.kt = m,kt
        self.Fex = Fex

    def update_frame(self,frame:dict) -> None:
        """
        Method to update the frame related variables of the controller.
        
        Args:
            - frame: Config Dict of the (drone) frame.

        """
        m,kt = frame["mass"],frame["motor_thrust_coeff"]
        
        tXUd = th.TsFO_to_tXU(self.Tsd,self.FOd,m,kt,self.Fex)

        self.tXUd = tXUd
        self.m,self.kt = m,kt

    def control(self,tcr:float,xcr:np.ndarray,
                upr:np.ndarray=None,
                obj:np.ndarray=None,
                icr:None=None,zcr:None=None) -> tuple[
                    np.ndarray,None,np.ndarray]:

        # Unpack class variables
        _ = upr,obj,icr,zcr
        Nfo,Ndr = self.mts.Nfo,self.mts.Ndr
        Tpkf,FOkf = self.mts.get_ideal(None)
        Tsd,FOd = self.Tsd,self.FOd
        m,kt = self.m,self.kt
        Fex = self.Fex

        # ===================================================================
        thn = 1.0
        Nhn = int(thn*self.hz)
        # ===================================================================
        
        # Start timer
        t0 = time.time()

        # Start point
        FO1 = th.x_to_fo(xcr,Nfo,Ndr)[None,:,:]

        # End point
        # pos_l2_err = np.linalg.norm(FOd[:,0:3,0]-FO1[:,0:3,0],axis=1)
        # idx = np.argmin(pos_l2_err)+Nhn
        idx = np.where(Tsd<=tcr)[0][-1]+Nhn
        idx = np.clip(idx,0,FOd.shape[0]-1)
        FO2 = FOd[idx:idx+1,:,:]

        FOcr = np.concatenate((FO1,FO2),axis=0)
        dTcr = np.array([thn])

        # Solve
        t1 = time.time()
        dT,Pn = self.mts.solve(FOcr,dTcr)

        # Extract command
        txu = th.dTPn_to_tXU([0.0],dT,Pn,m,kt,Fex)
        ucc = txu[11:15,0]

        t2 = time.time()

        # Compute timer values
        tsol = np.array([t1-t0,t2-t1,0.0,0.0])

        return ucc,None,tsol