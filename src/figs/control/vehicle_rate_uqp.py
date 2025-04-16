
import numpy as np
import scipy.sparse as sps
import figs.utilities.polynomial_helper as ph
import figs.utilities.transform_helper as th
import qpsolvers
from pathlib import Path
from figs.control.base_controller import BaseController
from figs.dynamics.external_forces import ExternalForces
import figs.tsplines.min_snap as ms
from figs.tsplines.min_time_snap import MinTimeSnap
import time

class VehicleRateUQP(BaseController):
    def __init__(self,
                 policy:dict,course:dict,frame:dict=None,
                 name:str="vroqp",
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
        Fex = ExternalForces(Fcfg)

        # Frame Parameters
        if frame is None:       # Use placeholder values
            m,kt = 1.0,7.0
        else:
            m,kt = frame["mass"],frame["motor_thrust_coeff"]

        # Compute desired trajectory
        mts = MinTimeSnap(WPs,kT,hz)

        tf = mts.Tp[-1]
        Ts = np.linspace(0.0,tf,int(tf*hz)+1)
        
        Nn = 60
        t0 = time.time()
        for i in range(Nn):
            Tp,Pn = mts.solve()
        t1 = time.time()
        dt = (t1-t0)/Nn
        hz_out = 1/dt
        print(f"VehicleRateUQP: {hz_out} Hz")


        self.tXUd = th.TpPn_to_tXU(Tp,Pn,hz,m,kt,Fex)

        # # Compute controller P,V matrices
        # ti,tf = 0.0,Nhn/hz
        # Ps,Vs = [],[]
        # for j in range(Nfo):
        #     Ps.append(lms.P_gen([ti,tf],Kdr[j],Nco))
        #     Vs.append(ph.get_control_points_map(ti,tf,Nco))
        # P = sps.block_diag(Ps, format='csc')    
        # V = sps.block_diag(Vs)

        # # Class variables
        # self.hz = hz
        # self.Nfo,self.Nco = Nfo,Nco
        # self.Nsh,self.Kdr = Nsh,Kdr
        # self.Nhn = Nhn

        # self.m,self.kt = m,kt
        # self.Tsd,self.FOd = Tsd,FOd
        # self.tXUd,self.Fex = tXUd,Fex

        # self.P,self.V = P,V

    def control(self,
                tpr:float, xpr:np.ndarray, upr:np.ndarray) -> np.ndarray:

        """
        Control method to compute the control input for the vehicle.

        Args:
            - tcr: Current time.
            - xcr: Current state vector.
            - xupr: Previous state vector (optional).

        Returns:
            - ucr: Control input.
        """

        # Some useful constants
        hz = self.hz
        dt = self.Nhn/self.hz
        Nfo,Nco = self.Nfo,self.Nco
        Nsh = self.Nsh
        Nd = self.Tsd.shape[0]
        Nhn = self.Nhn
        m,kt = self.m,self.kt
        P,V = self.P,self.V
        
        Ts = [0.0,dt]

        # Get estimate of previous immediate flat output
        fex = self.Fex.get_forces(xpr[0:6])
        xu_pr = np.hstack((xpr,upr))
        fok = th.xu_to_fo(xu_pr,m,kt,fex)

        # Find the nearest point in the trajectory
        fosh = fok[:,0]

        idxi = int(hz*tpr)
        idxs_sh = np.arange(idxi+Nsh[0],idxi+Nsh[1])
        idxs_sh = np.clip(idxs_sh,0,self.Tsd.shape[0]-1)        
        FOsh = self.FOd[idxs_sh[0]:idxs_sh[-1],:,0]

        dFO = np.linalg.norm(FOsh-fosh,axis=(1))
        idx0 = idxs_sh[0]+np.argmin(dFO)
        idxf = min(idx0+Nhn,Nd-1)
        
        # Find the solution trajectory with the least snap
        fo0 = fok.tolist()
        cost_prev = np.inf
        costs = []
        sigmas = []
        for i in range(idxf,Nd):
            fo1 = self.FOd[idxf,:,0:3].tolist()
        
            # Generate the QP terms
            As,bs = [],[]
            for j in range(Nfo):
                fos = [fo0[j],fo1[j]]

                Af,bf = lms.Ab_gen(Ts,fos,Nco)
                As.append(Af),bs.append(bf)
                
            A,b = sps.block_diag(As, format='csc'),np.vstack(bs)

            # Solve the QP
            sigma = qpsolvers.solve_qp(P,q=None,G=None,h=None,A=A,b=b,
                                solver="osqp")       # Solve QP

            # Termination condition
            cost = 0.5*sigma.T@P@sigma

            sigmas.append(sigma)
            costs.append(cost)

            # # if cost < cost_prev:
            # Ts = np.array(Ts)
            # CPs = np.array(V@sigma).reshape((Nfo,-1,Nco))
            # CPs = np.transpose(CPs,(1,0,2))

            # fo = th.CP_to_fo(Ts[0],Ts[1],CPs[0,:,:])
            # xu = th.fo_to_xu(fo,m,kt,fex)
            # ucr = xu[10:14]

            # cost_prev = cost
            # else:
            #     break
        idx_out = np.argmin(costs)
        sigma = sigmas[idx_out]
        Ts = np.array(Ts)
        CPs = np.array(V@sigma).reshape((Nfo,-1,Nco))
        CPs = np.transpose(CPs,(1,0,2))

        cost = costs[idx_out]

        Ts,FO = th.TpCP_to_TsFO(Ts,CPs,20)
        tXU = th.TsFO_to_tXU(Ts,FO,m,kt,self.Fex)
        return tXU,cost

