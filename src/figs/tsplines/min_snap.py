import numpy as np
import scipy.sparse as sps
import scipy.linalg as spl
import sys
import math
import qpsolvers

from typing import Union,Tuple,Optional,Dict,List

# Debugging
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)

# Fixed parameters for minimum snap quadcopter trajectory planning problem
Nfo = 4                                                         # Number of Flat Outputs
kdr = np.array([4,4,4,2])                                       # Target derivative to minimize
mu  = np.array([1.0,1.0,1.0,1.0])                               # Scaling for each parameter

def solve(fout_wps:Dict[str,Union[int,Tuple[np.float64,np.ndarray]]],Natt=5) -> Optional[Tuple[np.ndarray,np.ndarray]]:
    """
    Solve the minimum snap trajectory planning problem.

    Args:
    fout_wps:   Dictionary containing the flat waypoints.
    Natt:       Number of attempts to solve the QP problem.

    Returns:
    Tps:        Time vector for each keyframe.
    CPs:        Coefficient matrix for each keyframe.

    """
    # Unpack data from dictionary
    keyframes: Dict[str,Tuple[np.float64,np.ndarray]] = fout_wps["keyframes"]
    Tp = [item['t'] for item in keyframes.values()]
    FOp = [np.array(item['fo'],dtype=float) for item in keyframes.values()]
    Nco = fout_wps["Nco"]

    # Generate QP Terms
    P,q = Pq_gen(Tp,Nco)                                           # Min Snap Cost
    A,b = Ab_gen(Tp,FOp,Nco)                                       # Keyframe Constraints

    # Regularize P to ensure strict positive-definiteness (helps ill-conditioned problems)
    n = P.shape[0]
    P += 1e-8 * np.eye(n)

    # Convert to Sparse
    P_sp = sps.csc_matrix(P)
    A_sp = sps.csc_matrix(A)

    # Solver-specific keyword arguments
    solver_kwargs = {
        "clarabel": {},
        "proxqp":   {"max_iter": 20000, "eps_abs": 1e-7},
        "osqp":     {"max_iter": 20000, "eps_abs": 1e-7, "eps_rel": 1e-7, "polish": True},
        "scs":      {"max_iters": 20000, "eps_abs": 1e-7, "eps_rel": 1e-7},
    }

    solver_priority = ["clarabel", "proxqp", "osqp", "scs"]
    solvers_to_try = [s for s in solver_priority if s in qpsolvers.available_solvers]

    for solver_name in solvers_to_try:
        kwargs = solver_kwargs.get(solver_name, {})
        for attempt in range(Natt):
            try:
                sigma = qpsolvers.solve_qp(P_sp,q,G=None,h=None,A=A_sp,b=b,
                                        solver=solver_name, **kwargs)
                if sigma is None:
                    raise RuntimeError(f"{solver_name} returned None")
                SM = sigma.reshape((-1,Nfo,Nco))

                Nsm = SM.shape[0]
                TT = np.zeros((Nsm,Nco))
                for i in range(0,Nsm):
                    TT[i,:] = np.linspace(Tp[i],Tp[i+1],Nco)

                Tps = np.array(Tp)
                CPs = SM2CP(SM,TT,Nco)

                return Tps,CPs

            except Exception as e:
                print(f"Min-snap solve failed ({solver_name}, attempt {attempt + 1}/{Natt}): {e}")

        print(f"All {Natt} attempts exhausted for solver '{solver_name}', trying next...")

    print("All solvers failed.")
    return None
    
def Pq_gen(Tp:List[np.float64],Nco:int) -> Tuple[np.ndarray,np.ndarray]:
    # Unpack some stuff
    Nsm = len(Tp)-1            # Number of segments

    Plist = []
    for i in range(0,Nsm):
        t0 = Tp[i]
        tf = Tp[i+1]

        for j in range(0,Nfo):
            P = mu[j]*Ps_gen(kdr[j],t0,tf,Nco)
            Plist.append(P)

    P = spl.block_diag(*Plist)
    q = np.zeros(Nsm*Nfo*Nco)

    return P,q

def Ps_gen(kdr:np.float64,t0:np.float64,tf:np.float64,Nco:int) -> np.ndarray:
    Ps = np.zeros((Nco,Nco))
    for i in range(kdr,Nco):
        for j in range(i,Nco):
            c1 = cf_gen(i,kdr)
            c2 = cf_gen(j,kdr)
            tk = 1+i+j-(kdr*2)

            Pij = c1*c2*((tf**tk)-(t0**tk))/tk

            Ps[i,j] = Pij
            Ps[j,i] = Pij

    return Ps

def Ab_gen(Tp:List[np.float64],FOp:List[np.ndarray],Nco:int) -> Tuple[np.ndarray,np.ndarray]:
    # Some useful intermediate variables
    Nsm = len(Tp)-1                            # Number of segments

    # Initialize output variables
    A = np.zeros((0,(Nco*Nfo*Nsm)))
    b = np.zeros(0)

    for i in range(Nsm):
        for j in range(Nfo):
            idx = (i*Nfo+j)*Nco

            fo0 = FOp[i][j,:]
            for k in range(fo0.shape[0]):
                b0 = fo0[k]

                a0 = np.zeros(Nco*Nfo*Nsm)
                ap = poly2kdr(Tp[i],k,Nco)

                if np.isnan(b0):
                    pass
                else:
                    a0[idx:idx+Nco] = ap

                    A = np.vstack((A,a0))
                    b = np.append(b,b0)

            fof = FOp[i+1][j,:]
            for k in range(fof.shape[0]):
                b0 = fof[k]

                a0 = np.zeros(Nco*Nfo*Nsm)
                ap = poly2kdr(Tp[i+1],k,Nco)

                if np.isnan(b0):
                    idxp = ((i+1)*Nfo+j)*Nco
                    a0[idx:idx+Nco] = ap
                    a0[idxp:idxp+Nco] = -ap

                    b0 = 0
                else:
                    a0[idx:idx+Nco] = ap

                A = np.vstack((A,a0))
                b = np.append(b,b0)

    return A,b

def cf_gen(N:int,k:int) -> np.float64:
    cfac = math.factorial(N)/math.factorial(N-k)

    return cfac

def poly2kdr(t:np.float64,kdr:int,Nco:int) -> np.ndarray:
    a = np.zeros(Nco)
    for i in range(kdr,Nco):
        c1 = cf_gen(i,kdr)
        a[i] = c1*(t**(i-kdr))

    return a
    
def SM2CP(SM:np.ndarray,TT:np.ndarray,Nco:int) -> np.ndarray:
    # Unpack some stuff
    Nsm = SM.shape[0]
    Ncp = TT.shape[1]

    # Output Variable
    CP = np.zeros((Nsm,Nfo,Ncp))

    # Roll-out trajectory
    for i in range(0,Nsm):
        for j in range(0,Nfo):                    # at the ends, so we zero them accordingly.
            for k in range(0,Ncp):
                a = poly2kdr(TT[i,k],0,Nco)
                CP[i,j,k] = a@SM[i,j,:]        

    return CP