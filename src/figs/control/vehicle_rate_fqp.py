import time
import shutil
import os
import numpy as np
import scipy.sparse as sps
import scipy.linalg as spl
import math
import figs.tsplines.min_snap as ms
import figs.utilities.polynomial_helper as ph
import figs.dynamics.quadcopter_model as qm
from typing import Literal
import qpsolvers
from pathlib import Path
from numpy.polynomial.legendre import Legendre
from figs.control.base_controller import BaseController

class VehicleRateFQP(BaseController):
    def __init__(self,
                 policy:dict,course:dict,frame:dict=None,
                 name:str="vroqp",
                 configs_path:Path=None) -> None:
        
        """
        Constructor for the VehicleRateFQP class (Fielded Quadratic Program).
        
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

        # Course Parameters
        WPs = course["waypoints"]
        Fcfg = course["forces"]
        Nco = WPs["Nco"]
        KFs:dict = WPs["keyframes"]

        # Controller Parameters
        hz,Kdr = policy["hz"],policy["Kdr"]
        Nfo = len(Kdr)

        # Extract Flat Output Parameters
        Ts,FOs, = [],[]
        for keyframe in KFs.values():
            Ts.append(keyframe["t"])
            FOs.append(keyframe["fo"])
        Nsm = len(Ts)-1

        # Solve Min Snap for each Flat Output
        Ps,qs,As,bs = [],[],[],[]
        for i in range(Nfo):
            # Unpack some stuff
            kdr = Kdr[i]
            fos = [FO[i] for FO in FOs]

            # Compute cost terms
            Pf,qf = self.Pq_gen(Ts,kdr,Nco)    
            Ps.append(Pf),qs.append(qf)

            # Compute constraints terms
            Af,bf = self.Ab_gen(Ts,fos,Nco)
            As.append(Af),bs.append(bf)

        P = sps.block_diag(Ps)
        q = np.vstack(qs)
        A = sps.block_diag(As)
        b = np.vstack(bs)

        c = qpsolvers.solve_qp(P,q,G=None,h=None,A=A,b=b,
                                    solver="osqp")       # Solve QP
        
        # Construct Mapping matrix
        Ms = []
        for i in range(Nfo):
            for j in range(Nsm):
                Mj = ph.get_control_points_map(Ts[j],Ts[j+1],Nco,0)
                Ms.append(Mj)
        M = sps.block_diag(Ms)

        # Test out the mapping
        X = M@c
        X = X.reshape((4,Nsm,-1))

        for i in range(5):
            print(np.around(X[:,i,:],2))
            print("="*40)
        # cost = 0.5*c.T@P@c + q.T@c
        # print(f"Cost: {cost}")

        # t_eval = np.linspace(4.0,6.0,10)
        # tau_eval = 2*(t_eval-4.0)/(6.0-4.0)-1
        # rx = Legendre(coeffs[0,2,:])
        # ry = Legendre(coeffs[1,2,:])
        # rz = Legendre(coeffs[2,2,:])
        # psi = Legendre(coeffs[3,2,:])
        
        # X = np.zeros((4,10))
        # X[0,:] = rx(tau_eval)
        # X[1,:] = ry(tau_eval)
        # X[2,:] = rz(tau_eval)
        # X[3,:] = psi(tau_eval)
        # print(np.around(X,2))  
        # V = spl.block_diag(*[Vb]*4*(len(Ts)-1))
        # CPs = (V@c).reshape((Nfo,Nsm,-1))
        # print(np.around(CPs[:,0,:],2))
        # print(np.around(CPs[:,1,:],2))
        # print(np.around(CPs[:,2,:],2))
        # print(np.around(CPs[:,3,:],2))

        # print(CPs)
        # print("="*40)
    
    def Pq_gen(self,Ts:list[float], kdr:int, Nco:int,
               use_sparse:bool=True) -> tuple[np.ndarray,np.ndarray]:
        """
        Generate the P and q matrices for the quadratic program.

        Args:
            Ts:         List of time points.
            kdr:        Order of the derivative.
            Nco:        Number of coeffeciants.
            use_sparse: Use sparse matrix format.

        Returns:
            P:     Quadratic term.
            q:     Linear term.
        """

        Nsm = len(Ts)-1
        P = np.zeros((Nsm*Nco,Nsm*Nco))
        q = np.zeros((Nsm*Nco,1))

        for j in range(Nsm):
            idx0,idxf = j*Nco,(j+1)*Nco  
            Tj = Ts[j+1]-Ts[j]
            Pj = ph.get_legendre_integral(Tj,kdr,Nco)

            P[idx0:idxf,idx0:idxf] = Pj

        # Convert the matrix to sparse
        if use_sparse:
            P = sps.csc_matrix(P)

        return P,q
    
    def Ab_gen(self,Ts:list[float],fos:list[list[float,None]],Nco:int,
               use_sparse:bool=True) -> tuple[np.ndarray,np.ndarray]:
        """
        Generate the A and b matrices for the quadratic program.
        Args:
            Ts:         List of time points.
            fos:        List of flat outputs.
            Nsm:        Number of segments.
            Nco:        Number of control points.
            use_sparse: Use sparse matrix format.

        Returns:
            A:     Constraint matrix.
            b:     Constraint vector.
        """

        # Unpack some stuff
        Nsm = len(fos)-1
        Nct = self.get_Nct(fos)

        # Fill in the constraint matrix
        A,b,kct = np.zeros((Nct,(Nco*Nsm))),np.zeros((Nct,1)),0
        for ksm in range(Nsm):
            # Unpack some stuff
            Tk = Ts[ksm+1]-Ts[ksm]
            fok,fon = fos[ksm],fos[ksm+1]
            Npad = 0 if ksm == Nsm-1 else 1

            # Compute beginning constraints
            Asm,bsm = [],[]
            for kdr,val in enumerate(fok):
                if val is not None:
                    A0 = ph.get_legendre_vector(-1.0,Tk,kdr,Nco)
                    A1 = np.zeros((1,Nco*Npad))

                    Ai,bi = np.hstack((A0,A1)),np.array([val])
                else:
                    Ai,bi = np.zeros((0,Nco*(Npad+1))),np.zeros((0,1))

                Asm.append(Ai),bsm.append(bi)
            
            # Compute end constraints
            for kdr,val in enumerate(fon):
                if val is not None:
                    A0 = ph.get_legendre_vector(1.0,Tk,kdr,Nco)
                    A1 = np.zeros((1,Nco*Npad))

                    Ai,bi = np.hstack((A0,A1)),np.array([val])
                else:
                    Tn = Ts[ksm+2]-Ts[ksm+1]
                    A0 = ph.get_legendre_vector(1.0,Tk,kdr,Nco)
                    A1 = ph.get_legendre_vector(-1.0,Tn,kdr,Nco)

                    Ai,bi = np.hstack((A0,-A1)),np.array([0.0])

                Asm.append(Ai),bsm.append(bi)

            # Pack the constraints
            Asm,bsm = np.vstack(Asm),np.vstack(bsm)
            r0,r1 = kct,kct+Asm.shape[0]
            c0,c1 = ksm*Nco,(ksm+1+Npad)*Nco
            A[r0:r1,c0:c1],b[r0:r1,:] = Asm,bsm

            # Update the constraint index
            kct = r1

        # Convert the matrix to sparse
        if use_sparse:
            A = sps.csc_matrix(A)

        return A,b

    def get_Nct(self, fos:list[list[float,None]]) -> int:
        """
        Get the number of constraints for the given flat outputs.
        Args:
            fos:   List of flat outputs.

        Returns:
            Nct:   Number of constraints.
        """
        Nct,Nsm = 0,len(fos)-1
        for idx,fo in enumerate(fos):
            for x in fo:
                if x is None or idx == 0 or idx == Nsm:
                    Nct += 1
                else:
                    Nct += 2

        return Nct
    
    def control(self, tcr: float, xcr: np.ndarray):
        return 'ha'
    
def solve(WPs:dict[str,int|tuple[np.float64,np.ndarray]],hz:int=20,
          Natt=5) -> dict[str,tuple[np.ndarray,np.ndarray]]:
    """
    Solve the minimum snap trajectory planning problem.

    Args:
        config: Dictionary containing the course configuration.
        Natt:   Number of attempts to solve the QP problem.

    Returns:
        output: Dictionary containing the solution in its various forms.

    """

    # Unpack data from dictionary
    keyframes:dict = WPs["keyframes"]
    Tp = [item['t'] for item in keyframes.values()]
    FOp = [np.array(item['fo'],dtype=float) for item in keyframes.values()]
    Nco = WPs["Nco"]
    
    
#     # Generate QP Terms
#     P,q = Pq_gen(Tp,Nco)                                           # Min Snap Cost
#     A,b = Ab_gen(Tp,FOp,Nco)                                       # Keyframe Constraints

#     # Convert to Sparse
#     P = sps.csc_matrix(P)
#     A = sps.csc_matrix(A)

#     # Solve QP to get coefficient solution (spline variables)
#     for attempt in range(Natt):
#         try:
#             sigma = qpsolvers.solve_qp(P,q,G=None,h=None,A=A,b=b,
#                                     solver="osqp")       # Solve QP
#             SM = sigma.reshape((-1,Nfo,Nco))                                # Reshape to match keyframes

#             Nsm = SM.shape[0]
#             TT = np.zeros((Nsm,Nco))
#             for i in range(0,Nsm):
#                 TT[i,:] = np.linspace(Tp[i],Tp[i+1],Nco)
            
#             Tps,CPs = np.array(Tp),SM2CP(SM,TT,Nco)
#             Tss,FOs = th.TpCP_to_TsFO(Tps,CPs,hz)          # Convert to Flat Output
            
#             # Package Output
#             output = {
#                 "QP": {"P":P,"q":q,"x":sigma,"A":A,"b":b},
#                 "CP": (Tps,CPs),
#                 "FO": (Tss,FOs),
#             }

#             return output
        
#         except:
#             print(f"Minimum Snap Trajectory Solve Failed (Attempt {attempt + 1}) failed. Retrying...")
#             if attempt == Natt - 1:
#                 raise Exception("Minimum Snap Trajectory Solve Failed. Please check the input data.")
    
# def Pq_gen(Tp:list[float],Nco:int) -> tuple[np.ndarray,np.ndarray]:
#     # Unpack some stuff
#     Nsm = len(Tp)-1            # Number of segments

#     Plist = []
#     for i in range(0,Nsm):
#         t0 = Tp[i]
#         tf = Tp[i+1]

#         for j in range(0,Nfo):
#             P = mu[j]*Ps_gen(kdr[j],t0,tf,Nco)
#             Plist.append(P)

#     P = spl.block_diag(*Plist)
#     q = np.zeros(Nsm*Nfo*Nco)

#     return P,q

# def Ps_gen(kdr:float,t0:float,tf:float,Nco:int) -> np.ndarray:
#     Ps = np.zeros((Nco,Nco))
#     for i in range(kdr,Nco):
#         for j in range(i,Nco):
#             c1 = cf_gen(i,kdr)
#             c2 = cf_gen(j,kdr)
#             tk = 1+i+j-(kdr*2)

#             Pij = c1*c2*((tf**tk)-(t0**tk))/tk

#             Ps[i,j] = Pij
#             Ps[j,i] = Pij

#     return Ps

# def Ab_gen(Tp:list[float],FOp:list[np.ndarray],Nco:int) -> tuple[np.ndarray,np.ndarray]:
#     # Some useful intermediate variables
#     Nsm = len(Tp)-1                            # Number of segments

#     # Initialize output variables
#     A = np.zeros((0,(Nco*Nfo*Nsm)))
#     b = np.zeros(0)

#     for i in range(Nsm):
#         for j in range(Nfo):
#             idx = (i*Nfo+j)*Nco

#             fo0 = FOp[i][j,:]
#             for k in range(fo0.shape[0]):
#                 b0 = fo0[k]

#                 a0 = np.zeros(Nco*Nfo*Nsm)
#                 ap = poly2kdr(Tp[i],k,Nco)

#                 if np.isnan(b0):
#                     pass
#                 else:
#                     a0[idx:idx+Nco] = ap

#                     A = np.vstack((A,a0))
#                     b = np.append(b,b0)

#             fof = FOp[i+1][j,:]
#             for k in range(fof.shape[0]):
#                 b0 = fof[k]

#                 a0 = np.zeros(Nco*Nfo*Nsm)
#                 ap = poly2kdr(Tp[i+1],k,Nco)

#                 if np.isnan(b0):
#                     idxp = ((i+1)*Nfo+j)*Nco
#                     a0[idx:idx+Nco] = ap
#                     a0[idxp:idxp+Nco] = -ap

#                     b0 = 0
#                 else:
#                     a0[idx:idx+Nco] = ap

#                 A = np.vstack((A,a0))
#                 b = np.append(b,b0)

#     return A,b

# def cf_gen(N:int,k:int) -> np.float64:
#     cfac = math.factorial(N)/math.factorial(N-k)

#     return cfac

# def poly2kdr(t:float,kdr:int,Nco:int) -> np.ndarray:
#     a = np.zeros(Nco)
#     for i in range(kdr,Nco):
#         c1 = cf_gen(i,kdr)
#         a[i] = c1*(t**(i-kdr))

#     return a
    
# def SM2CP(SM:np.ndarray,TT:np.ndarray,Nco:int) -> np.ndarray:
#     # Unpack some stuff
#     Nsm = SM.shape[0]
#     Ncp = TT.shape[1]

#     # Output Variable
#     CP = np.zeros((Nsm,Nfo,Ncp))

#     # Roll-out trajectory
#     for i in range(0,Nsm):
#         for j in range(0,Nfo):                    # at the ends, so we zero them accordingly.
#             for k in range(0,Ncp):
#                 a = poly2kdr(TT[i,k],0,Nco)
#                 CP[i,j,k] = a@SM[i,j,:]        

#     return CP