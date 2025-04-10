import time
import shutil
import os
import numpy as np
import scipy.sparse as sps
import scipy.linalg as spl
import math
import figs.tsplines.min_snap as ms
import figs.utilities.trajectory_helper as th
import figs.dynamics.quadcopter_model as qm
from typing import Literal
import qpsolvers
from numpy.polynomial.legendre import Legendre

from pathlib import Path
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

        # Get base matrices
        Vb = self.get_base_projection(Nco)

        # Extract Flat Output Parameters
        Ts,FOs, = [],[]
        for keyframe in KFs.values():
            Ts.append(keyframe["t"])
            FOs.append(keyframe["fo"])
        Nsm = len(Ts)-1

        # Solve Min Snap for each Flat Output
        P,q,A,b = [],[],[],[]
        for i in range(len(Kdr)):
            fos = [FO[i] for FO in FOs]

            # Compute Cost Matrices
            Pb = self.get_base_integral_cost(Kdr[i],Nco)
            Pi,qi = self.Pq_gen(Ts,Pb,Nco)
            P.append(Pi)
            q.append(qi)

            # Compute Constraint Matrices
            Ai,bi = self.Ab_gen(fos,Nco)
            A.append(Ai)
            b.append(bi)

        P = sps.block_diag(P)
        q = np.vstack(q)
        A = sps.block_diag(A)
        b = np.vstack(b)

        c = qpsolvers.solve_qp(P,q,G=None,h=None,A=A,b=b,
                                    solver="osqp")       # Solve QP
        cost = 0.5*c.T@P@c + q.T@c
        print(f"Cost: {cost}")
        # V = spl.block_diag(*[Vb]*4*(len(Ts)-1))
        # CPs = (V@c).reshape((Nfo,Nsm,-1))
        # print(np.around(CPs[:,0,:],2))
        # print(np.around(CPs[:,1,:],2))
        # print(np.around(CPs[:,2,:],2))
        # print(np.around(CPs[:,3,:],2))

        # print(CPs)
        # print("="*40)
    def get_base_integral_cost(self,p:int,n:int) -> np.ndarray:
        """
        Generate the base integral cost (integral of the square of the p-th
        derivative).

        Args:
            p:     Order of the derivative.
            Ncp:   Number of control points.

        Returns:
            Q:     Base integral cost matrix.
        """

        Q = np.zeros((n,n))
        for i in range(p,n):
            for j in range(p,n):
                fi = math.factorial(i)/math.factorial(i-p)
                fj = math.factorial(j)/math.factorial(j-p)
                den = 1+i+j-(2*p)

                Q[i,j] = fi*fj/den  # Integral of the square of the p-th derivative

        return Q

    def get_base_projection(self,n:int) -> np.ndarray:
        """
        Generate the base projection matrix that maps coefficients to
        control points.

        Args:
            n:     Number of control points.

        Returns:
            V:     Base projection matrix (coefficients to control points).
        """

        P = np.zeros((n,n))
        for i in range(0,n):
            for j in range(0,n):
                P[i,j] = (i/(n-1))**j

        return P

    def compute_equality(self, x:float, kdr:int, Nco:int, mode:Literal["initial","final"]):
        """
        Generate the equality constraint matrix.

        Args:
            x:      Value of the derivative.
            kdr:    Order of the derivative.
            Nco:    Number of control points.
            mode:   Mode of the equality constraint ("initial" or "final").

        Returns:
            Aeq:   Continuity constraint matrix.
            beq:   Continuity constraint vector.
        """

        Aeq,beq = np.zeros((1,Nco)), np.array(-x).reshape((1,1))
        if mode == "initial":
            Aeq[0,kdr] = -math.factorial(kdr)
        elif mode == "final":
            for i in range(kdr,Nco):
                Aeq[0,i] = math.factorial(i)/math.factorial(i-kdr)
        else:
            raise ValueError("Invalid mode. Use 'initial' or 'final'.")
        
        return Aeq,beq
        
    def compute_continuity(self, kdr:int, Nco:int):
        """
        Generate the continuity constraint matrix.

        Args:
            kdr:    Order of the derivative.
            Nco:    Number of control points.

        Returns:
            Aeq:    Continuity constraint matrix.
            beq:    Continuity constraint vector.
        """

        Aeq0 = self.compute_equality(0, kdr, Nco, mode="final")[0]
        Aeq1 = self.compute_equality(0, kdr, Nco, mode="initial")[0]
        Aeq = np.hstack((Aeq0, Aeq1))
        beq = np.zeros((1,1))

        return Aeq,beq

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
    
    def Pq_gen(self,Ts:list[float],Pb,Nco:int) -> tuple[np.ndarray,np.ndarray]:
        """
        Generate the P and q matrices for the quadratic program.

        Args:
            Ts:    List of time points.
            Nco:   Number of control points.
            Pb:    Base integral cost matrix.

        Returns:
            P:     Quadratic term.
            q:     Linear term.
        """
        Nsm = len(Ts)-1
        P = np.zeros((Nsm*Nco,Nsm*Nco))
        q = np.zeros((Nsm*Nco,1))

        for j in range(Nsm):
            idx0,idxf = j*Nco,(j+1)*Nco
            h = (Ts[j+1]-Ts[j])**2
            P[idx0:idxf,idx0:idxf] = h*Pb

        # Convert the matrix to sparse
        P = sps.csc_matrix(P)

        return P,q
    
    def Ab_gen(self,fos:list[list[float,None]],Nco) -> tuple[np.ndarray,np.ndarray]:
        """
        Generate the A and b matrices for the quadratic program.
        Args:
            fos:   List of flat outputs.
            Nsm:   Number of segments.
            Nco:   Number of control points.

        Returns:
            A:     Constraint matrix.
            b:     Constraint vector.
        """
        Nsm = len(fos)-1
        Nct = self.get_Nct(fos)

        A = np.zeros((Nct,(Nsm*Nco)))
        b = np.zeros((Nct,1))
        rct = 0
        for j in range(Nsm+1):
            for k,fo in enumerate(fos[j]):
                if fo is None:
                    # Continuity Constraint
                    Aeq,beq = self.compute_continuity(k,Nco)
                    c0,cf = (j-1)*Nco,(j+1)*Nco
                    r0,rf = rct,rct+1
                else:
                    # Equality Constraint
                    if j == 0:
                        Aeq,beq = self.compute_equality(fo,k,Nco,mode="initial")
                        c0,cf = j*Nco,(j+1)*Nco
                        r0,rf = rct,rct+1
                    elif j == Nsm:
                        Aeq,beq = self.compute_equality(fo,k,Nco,mode="final")
                        c0,cf = (j-1)*Nco,(j)*Nco
                        r0,rf = rct,rct+1
                    else:
                        Aeq0,beq0 = self.compute_equality(fo,k,Nco,mode="final")
                        Aeq1,beq1 = self.compute_equality(fo,k,Nco,mode="initial")
                        Aeq = spl.block_diag(Aeq0,Aeq1)
                        beq = np.vstack((beq0,beq1))
                        c0,cf = (j-1)*Nco,(j+1)*Nco
                        r0,rf = rct,rct+2

                # Fill in the constraint matrix
                A[r0:rf,c0:cf] = Aeq
                b[r0:rf] = beq
                rct = rf

        # Convert the matrix to sparse
        A = sps.csc_matrix(A)

        return A,b
    
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