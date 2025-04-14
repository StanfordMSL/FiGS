import numpy as np
import scipy.sparse as sps
import scipy.linalg as spl
from scipy.optimize import minimize
import sys
import math
import qpsolvers
import figs.utilities.polynomial_helper as ph
import figs.utilities.transform_helper as th

# Debugging
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)

class MinTimeSnap():
    """
    Class for generating minimum time snap trajectories.
    """

    def __init__(self, WPs:dict[str,int|tuple[np.float64,np.ndarray]],
                 kT:float,Kdr:np.ndarray,mus:np.ndarray,Nco:int,hz:int,
                 bnds = [(0.05, 10.0)]):
        """
        Initialize the class with waypoints and sampling frequency.

        Args:
            WPs:    Dictionary containing the course configuration.
            kT:     Minimum time weight.
            Kdr:    Order of the derivative in the integral cost.
            mus:    Weight for each flat output
            Nco:    Number of coefficients for each flat output.
            hz:     Sampling frequency.
            bnds:   (Piecewise) bounds for the time intervals.
        """

        # Extract Flat Output Parameters
        KFs:dict = WPs["keyframes"]
        Nco = WPs["Nco"]

        Ts,FOs, = [],[]
        for keyframe in KFs.values():
            Ts.append(keyframe["t"])
            FOs.append(keyframe["fo"])
        Nsm = len(Ts)-1
        Nfo = len(Kdr)

        # Generate time intervals
        dTs = [ (Ts[i+1]-Ts[i]) for i in range(Nsm) ]

        # Fixed Parameters
        self.kT = kT
        self.Kdr = Kdr
        self.mus = mus
        self.Nfo,self.Nco = Nfo,Nco
        self.Bnds = bnds * len(dTs)
        self.hz = hz

        # Generate Continuity and Equality Constraints
        A,b = self.Ab_gen(dTs,FOs)
        self.A,self.b = A,b
        self.dTs0 = dTs

    def solve(self):
        # Unpack some stuff
        dTs0 = self.dTs0
        A,b = self.A,self.b
        kT = self.kT
        Nfo,Nsm,Nco = self.Nfo,len(dTs0),self.Nco
        hz = self.hz

        # Solve the Time Snap QP
        res = minimize(lambda x: self.time_snap_cost(x,A,b,kT),
                            x0=dTs0,bounds=self.Bnds,method='SLSQP',
                            options={
                                'maxiter': 100,
                                'disp': True
                                }
                            )
        dTs = res.x
        Tps = np.zeros((len(dTs)+1))
        for i in range(len(dTs)):
            Tps[i+1] = Tps[i] + dTs[i]

        # Construct Mapping matrix
        Vs = []
        for i in range(Nfo):
            for j in range(Nsm):
                Mj = ph.get_control_points_map(Tps[j],Tps[j+1],Nco)
                Vs.append(Mj)
        V = sps.block_diag(Vs)

        # Get the Coefficient Solution
        P = self.P_gen(dTs)
        x = qpsolvers.solve_qp(P,q=None,G=None,h=None,A=A,b=b,solver="osqp")

        # Package Output        
        CPs = np.array(V@x).reshape((Nfo,Nsm,Nco))
        CPs = np.transpose(CPs,(1,0,2))
        Tss,FOs = th.TpCP_to_TsFO(Tps,CPs,hz) 
        
        # Package Output
        output = {
            "QP": {"P":P,"A":A,"b":b,"x":x},
            "CP": (Tps,CPs),
            "FO": (Tss,FOs),
        }

        print(sum(Tps),Tps)

        return output

    def time_snap_cost(self,dTs:list[float],A:np.ndarray,b:np.ndarray,kT:float) -> float:
        """
        Compute the total cost of the trajectory.

        Args:
            dTs:   Time intervals.

        Returns:
            total_cost:   Total cost of the trajectory.
        """

        # Solve Inner QP
        P = self.P_gen(dTs)
        x = qpsolvers.solve_qp(P,q=None,G=None,h=None,A=A,b=b,solver="osqp")

        # Penalize infeasible solutions
        if x is None:
            return 1e6

        # Compute the cost
        snap_cost = x.T@P@x
        time_cost = kT*sum(dTs)
        total_cost = snap_cost + time_cost

        return total_cost

    def P_gen(self,ndTs:list[float],use_sparse:bool=True) -> tuple[np.ndarray,np.ndarray]:
        """
        Generate the P matrix for the quadratic program.

        Args:
            ndTs:       Normalized time intervals.
            use_sparse: Use sparse matrix format.

        Returns:
            P:  Quadratic term.
        """
        
        # Unpack some stuff
        Nfo,Nco = self.Nfo,self.Nco
        Kdr,mus = self.Kdr,self.mus
        Nsm = len(ndTs)

        # Generate the P matrix
        P = np.zeros((Nfo*Nsm*Nco,Nfo*Nsm*Nco))
        for i in range(Nfo):
            # Unpack some stuff
            kdr = Kdr[i]
            mu = mus[i]

            # Compute the cost term
            for j in range(Nsm):
                idx0 = i*Nco*Nsm + j*Nco
                idxf = idx0 + Nco

                ndTj = ndTs[j]
                Pj = ph.get_legendre_integral(ndTj,kdr,Nco)

                P[idx0:idxf,idx0:idxf] = mu*Pj

        # Convert the matrix to sparse
        if use_sparse:
            P = sps.csc_matrix(P)

        return P

    def Ab_gen(self,dTs:list[float],FOs:list[list[float,None]],use_sparse:bool=True) -> tuple[np.ndarray,np.ndarray]:
        """
        Generate the A and b matrices for the quadratic program.
        Args:
            dTs:       List of time intervals.
            FOs:        List of flat outputs.
            use_sparse: Use sparse matrix format.

        Returns:
            A:     Constraint matrix.
            b:     Constraint vector.
        """

        # Unpack some stuff
        Nsm = len(FOs)-1
        Nfo,Nco = self.Nfo,self.Nco

        # Fill in the constraint matrix
        As,bs = [],[]
        for i in range(Nfo):
            fos = [FO[i] for FO in FOs]
            Nct = self.get_Nct(fos)

            Aeq,beq,kct = np.zeros((Nct,(Nco*Nsm))),np.zeros((Nct,1)),0
            for ksm in range(Nsm):
                # Unpack some stuff
                dTk = dTs[ksm]
                fok,fon = fos[ksm],fos[ksm+1]
                Npad = 0 if ksm == Nsm-1 else 1

                # Compute beginning constraints
                Asm,bsm = [],[]
                for kdr,val in enumerate(fok):
                    if val is not None:
                        A0 = ph.get_legendre_vector(-1.0,dTk,kdr,Nco)
                        A1 = np.zeros((1,Nco*Npad))

                        Ai,bi = np.hstack((A0,A1)),np.array([val])
                    else:
                        Ai,bi = np.zeros((0,Nco*(Npad+1))),np.zeros((0,1))

                    Asm.append(Ai),bsm.append(bi)
                
                # Compute end constraints
                for kdr,val in enumerate(fon):
                    if val is not None:
                        A0 = ph.get_legendre_vector(1.0,dTk,kdr,Nco)
                        A1 = np.zeros((1,Nco*Npad))

                        Ai,bi = np.hstack((A0,A1)),np.array([val])
                    else:
                        Tn = dTs[ksm+1]
                        A0 = ph.get_legendre_vector(1.0,dTk,kdr,Nco)
                        A1 = ph.get_legendre_vector(-1.0,Tn,kdr,Nco)

                        Ai,bi = np.hstack((A0,-A1)),np.array([0.0])

                    Asm.append(Ai),bsm.append(bi)

                # Pack the constraints
                Asm,bsm = np.vstack(Asm),np.vstack(bsm)
                r0,r1 = kct,kct+Asm.shape[0]
                c0,c1 = ksm*Nco,(ksm+1+Npad)*Nco
                Aeq[r0:r1,c0:c1],beq[r0:r1,:] = Asm,bsm

                # Update the constraint index
                kct = r1

            # Append the constraints
            As.append(Aeq),bs.append(beq)

        # Stack the constraints
        A,b = sps.block_diag(As),np.vstack(bs)

        # Convert the matrix to sparse
        if use_sparse:
            A = sps.csc_matrix(A)

        return A,b

    def get_Nct(self,fos:list[list[float,None]]) -> int:
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