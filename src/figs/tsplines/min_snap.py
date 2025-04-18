import numpy as np
import scipy.sparse as sps
import scipy.linalg as spl
import sys
import qpsolvers
import figs.utilities.transform_helper as th
import figs.utilities.polynomial_helper as ph

# Debugging
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)

class MinSnap():
    """
    Class for generating minimum snap trajectories.
    """

    def __init__(self, WPs:dict[str,int|tuple[np.float64,np.ndarray]],hz:int=20):
        """
        Initialize the class with waypoints and sampling frequency.

        Args:
            WPs:    Dictionary containing the course configuration.
            hz:     Sampling frequency.

        """
        
        # Some class constants
        Kdr = np.array([4,4,4,4])
        Ndr = 5
        
        Nfo,Nco = len(Kdr),WPs["Nco"]

        # Class constant variables
        self.Kdr,self.Ndr = Kdr,Ndr
        self.Nfo,self.Nco = Nfo,Nco
        self.hz = hz

        # Extract Flat Output Variables (Tp desired and FO desired)
        Tkf,FOkf = th.KF_to_TpFO(WPs["keyframes"],None)
        dT = np.diff(Tkf)

        # Get initial solution
        self.dTd,self.Pnd = self.solve(FOkf,dT)

    def solve(self,FOkf:np.ndarray,dT:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        """
        Solve the minimum time snap problem.

        Args:
            FOkf:   Flat output keyframes to pass through.
            dTkf:   Time intervals.

        Returns:
            dT:  Time intervals.
            Pn:  Polynomial coefficients.
        """

        # Unpack some stuff
        Nfo = self.Nfo
        Nsm = len(dT)

        # Some useful intermediate variables
        P = self.P_gen(dT)
        A,b = self.Ab_gen(dT,FOkf)

        # Solve QP to get coefficient solution (spline variables)
        x = qpsolvers.solve_qp(P,q=None,G=None,h=None,A=A,b=b,solver="osqp")
        Pn = x.reshape((Nfo,Nsm,-1))

        return dT,Pn
    
    def get_ideal(self,hz:int|None) -> tuple[np.ndarray,np.ndarray]:
        """
        Get the desired time and flat output values. If hz is None,
        return the keyframe values.

        Returns:
            Ts:  Time values.
            FO:  Flat output values.
        """

        # Unpack some stuff
        dTd,Pnd = self.dTd,self.Pnd
        Ndr = self.Ndr
        
        # Generate the time and flat output values
        if hz is None:
            Ts = np.hstack((0.0,np.cumsum(dTd)))
        else:
            tf = np.sum(dTd)
            Ts = np.arange(0.0,tf,1/hz)

        FO = th.dTPn_to_FO(Ts,dTd,Pnd,Ndr)

        return Ts,FO
    
    def P_gen(self,dT:list[float],use_sparse:bool=True) -> tuple[np.ndarray,np.ndarray]:
        """
        Generates the cost matrix for the minimum snap problem.

        Args:
            dT:         Time intervals.
            use_sparse: Use space or time cost matrix.

        Returns:
            P:  Cost matrix.
        """

        # Unpack some stuff
        Nfo,Nco = self.Nfo,self.Nco
        Kdr = self.Kdr

        # Generate the cost matrix pieces
        Ps = []
        for i in range(Nfo):
            Pi = ph.generate_Q(dT,Kdr[i],Nco)
            Ps.append(Pi)

        # Assemble the cost matrix
        P = spl.block_diag(*Ps)

        # Convert the matrix to sparse
        if use_sparse:
            P = sps.csc_matrix(P)

        return P

    def Ab_gen(self,dT:list[float],FO:list[list[float]],
               use_sparse:bool=True) -> tuple[sps.csc_matrix|np.ndarray,np.ndarray]:
        """
        Generates the A and b matrices for the minimum snap problem.

        Args:
            dT:     Time intervals.
            FOp:    Flat output keyframes to pass through.
            Nfo:    Number of flat outputs.
            Nco:    Number of coefficients.

        Returns:
            A:      A matrix.
            b:      b vector.
        """

        # Unpack some stuff
        Nfo,Ndr = self.Nfo,self.Ndr
        Nco = self.Nco
        Nsm = len(dT)

        # Generate continuity and fixed A matrices
        As,bs = [],[]
        for i in range(Nfo):
            Ais,bis = [],[]
            Ncn = 0
            for j in range(Nsm):
                # Extract the flat outputs
                fo0,fo1 = FO[j][i],FO[j+1][i]

                # Generate the stagewise boundary A matrices
                A0,A1 = ph.generate_As([0.0,dT[j]],dT[j],Nco,Ndr)

                if j < Nsm-1:
                    An0 = np.zeros((Ndr,Nco))
                    Anc = ph.generate_As([0.0],dT[j+1],Nco,Ndr)[0]
                else:
                    An0 = np.zeros((Ndr,0))
                    Anc = np.zeros((Ndr,0))

                Aa = np.hstack((A0, An0))
                Ab = np.hstack((A1, An0))
                Ac = np.hstack((A1,-Anc))
                
                # Populate
                Ajs,bjs = [],[]
                for k,fo in enumerate(fo0):
                    if np.isnan(fo):
                        pass
                    else:
                        Ajs.append(Aa[k,:])
                        bjs.append(fo)
                
                for k,fo in enumerate(fo1):
                    if np.isnan(fo):
                        Ajs.append(Ac[k,:])
                        bjs.append(0.0)
                    else:
                        Ajs.append(Ab[k,:])
                        bjs.append(fo)
                        
                # Stack the A and b matrices for this segment
                Aj = np.vstack(Ajs)
                bj = np.vstack(bjs)

                # Add to the segment list
                Ais.append(Aj)
                bis.append(bj)

                # Update the number of continuity constraints
                Ncn += len(bj)
            
            # Stack the A and b matrices for this flat output
            Ai,r0 = np.zeros((Ncn,Nco*Nsm)),0
            for i in range(len(Ais)):
                r1 = r0+Ais[i].shape[0]
                c0,c1 = i*Nco,Ais[i].shape[1]+i*Nco
                Ai[r0:r1,c0:c1] = Ais[i]

                r0 = r1
            bi = np.vstack(bis)

            # Add to the list of A and b matrices
            As.append(Ai)
            bs.append(bi)

        # Stack the A and b matrices
        A = spl.block_diag(*As)
        b = np.vstack(bs)

        # Convert the matrix to sparse
        if use_sparse:
            A = sps.csc_matrix(A)

        return A,b
