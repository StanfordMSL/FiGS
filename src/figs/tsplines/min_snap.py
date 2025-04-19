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

    def __init__(self,
                 WPs:dict[str,int|tuple[np.float64,np.ndarray]],
                 hz:int,
                 Kdr:np.ndarray=np.array([4,4,4,2])) -> None:
        """
        Initialize the class with waypoints and sampling frequency.

        Args:
            WPs:    Dictionary containing the course configuration.
            hz:     Sampling frequency.
            Kdr:    Derivative order for each flat output cost.
        """
        
        # Some useful constants
        Ndr = np.max(Kdr)+1
        Nfo,Nco = len(Kdr),WPs["Nco"]

        # Extract Flat Output Variables (Tp desired and FO desired)
        Tkf,FOkf = th.KF_to_TpFO(WPs["keyframes"],None)
        dT = np.diff(Tkf)

        # Class compute variables
        self.Kdr,self.Ndr = Kdr,Ndr
        self.Nfo,self.Nco = Nfo,Nco

        # Compute initial solution
        dTd,Pnd = self.solve(FOkf,dT)
        Ts,FO = th.dTPn_to_TsFO(dTd,Pnd,Ndr,hz)

        # Class trajectory variables
        self.dTd,self.Pnd = dTd,Pnd
        self.Tkf,self.FOkf = Tkf,FOkf
        self.Tsd,self.FOd = Ts,FO
    
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

    def get_desired_trajectory(self) -> tuple[np.ndarray,np.ndarray]:
        """
        Get the time and flat output trajectory values.

        Returns:
            Ts:  Time values.
            FO:  Flat output values.
        """

        # Return the time and flat output values
        return self.Tsd,self.FOd
    
    def get_velocity_statistics(self,FO:np.ndarray=None) -> tuple[float,float,float]:
        """
        Get the velocity statistics.

        Returns:
            v_mean:  Mean velocity.
            v_std:   Standard deviation of velocity.
            v_max:   Maximum velocity.
        """
        
        # Unpack some stuff
        if FO is None:
            FO = self.FOd

        # Compute velocity statistics
        Vmag = np.linalg.norm(FO[:,0:3,1],axis=1)
        v_mean = np.mean(Vmag)
        v_std = np.std(Vmag)
        v_max = np.max(Vmag)
        
        # Return the velocity statistics
        return v_mean,v_std,v_max