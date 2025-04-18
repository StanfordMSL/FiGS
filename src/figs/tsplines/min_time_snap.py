import numpy as np
import scipy.sparse as sps
import scipy.linalg as spl
from scipy.optimize import minimize
import sys
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
                 kT:float,hz:int):
        """
        Initialize the class with waypoints and sampling frequency.

        Args:
            WPs:    Dictionary containing the course configuration.
            kT:     Minimum time weight.
            hz:     Sampling frequency.
        """

        # Some class constants
        Kdr = np.array([4,4,4,4])
        Tau = np.array([0.0,1.0])
        Ndr = 5
        bnds = [(0.05, 10.0)]

        Nfo,Ncd = len(Kdr),len(Tau)

        # Class constant variables
        self.Kdr,self.Ndr = Kdr,Ndr
        self.Nfo,self.Ncd = Nfo,Ncd
        self.kT,self.bnds = kT,bnds
        self.hz = hz

        # Extract Flat Output Variables (Tp guess and FO desired)
        Tkf0,FOkf0 = th.KF_to_TpFO(WPs["keyframes"],Ndr)
        dT0 = np.diff(Tkf0)

        # Get initial solution
        self.dTd,self.Pnd = self.solve(FOkf0,dT0)

    def solve(self,FOkf:np.ndarray,dT0:np.ndarray=None) -> tuple[np.ndarray,np.ndarray]:
        """
        Solve the minimum time snap problem.

        Args:
            FOkf:   Flat output keyframes to pass through.
            dT0:    Initial guess for time intervals.

        Returns:
            dT:  Time intervals.
            Pn:  Polynomial coefficients.
        """

        # Unpack some stuff
        kT = self.kT
        Nfo = self.Nfo

        # Generate the time intervals if not provided
        if dT0 is None:
            Nsm = FOkf.shape[0]-1
            dt0 = self.bnds[1]/2
            dT0 = dt0*np.ones(Nsm)

        # Some useful constants
        Nsm = len(dT0)
        Bnds = self.bnds*Nsm

        # Solve the Time Snap QP
        res = minimize(lambda dT: self.time_snap_cost(dT,FOkf,kT),
                            x0=dT0,bounds=Bnds,method='SLSQP',
                            options={'maxiter': 100}
                            )
        
        # Package Results
        dT = res.x
        x = self.solve_uqp(dT,FOkf)[0]
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
            Ts = np.linspace(0.0,tf,int(tf*hz)+1)

        FO = th.dTPn_to_FO(Ts,dTd,Pnd,Ndr)

        return Ts,FO
        
    def time_snap_cost(self,dT:np.ndarray,FOkf:np.ndarray,kT:float) -> float:
        """
        Compute the cost function for the minimum time snap problem.

        Args:
            dT:     Time intervals.
            FOkf:   Flat output keyframes to pass through.
            kT:     Minimum time weight.

        Returns:
            J:   Cost function value.
        """

        # Solve the unconstrained quadratic program
        x,Q = self.solve_uqp(dT,FOkf)

        # Compute the cost
        J = x.T@Q@x + kT*sum(dT)

        return J
        
    def solve_uqp(self,dT:np.ndarray,FOkf:np.ndarray):
        """
        Solve the minimum snap problem (given fixed time) using unconstrained
        quadratic programming.
        
        Args:
            dT:     Time intervals.
            FOkf:   Flat output keyframes to pass through.

        Returns:
            x:  Coefficients of the polynomial.
            Q:  Quadratic cost matrix.
        """

        # Compute the Q and A matrices and the b vector
        Q = self.build_Q(dT)
        A,b = self.build_Ab(dT,FOkf)

        # Compute the extensions
        iA = ph.get_inverse(A)
        C,df = self.build_Cdf(b)

        # Solve the quadratic program
        if df.shape[0] != C.shape[0]:
            # Compute R
            R = C@iA.T@Q@iA@C.T

            # Extract Rfp and Rpp
            ndf = df.shape[0]
            Rfp = R[:ndf,ndf:]
            Rpp = R[ndf:,ndf:]

            # Solve the quadratic program
            dp = -ph.get_inverse(Rpp)@Rfp.T@df
            d = np.vstack((df,dp))
        else:
            d = df

        # Extract the coefficients
        x = iA@C.T@d

        return x,Q

    def build_Q(self,dT:np.ndarray,use_sparse:bool=True) -> sps.csc_matrix|np.ndarray:
        """
        Generate the cost matrix for the quadratic program.

        Args:
            dT:         Time intervals.
            use_sparse: Use sparse matrix format.

        Returns:
            Q:  Cost matrix.
        """

        # Unpack some stuff
        Nfo,Ndr = self.Nfo,self.Ndr
        Ncd,Kdr = self.Ncd,self.Kdr

        # Generate the cost matrix pieces
        Qs = []
        for i in range(Nfo):
            kdr = Kdr[i]
            Nco = Ndr*Ncd
        
            Qi = ph.generate_Q(dT,kdr,Nco)
            Qs.append(Qi)

        # Assemble the cost matrix
        Q = spl.block_diag(*Qs)

        # Convert the matrix to sparse
        if use_sparse:
            Q = sps.csc_matrix(Q)

        return Q
    
    def build_Ab(self,dT:np.ndarray,FO:np.ndarray,
                 use_sparse:bool=True) -> tuple[sps.csc_matrix|np.ndarray,np.ndarray]:
        """
        Generate the A matrix and b vector for the quadratic program.

        Args:
            dT:         Time intervals.
            FO:         Flat outputs
            use_sparse: Use sparse matrix format.

        Returns:
            A:  Constraint matrix.
            b:  Constraint vector.
        """
        # Unpack some stuff
        Nfo,Ndr = self.Nfo,self.Ndr
        Ncd = self.Ncd

        Nsm = len(dT)
        Nco = Ndr*Ncd

        # Generate continuity and fixed A matrices
        As,bs = [],[]
        for i in range(Nfo):
            # Generate continuity and fixed A matrices
            Abd = np.zeros(((Nsm+1)*Ndr,Nsm*Nco))
            Acn = np.zeros(((Nsm-1)*Ndr,Nsm*Nco))
            for j in range(Nsm):
                # Calculate the indices
                r0,r1 = j*Ndr,(j+1)*Ndr
                c0,c1cn,c1bd = j*Nco,(j+2)*Nco,(j+1)*Nco

                # Generate the stagewise boundary A matrices
                A0,A1 = ph.generate_As([0.0,dT[j]],dT[j],Nco,Ndr)

                # Populate the matrices
                Abd[r0:r1,c0:c1bd] = A0
                if j < Nsm-1:
                    A2 = ph.generate_As([0.0],dT[j+1],Nco,Ndr)[0]
                    Acn[r0:r1,c0:c1cn] = np.hstack((-A1,A2))
                
            Abd[-Ndr:,-Nco:] = A1           # Last row

            # Generate the continuity and fixed b vectors
            bbd = FO[:,i,:].reshape((-1,1))
            bcn = np.zeros(((Nsm-1)*Ndr,1))

            # Stack the A and b matrices
            Ai = np.vstack((Abd,Acn))
            bi = np.vstack((bbd,bcn))

            As.append(Ai),bs.append(bi)

        A = spl.block_diag(*As)
        b = np.vstack(bs)

        # Convert the matrix to sparse
        if use_sparse:
            A = sps.csc_matrix(A)

        return A,b

    def build_Cdf(self,b:np.ndarray,use_sparse:bool=True) -> tuple[sps.csc_matrix|np.ndarray,np.ndarray]:
        """
        Generate the C matrix and the df vector for the UQP

        Args:
            b:          Constraint vector.
            use_sparse: Use sparse matrix format.

        Returns:
            C:  Constraint mapping matrix.
            df: Known values vector.
        """

        # Generate initial C matrix
        Ci = np.eye(b.shape[0])

        con_mask = np.isnan(b)
        idx_f = np.where(~con_mask)[0]
        idx_p = np.where(con_mask)[0]
        Cf,Cp = Ci[idx_f,:],Ci[idx_p,:]

        # Generate the C matrix
        C = np.vstack((Cf,Cp))

        # Generate the df vector
        df = b[idx_f]

        # Convert the matrix to sparse
        if use_sparse:
            C = sps.csc_matrix(C)

        return C,df