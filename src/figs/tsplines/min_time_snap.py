import numpy as np
import scipy.sparse as sps
import scipy.linalg as spl
from scipy.optimize import minimize
import sys
import math
import qpsolvers
import figs.utilities.polynomial_helper as ph
import figs.utilities.transform_helper as th
from figs.dynamics.external_forces import ExternalForces

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

        Nfo,Ncd = len(Kdr),len(Tau)

        # Extract Flat Output Variables
        Tp,FO = th.KF_to_TpFO(WPs["keyframes"],Ndr,impose=True)

        # Compute the Q and A matrices and the b vector
        Qs,As,bs = [],[],[]
        for i in range(Nfo):
            kdr = Kdr[i]
            Nco = Ndr*Ncd
        
            Qi = self.build_Q(Tp,kdr,Nco)
            Ai,bi = self.build_Ab(Tp,Ndr,Nco,FO[:,i,:])

            Qs.append(Qi)
            As.append(Ai),bs.append(bi)

        Q = sps.block_diag(Qs,format='csc')
        A = sps.block_diag(As,format='csc')
        b = np.vstack(bs)
        iA = ph.get_inverse(A)
        C,df = self.build_Cdf(b)

        # Save class variables
        self.Nfo = Nfo
        self.Tp = Tp
        self.Q = Q
        self.iA,self.C,self.df = iA,C,df

        self.A,self.b = A,b
        self.hz = hz

    def solve(self):
        """
        Solve the minimum time snap problem using quadratic programming.

        Returns:
            Tp:     Time points.
            Pn:     Flat output array.
        """

        # Unpack some stuff
        Tp = self.Tp
        Q = self.Q
        iA,C,df = self.iA,self.C,self.df
        Nfo = self.Nfo
        Nsm = len(Tp)-1
        
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

        p = iA@C.T@d

        # Package
        Pn = p.reshape((Nfo,Nsm,-1))

        return Tp,Pn

    def build_Q(self,Tp:np.ndarray,kdr:int,Nco:int):
        """
        Generate the Q matrix for the quadratic program.

        Args:
            Tp:  Time points.
            kdr: Number of derivatives.
            Nco: Number of coefficients.

        Returns:
            Q:  Quadratic matrix.
        """

        # Generate the Q matrix
        Q = ph.generate_Q(Tp,kdr,Nco)

        return Q
    
    def build_Ab(self,Tp:np.ndarray,Ndr:int,Nco:int,FOi:np.ndarray,
                 use_sparse:bool=True) -> tuple[sps.csc_matrix|np.ndarray,np.ndarray]:
        """
        Generate the A inverse matrix for the quadratic program.

        Args:
            Tp:         Time points.
            Ndr:        Number of the derivatives.
            Nco:        Number of coefficients.
            FOi:        Array of derivatives of a given flat output
            use_sparse: Use sparse matrix format.

        Returns:
            A:  Constraint matrix.
            b:  Constraint vector.
        """

        # Unpack some stuff
        Nsm = len(Tp)-1
        
        # Generate continuity and fixed A matrices
        Abd = np.zeros(((Nsm+1)*Ndr,Nsm*Nco))
        Acn = np.zeros(((Nsm-1)*Ndr,Nsm*Nco))
        for i in range(Nsm):
            # Calculate the indices
            r0,r1 = i*Ndr,(i+1)*Ndr
            c0,c1cn,c1bd = i*Nco,(i+2)*Nco,(i+1)*Nco

            # Generate the stagewise boundary A matrices
            t0,t1 = Tp[i],Tp[i+1]
            A0,A1 = ph.generate_As([t0,t1],t0,t1,Nco,Ndr)

            # Populate the matrices
            Abd[r0:r1,c0:c1bd] = A0
            if i < Nsm-1:
                A2 = ph.generate_As([t1],t1,Tp[i+2],Nco,Ndr)[0]
                Acn[r0:r1,c0:c1cn] = np.hstack((-A1,A2))
            
        Abd[-Ndr:,-Nco:] = A1           # Last row

        # Generate the continuity and fixed b vectors
        bbd = FOi.reshape((-1,1))
        bcn = np.zeros(((Nsm-1)*Ndr,1))

        # Stack the A and b matrices
        A = np.vstack((Abd,Acn))
        b = np.vstack((bbd,bcn))

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