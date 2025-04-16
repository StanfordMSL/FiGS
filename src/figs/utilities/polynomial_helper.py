"""
Helper functions for polynomials.
"""

import numpy as np
import math
import scipy.sparse as sps
import scipy.linalg as spl

from scipy.sparse.linalg import inv, LinearOperator
from scipy.integrate import quad
from numpy.polynomial.legendre import Legendre,legval,legder

def generate_Q(Tp:np.ndarray,kdr:int,Nco:int,
               use_sparse:bool=True) -> np.ndarray|sps.csc_matrix:
        """
        Generates the Q matrix for minimum kdr derivative integral cost.

        Args:
            Tp:         Time points
            kdr:        Order of the derivative.
            Nco:        Number of coefficients.
            use_sparse: Use sparse matrix format.

        Returns:
            Q:          Q cost matrix.
        """

        # Extract the time intervals
        dTp = np.diff(Tp)

        # Some useful constants
        Nsm = len(dTp)

        # Generate the Q matrix for each segment
        Qs = []
        for i in range(Nsm):
            dt = Tp[i+1] - Tp[i]
            dtau_dt = 2/dt

            Qi = np.zeros((Nco,Nco))
            for j in range(Nco):
                Pj = Legendre.basis(j).deriv(kdr)
                for k in range(j,Nco):
                    Pk = Legendre.basis(k).deriv(kdr)
                    integrand = lambda tau: Pj(tau) * Pk(tau)
                    val, _ = quad(integrand, -1, 1)
                    Qi[j, k] = Qi[k, j] = val * (dtau_dt**(2*kdr))
                    
            Qs.append(Qi)

        # Combine into a single matrix
        Q = spl.block_diag(*Qs)

        # Convert the matrix to sparse
        if use_sparse:
            Q = sps.csc_matrix(Q)
            
        return Q

def generate_As(Tp:np.ndarray,t0:float,tf:float,Nco:int,Ndr:int=1,debug=False) -> dict[float,np.ndarray|sps.csc_matrix]:
    """
    Generates the A matrices for polynomial constraints.

    Args:
        Tp:         Time points
        t0:         Interval start time.
        tf:         Interval end time.
        Nco:        Number of coefficients.
        Ndr:        Number of derivatives.
        use_sparse: Use sparse matrix format.

    Returns:
        As: List of A matrices.
    """

    # Get normalized time points
    Tau = normalize_time(Tp, t0, tf)
    dtau_dt = 2/(tf-t0)

    # Generate dictionary of A matrices
    As = []
    for tau in Tau:
        A = np.zeros((Ndr,Nco))
        for i in range(Ndr):
            ctau = dtau_dt**i
            for j in range(Nco):
                Pj = Legendre.basis(j).deriv(i)
                A[i,j] = ctau * Pj(tau)

        As.append(A)

    return As

def get_inverse(M:np.ndarray|sps.csc_matrix) -> LinearOperator|np.ndarray:
    """
    Generates the inverse operator for the matrix A

    Args:
        M:          Input matrix.

    Returns:
        iM:         Operator for the inverse of the polynomial constraint matrix.
    """

    if isinstance(M,sps.csc_matrix):
        iM = inv(M)
    else:
        iM = np.linalg.inv(M)

    return iM

def generate_M(Ncp:int) -> np.ndarray:
    """
    Generates the M matrix for polynomial interpolation from control points.

    Args:
        - Ncp:    Number of control points.

    Returns:
        - M:      Polynomial interpolation matrix.
    """
    M = np.zeros((Ncp,Ncp))
    for i in range(Ncp):
        ci = (1/(Ncp-1))*i
        for j in range(Ncp):
            M[i,j] = ci**j
    M = np.linalg.inv(M)

    return M

def normalize_time(t, t0, tf):
    """
    Normalize time to the range [-1, 1] based on the start and end times.
    Args:
        t:   Current time.
        t0:  Start time.
        tf:  End time.
    Returns:
        t:   Normalized time.
    """
    return 2 * (t - t0) / (tf - t0) - 1

def get_segment_index(t:float,Tp:np.ndarray) -> int:
    """
    Get the segment index for a given time point.

    Args:
        t:          Time point.
        Tp:         Time points.

    Returns:
        i:          Segment index.
    """

    # Get indices
    idxs = np.where(Tp <= t)[0]

    # Boundary conditions
    if len(idxs) == 0:
        idx = 0
    else:
        idx = idxs[-1]
        if idx == len(Tp)-1:
            idx = len(Tp)-2

    return idx