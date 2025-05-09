"""
Helper functions for polynomials.
"""

import numpy as np
import scipy.sparse as sps
import scipy.linalg as spl

from scipy.sparse.linalg import inv, LinearOperator
from numpy.polynomial.legendre import Legendre,leggauss

def generate_Q(dTp:np.ndarray,kdr:int,Nco:int,
               Ngqp:int=50) -> np.ndarray|sps.csc_matrix:
        """
        Generates the Q matrix for minimum kdr derivative integral cost.

        Args:
            dTp:        Time points
            kdr:        Order of the derivative.
            Nco:        Number of coefficients.
            Ngqp:       Number of Gauss quadrature points.

        Returns:
            Q:          Q cost matrix.
        """

        # Some useful constants
        Nsm = len(dTp)
        tau_vals, weights = leggauss(Ngqp)

        # Precompute polynomial derivatives at tau_vals
        P_derivs = [Legendre.basis(j).deriv(kdr)(tau_vals) for j in range(Nco)]

        # Generate the Q matrix for each segment
        Qs = []
        for i in range(Nsm):
            dt = dTp[i]
            dtau_dt = 2/dt

            Qi = np.zeros((Nco,Nco))
            for j in range(Nco):
                for k in range(j,Nco):
                    integrand_vals = P_derivs[j] * P_derivs[k]
                    val = np.dot(weights, integrand_vals)
                    Qi[j, k] = Qi[k, j] = val * (dtau_dt ** (2 * kdr))
                    
            Qs.append(Qi)

        # Combine into a single matrix
        Q = spl.block_diag(*Qs)

        return Q

def generate_As(Tsm:list|np.ndarray,dtp:float,Nco:int,
                Ndr:int=1) -> dict[float,np.ndarray|sps.csc_matrix]:
    """
    Generates the A matrices for polynomial constraints.

    Args:
        Tsm:        Times within segment.
        dtp:        Time step.
        Nco:        Number of coefficients.
        Ndr:        Number of derivatives.

    Returns:
        As: List of A matrices.
    """

    # Check if Tsm is a list or numpy array
    if isinstance(Tsm, list):
        Tsm = np.array(Tsm)
    elif not isinstance(Tsm, np.ndarray):
        raise ValueError("Input Tsm must be a list or numpy array.")
    
    # Get normalized time points
    Tau = normalize_time(Tsm, 0.0, dtp)
    dtau_dt = 2/dtp

    # Precompute Legendre polynomials
    Pls = [Legendre.basis(j) for j in range(Nco)]

    # Generate dictionary of A matrices
    As = []
    for tau in Tau:
        A = np.zeros((Ndr,Nco))
        for i in range(Ndr):
            factor = dtau_dt ** i
            for j, Pl in enumerate(Pls):
                A[i, j] = factor * Pl.deriv(i)(tau)

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

def normalize_time(t:np.ndarray, t0:float, tf:float):
    """
    Normalize time to the range [-1, 1] based on the start and end times.
    Args:
        t:   Current time.
        t0:  Start time.
        tf:  End time.

    Returns:
        t:   Normalized time.
    """
    t = 2*(t-t0)/(tf-t0) - 1

    return t

def get_segment_vars(ti:float,dT:np.ndarray) -> int:
    """
    Get the segment current time and total time.

    Args:
        ti: Input time.
        dT: Time intervals.

    Returns:
        idx:    Index of the segment.
        tsm:    Segment time.
        dt:     Total segment time.
    """

    # Get index
    Tp = np.hstack([0.0,np.cumsum(dT)])
    idx = np.where(Tp <= ti)[0][-1]
    
    # Bound the index
    idx = np.clip(idx, 0, len(dT)-1)
    
    # Get the segment time amd total time
    tsm = ti-Tp[idx]
    dt = dT[idx]

    return idx,tsm,dt
