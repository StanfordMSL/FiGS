"""
Helper functions for polynomials.
"""

import numpy as np
import math

from numpy.polynomial.legendre import Legendre,legval
from scipy.integrate import quad

def get_M(Ncp:int) -> np.ndarray:
    """
    Generates the M matrix for polynomial interpolation.

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
    M = np.linalg.inv(M).T

    return M

def get_nt(rt:float,ndr:int,Ncp:int) -> np.ndarray:  
    """
    Generates the normalized time vector based on desired derivative order.

    Args:
        rt:   Segment time ratio
        ndr:  Derivative order.
        Ncp:  Number of control points.

    Returns:
        nt:   Normalized time vector.
    """

    nt = np.zeros(Ncp)
    for i in range(ndr,Ncp):
        c = math.factorial(i)/math.factorial(i-ndr)
        nt[i] = c*rt**(i-ndr)
    
    return nt

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

def get_legendre_integral(dt:float,kdr:int,Nco:int) -> np.ndarray:
    """
    Generate the base integral cost (integral of the square of the p-th
    derivative).

    Args:
        dt:   Time step.
        kdr:  Order of the derivative.
        Nco:  Number of control points.

    Returns:
        Q:     Base integral cost matrix.
    """

    dtau_dt = 2/dt

    Q = np.zeros((Nco,Nco))
    for i in range(Nco):
        Pi = Legendre.basis(i).deriv(kdr)
        for j in range(i,Nco):
            Pj = Legendre.basis(j).deriv(kdr)
            integrand = lambda tau: Pi(tau) * Pj(tau)
            val, _ = quad(integrand, -1, 1)
            Q[i, j] = Q[j, i] = val * (dtau_dt**(2*kdr))

    return Q

def get_legendre_vector(tau:float,dt:float,kdr:int,Nco:int) -> np.ndarray:
    """
    Generate the Legendre polynomial value at time t.

    Args:
        tau:    Normalized time.
        dt:     Segment time step.
        kdr:    Order of the derivative.
        Nco:    Number of control points.

    Returns:
        A:      Legendre vector.
    """

    dtau_dt = 2/dt

    A = np.zeros((1,Nco))
    for i in range(Nco):
        P = Legendre.basis(i).deriv(kdr)
        A[0,i] = P(tau)*(dtau_dt**kdr)

    return A

def get_control_points_map(t0:float, tf:float, Nco:int) -> np.ndarray:
    """
    Generate the projection matrix that maps Legendre coefficients to
    control points.

    Args:
        t0:     Start time.
        tf:     End time.
        Nco:    Number of control points.

    Returns:
        M:     Projection matrix (coefficients to control points).
    """
    # Generate time vector
    T = np.linspace(t0, tf, Nco)

    # Normalize times to [-1, 1]
    tau = 2 * (T - t0) / (tf - t0) - 1

    # Vandermonde-style matrix using Legendre basis
    M = np.zeros((len(tau), Nco))
    for i, tau_i in enumerate(tau):
        coeffs = np.eye(Nco)  # Identity rows = individual basis functions
        M[i] = legval(tau_i, coeffs)

    return M