"""
Helper functions for polynomials.
"""

import numpy as np
import math
import scipy.sparse as sps
import scipy.linalg as spl

from numpy.polynomial.legendre import Legendre,legval
from scipy.integrate import quad

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