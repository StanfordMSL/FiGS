"""
Helper functions for orientations.
"""

import numpy as np

def obedient_quaternion(qcr:np.ndarray,qrf:np.ndarray) -> np.ndarray:
    """
    Ensure that the quaternion is well-behaved (unit norm and closest to reference).
    
    Args:
        - qcr:    Current quaternion.
        - qrf:    Previous quaternion.

    Returns:
        - qcr:     Closest quaternion to reference.
    """
    qcr = qcr/np.linalg.norm(qcr)

    if np.dot(qcr,qrf) < 0:
        qcr = -qcr

    return qcr