import numpy as np
import scipy.sparse as sps
import scipy.linalg as spl
import sys
import math
import qpsolvers
import figs.utilities.polynomial_helper as ph
import figs.utilities.transform_helper as th

# Debugging
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)

# Fixed parameters for minimum snap quadcopter trajectory planning problem
Kdr = np.array([4,4,4,2])                                       # Target derivative to minimize
mus = np.array([1.0,1.0,1.0,1.0])                               # Scaling for each parameter
Nfo = len(Kdr)                                                  # Number of Flat Outputs

def solve(WPs:dict[str,int|tuple[np.float64,np.ndarray]],hz:int=20,
          Natt=5) -> dict[str,tuple[np.ndarray,np.ndarray]]:
    """
    Solve the minimum snap trajectory planning problem.

    Args:
        WPs:    Dictionary containing the course configuration.
        hz:     Sampling frequency.
        Natt:   Number of attempts to solve the QP problem.

    Returns:
        output: Dictionary containing the solution in its various forms.

    """

    # Extract Flat Output Parameters
    KFs:dict = WPs["keyframes"]
    Nco = WPs["Nco"]

    Ts,FOs, = [],[]
    for keyframe in KFs.values():
        Ts.append(keyframe["t"])
        FOs.append(keyframe["fo"])
    Nsm = len(Ts)-1

    # # Normalize Time
    # Ts = [t/Ts[-1] for t in Ts]

    # Construct Mapping matrix
    Vs = []
    for i in range(Nfo):
        for j in range(Nsm):
            Mj = ph.get_control_points_map(Ts[j],Ts[j+1],Nco)
            Vs.append(Mj)

    V = sps.block_diag(Vs)

    x,P,A,b = solve_inner_qp(Ts,FOs,Nco,Kdr,mus)

    Tps = np.array(Ts)
    CPs = np.array(V@x).reshape((Nfo,Nsm,Nco))
    CPs = np.transpose(CPs,(1,0,2))
    Tss,FOs = th.TpCP_to_TsFO(Tps,CPs,hz) 
    
    # Package Output
    output = {
        "QP": {"P":P,"A":A,"b":b,"x":x},
        "CP": (Tps,CPs),
        "FO": (Tss,FOs),
    }

    cost = x.T@P@x
    print(f"Cost: {cost:.3f}")

    return output

def solve_inner_qp(Ts,FOs,Nco,Kdr,mus):
    # Solve Min Snap for each Flat Output
    Ps,As,bs = [],[],[]
    for i in range(Nfo):
        # Unpack some stuff
        kdr = Kdr[i]
        fos = [FO[i] for FO in FOs]

        # Compute cost term
        uPf = P_gen(Ts,kdr,Nco)         # Unweighted
        Pf = mus[i]*uPf                 # Weighted

        # Compute constraints terms
        Af,bf = Ab_gen(Ts,fos,Nco)

        # Append the terms
        Ps.append(Pf),As.append(Af),bs.append(bf)

    # Solve the QP
    P = sps.block_diag(Ps, format='csc')
    A,b = sps.block_diag(As, format='csc'),np.vstack(bs)

    sigma = qpsolvers.solve_qp(P,q=None,G=None,h=None,A=A,b=b,solver="osqp")

    return sigma,P,A,b

def P_gen(Ts:list[float], kdr:int, Nco:int,
          use_sparse:bool=True) -> tuple[np.ndarray,np.ndarray]:
    """
    Generate the P matrix for the quadratic program.

    Args:
        Ts:         List of time points.
        kdr:        Order of the derivative.
        Nco:        Number of coeffeciants.
        use_sparse: Use sparse matrix format.

    Returns:
        P:     Quadratic term.
    """

    Nsm = len(Ts)-1
    P = np.zeros((Nsm*Nco,Nsm*Nco))

    for j in range(Nsm):
        idx0,idxf = j*Nco,(j+1)*Nco  
        Tj = Ts[j+1]-Ts[j]
        Pj = ph.get_legendre_integral(Tj,kdr,Nco)

        P[idx0:idxf,idx0:idxf] = Pj

    # Convert the matrix to sparse
    if use_sparse:
        P = sps.csc_matrix(P)

    return P

def Ab_gen(Ts:list[float],fos:list[list[float,None]],Nco:int,
           use_sparse:bool=True) -> tuple[np.ndarray,np.ndarray]:
    """
    Generate the A and b matrices for the quadratic program.
    Args:
        Ts:         List of time points.
        fos:        List of flat outputs.
        Nsm:        Number of segments.
        Nco:        Number of control points.
        use_sparse: Use sparse matrix format.

    Returns:
        A:     Constraint matrix.
        b:     Constraint vector.
    """

    # Unpack some stuff
    Nsm = len(fos)-1
    Nct = get_Nct(fos)

    # Fill in the constraint matrix
    A,b,kct = np.zeros((Nct,(Nco*Nsm))),np.zeros((Nct,1)),0
    for ksm in range(Nsm):
        # Unpack some stuff
        Tk = Ts[ksm+1]-Ts[ksm]
        fok,fon = fos[ksm],fos[ksm+1]
        Npad = 0 if ksm == Nsm-1 else 1

        # Compute beginning constraints
        Asm,bsm = [],[]
        for kdr,val in enumerate(fok):
            if val is not None:
                A0 = ph.get_legendre_vector(-1.0,Tk,kdr,Nco)
                A1 = np.zeros((1,Nco*Npad))

                Ai,bi = np.hstack((A0,A1)),np.array([val])
            else:
                Ai,bi = np.zeros((0,Nco*(Npad+1))),np.zeros((0,1))

            Asm.append(Ai),bsm.append(bi)
        
        # Compute end constraints
        for kdr,val in enumerate(fon):
            if val is not None:
                A0 = ph.get_legendre_vector(1.0,Tk,kdr,Nco)
                A1 = np.zeros((1,Nco*Npad))

                Ai,bi = np.hstack((A0,A1)),np.array([val])
            else:
                Tn = Ts[ksm+2]-Ts[ksm+1]
                A0 = ph.get_legendre_vector(1.0,Tk,kdr,Nco)
                A1 = ph.get_legendre_vector(-1.0,Tn,kdr,Nco)

                Ai,bi = np.hstack((A0,-A1)),np.array([0.0])

            Asm.append(Ai),bsm.append(bi)

        # Pack the constraints
        Asm,bsm = np.vstack(Asm),np.vstack(bsm)
        r0,r1 = kct,kct+Asm.shape[0]
        c0,c1 = ksm*Nco,(ksm+1+Npad)*Nco
        A[r0:r1,c0:c1],b[r0:r1,:] = Asm,bsm

        # Update the constraint index
        kct = r1

    # Convert the matrix to sparse
    if use_sparse:
        A = sps.csc_matrix(A)

    return A,b

def get_Nct(fos:list[list[float,None]]) -> int:
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