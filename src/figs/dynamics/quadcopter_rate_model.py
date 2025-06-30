from acados_template import AcadosModel
from casadi import SX,vertcat

def export_model() -> AcadosModel:

    model_name = 'quadcopter_full_model'

    # set up states (x)
    px = SX.sym('px')
    py = SX.sym('py')
    pz = SX.sym('pz')

    vx = SX.sym('vx')
    vy = SX.sym('vy')
    vz = SX.sym('vz')

    qx = SX.sym('qx')
    qy = SX.sym('qy')
    qz = SX.sym('qz')
    qw = SX.sym('qw')

    x = vertcat(px,py,pz,vx,vy,vz,qx,qy,qz,qw)

    # set up controls (u)
    uf = SX.sym('uf')
    wx = SX.sym('wx')
    wy = SX.sym('wy')
    wz = SX.sym('wz')
    
    u = vertcat(uf,wx,wy,wz)
    
    # set up parameters (p)
    m = SX.sym('m')
    kt = SX.sym('kt')
    fx = SX.sym('fx')
    fy = SX.sym('fy')
    fz = SX.sym('fz')
    
    p = vertcat(m,kt,fx,fy,fz)

    # xdot
    px_dot = SX.sym('px_dot')
    py_dot = SX.sym('py_dot')
    pz_dot = SX.sym('pz_dot')
    p_dot = vertcat(px_dot,py_dot,pz_dot)

    vx_dot = SX.sym('vx_dot')
    vy_dot = SX.sym('vy_dot')
    vz_dot = SX.sym('vz_dot')
    v_dot = vertcat(vx_dot,vy_dot,vz_dot)

    qx_dot = SX.sym('qx_dot')
    qy_dot = SX.sym('qy_dot')
    qz_dot = SX.sym('qz_dot')
    qw_dot = SX.sym('qw_dot')
    q_dot = vertcat(qx_dot,qy_dot,qz_dot,qw_dot)

    xdot = vertcat(p_dot,v_dot,q_dot)

    # some intermediate variables
    V1a = vertcat(0.0, 0.0, 9.81)
    V1b = (4*kt*uf/m)*vertcat(
            2.0*(qx*qz + qy*qw),
            2.0*(qy*qz - qx*qw),
            qw*qw - qx*qx - qy*qy + qz*qz
            )
    V1c = (1/m)*vertcat(
            fx,
            fy,
            fz
            )
    
    V2 = (1/2)*vertcat(
         qw*wx - qz*wy + qy*wz,
         qz*wx + qw*wy - qx*wz,
        -qy*wx + qx*wy + qw*wz,
        -qx*wx - qy*wy - qz*wz)

    # dynamics
    f_expl = vertcat(
                vx,vy,vz,
                V1a + V1b + V1c,
                V2
                )
    
    f_impl = xdot-f_expl

    # Pack into acados model
    model = AcadosModel()

    model.x = x
    model.u = u
    model.p = p
    model.xdot = xdot
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl

    model.name = model_name

    return model

