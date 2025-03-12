from acados_template import AcadosModel
from casadi import SX,vertcat

def export_base_ode_model(m:float,tn:float) -> AcadosModel:

    model_name = 'quadcopter_ode_model'

    # set up states
    px = SX.sym('px')
    py = SX.sym('py')
    pz = SX.sym('pz')
    p = vertcat(px,py,pz)

    vx = SX.sym('vx')
    vy = SX.sym('vy')
    vz = SX.sym('vz')
    v = vertcat(vx,vy,vz)

    qx = SX.sym('qx')
    qy = SX.sym('qy')
    qz = SX.sym('qz')
    qw = SX.sym('qw')
    q = vertcat(qx,qy,qz,qw)

    x = vertcat(p,v,q)

    # set up controls
    uf = SX.sym('uf')

    wx = SX.sym('wx')
    wy = SX.sym('wy')
    wz = SX.sym('wz')
    uw = vertcat(wx,wy,wz)

    u = vertcat(uf,uw)

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
    V1b = (tn*uf/m)*vertcat(
            2.0*(qx*qz + qy*qw),
            2.0*(qy*qz - qx*qw),
            qw*qw - qx*qx - qy*qy + qz*qz
            )
    
    V2 = (1/2)*vertcat(
         qw*wx - qz*wy + qy*wz,
         qz*wx + qw*wy - qx*wz,
        -qy*wx + qx*wy + qw*wz,
        -qx*wx - qy*wy - qz*wz)

    # dynamics
    f_expl = vertcat(
                v,
                V1a + V1b,
                V2
                )
    
    f_impl = xdot-f_expl

    # Pack into acados model
    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    return model

