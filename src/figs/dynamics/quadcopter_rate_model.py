from acados_template import AcadosModel
from casadi import SX,vertcat,horzcat

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

def extract_pixel_states(model: AcadosModel, params: dict) -> SX:
    """
    Extracts the pixel states from the full state vector.
    
    Args:
        x (SX): Full state vector.
        
    Returns:
        SX: Pixel states (px, py, pz).
    """
    # set up states (x)
    pt_x = SX.sym('pt_x')
    pt_y = SX.sym('pt_y')
    pt_z = SX.sym('pt_z')

    pt_w = vertcat(pt_x, pt_y, pt_z)

    # Extract some useful variables
    fx,fy,cx,cy = params['fx'], params['fy'], params['cx'], params['cy']
    Tb2c = params['Tb2c']

    pb_w = vertcat(model.x[0], model.x[1], model.x[2])
    vb_w = vertcat(model.x[3], model.x[4], model.x[5])
    qx,qy,qz,qw = model.x[6],model.x[7],model.x[8],model.x[9]
    wx,wy,wz = model.u[1],model.u[2],model.u[3]

    Rb2c = Tb2c[:3,:3]
    pb_c = Tb2c[0:3,3]

    # Compute rotation matrix from quaternion
    Rb2w = vertcat(
        horzcat(1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)),
        horzcat(2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)),
        horzcat(2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2))
    )
    Rw2b = Rb2w.T

    # Compute the skew-symmetric matrix
    Wss = vertcat(
        horzcat(0, -wz, wy),
        horzcat(wz, 0, -wx),
        horzcat(-wy, wx, 0)
    )

    # Compute the pixel states
    pt_b = Rw2b@(pt_w-pb_w)             # Target in body frame
    pt_c = Rb2c@(pt_b-pb_c)             # Target in camera frame
    vt_c = -Rb2c@(Rw2b@vb_w + Wss@pt_b) # Velocity in camera frame

    u = fx * pt_c[0] / pt_c[2] + cx
    v = fy * pt_c[1] / pt_c[2] + cy
    ud = fx*(vt_c[0]*pt_c[2]-pt_c[0]*vt_c[2]) / (pt_c[2]**2)
    vd = fy*(vt_c[1]*pt_c[2]-pt_c[1]*vt_c[2]) / (pt_c[2]**2)

    s = vertcat(u,v,ud,vd)
    
    return s,pt_w