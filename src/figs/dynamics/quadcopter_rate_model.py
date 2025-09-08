import numpy as np
from acados_template import AcadosModel
from casadi import SX,vertcat,horzcat,simplify,exp,norm_2

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
    V1b = (4*kt*uf/(m+1e-5))*vertcat(
            2.0*(qx*qz + qy*qw),
            2.0*(qy*qz - qx*qw),
            qw*qw - qx*qx - qy*qy + qz*qz
            )
    V1c = (1/(m+1e-5))*vertcat(
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

def extract_homing_variables(model:AcadosModel, pt_w:np.ndarray, Tb2c:np.ndarray) -> tuple[SX, SX]:
    """
    Extracts the pixel states from the full state vector.
    
    """

    # Extract some useful variables
    pb_w = vertcat(model.x[0], model.x[1], model.x[2])
    vb_w = vertcat(model.x[3], model.x[4], model.x[5])
    qx,qy,qz,qw = model.x[6],model.x[7],model.x[8],model.x[9]
    wx,wy,wz = model.u[1],model.u[2],model.u[3]

    # Extract transforms
    Rb2c = Tb2c[:3,:3]
    pb_c = Tb2c[0:3,3]

    # Compute skew symmetric matrix
    Wb = vertcat(
        horzcat(0.0, -wz, wy),
        horzcat(wz, 0.0, -wx),
        horzcat(-wy, wx, 0.0)
    )

    # Compute rotation matrix from quaternion
    Rb2w = vertcat(
        horzcat(1.0-2.0*(qy**2+qz**2), 2.0*(qx*qy-qw*qz), 2.0*(qx*qz+qw*qy)),
        horzcat(2.0*(qx*qy+qw*qz), 1.0-2*(qx**2+qz**2), 2.0*(qy*qz-qw*qx)),
        horzcat(2.0*(qx*qz-qw*qy), 2.0*(qy*qz+qw*qx), 1.0-2.0*(qx**2+qy**2))
    )
    Rw2b = Rb2w.T
    
    # Body frame velocities
    vb_b = Rw2b@vb_w
    
    # Compute the pixel states
    pt_b = Rw2b@(pt_w-pb_w)              # Target in body frame
    pt_c = Rb2c@pt_b + pb_c              # Target in camera frame
    l_c = -Rb2c@(Rw2b@vb_w+Wb@pt_b)      # Line of sight velocity in camera frame
    
    fx,fy,cx,cy = 462.956,463.002,323.076,181.184
    u = fx * pt_c[0] / pt_c[2] + cx
    v = fy * pt_c[1] / pt_c[2] + cy

    ud = fx*(l_c[0]*pt_c[2]-pt_c[0]*l_c[2])/pt_c[2]**2
    vd = fy*(l_c[1]*pt_c[2]-pt_c[1]*l_c[2])/pt_c[2]**2

    # Normalize pixel coordinates to [-1,1] from [0,640]x[0,360], and z to [0,6]m
    un,vn = 320,180

    u = (u-un)/un
    v = (v-vn)/vn
    ud = ud/un
    vd = vd/vn

    # Sigmoid the boi
    dist = norm_2(pt_w-pb_w)
    Kfr = 1/(1+exp(-3.68*(dist-1.75)))
    Knr = 1/(1e-3+1+exp( 1.50*(dist-2.50)))

    # Pack the output
    p_px = vertcat(u,v)
    v_px = vertcat(ud,vd)
    y_expr = vertcat(
        pt_b,
        vb_b,
        p_px,
        v_px,
        model.u
        )
    y_expr_e = vertcat(
        pt_b,
        vb_b,
        p_px
        )

    return y_expr, y_expr_e
