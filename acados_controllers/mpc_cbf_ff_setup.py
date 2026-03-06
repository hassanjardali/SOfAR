import numpy as np
import os
from pathlib import Path
from types import SimpleNamespace
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat, sin, cos, tan, arctan, exp, fmax, fmin

rebuild = True # Set to True to rebuild the acados solver. Do that if you change something not in the p vector.

mpc_params_dict = {
    "Tf": 1.5,  # Prediction horizon [s]
    "N": 20,  # Number of control intervals
    "Q": [0,1,0.0,0.5,0.0],  # State weights - [s, n, e, v, delta]
    "R": [0.10, 10.0],  # Weight for control input [a, delta_dot]
    "z_low": 200000,  # Lower bound for slack variables (L1)
    "z_up": 200000,  # Upper bound for slack variables (L1)
    "Z_low": 800000,  # Lower bound for slack variables (L2)
    "Z_up": 800000,  # Upper bound for slack variables (L2)
    "qp_solver": "FULL_CONDENSING_HPIPM",  # QP solver
    "nlp_solver_type": "SQP_RTI",  # NLP solver type
    "hessian_approx": "GAUSS_NEWTON",  # Hessian approximation
    "integrator_type": "ERK",  # Integrator type
    "sim_method_num_stages": 3,  # Number of stages in the integrator
    "sim_method_num_steps": 4,  # Number of steps in the integrator
    "nlp_solver_max_iter": 2000,  # Maximum number of iterations for the NLP solver
    "tol": 1e-3,  # Tolerance for the NLP solver
}

car_params_dict = {
    "L": 2.9718,  # Wheelbase length [m]
    "l_r": 1.2933,  # Distance from CG to rear axle [m]
}

constraints_dict = {
    "n_min": -2.0,  # Track width boundaries [m]
    "n_max": 2.0,  # Track width boundaries [m]
    "delta_min": -0.25,  # Steering angle limits [rad]
    "delta_max": 0.25,  # Steering angle limits [rad]
    "delta_dot_min": -0.35,  # Steering rate limits [rad/s]
    "delta_dot_max": 0.35,  # Steering rate limits [rad/s]
    "acceleration_min": -10.0,  # Acceleration limits [m/s^2]
    "acceleration_max": 3.0,  # Acceleration limits [m/s^2]
    "alat_min": -15.0,  # Lateral acceleration limit (minimum) [m/s^2]
    "alat_max": 15.0,  # Lateral acceleration limit (maximum) [m/s^2]
    "gamma": 0.5,  # CBF parameter
    "ellipse_width": 2.5,  # Lateral width (total)
    "ellipse_height": 6.0  # Longitudinal length (total)
}

x_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Starting at centerline with no heading error
u_0 = np.array([0.0, 0.0])


def FrenetFrameKinematicModel(n_obstacles=2):
    model_name = "FrenetFrameKinematicModel"

    # --- 1. Define Symbols ---
    s, n, e, v, delta = SX.sym("s"), SX.sym("n"), SX.sym("e"), SX.sym("v"), SX.sym("delta")
    x = vertcat(s, n, e, v, delta)

    a, delta_dot = SX.sym("a"), SX.sym("delta_dot")
    u = vertcat(a, delta_dot)
    
    xdot = SX.sym("xdot", 5, 1)

    # --- 2. Define Parameters (TVP) ---
    s_ref, n_ref, e_ref, v_ref, delta_ref, kappa_ref = SX.sym("s_ref"), SX.sym("n_ref"), SX.sym("e_ref"), SX.sym("v_ref"), SX.sym("delta_ref"), SX.sym("kappa_ref")
    n_max_p, n_min_p = SX.sym("n_max"), SX.sym("n_min")
    Q_diag, R_diag = SX.sym("Q_diag", 5), SX.sym("R_diag", 2)
    gamma_p = SX.sym("gamma")
    p_obs = SX.sym("p_obs", n_obstacles * 3)

    p = vertcat(s_ref, n_ref, e_ref, v_ref, delta_ref, kappa_ref, 
                n_max_p, n_min_p, Q_diag, R_diag, gamma_p, p_obs)

    # --- 3. Dynamics ---
    lr, L = car_params_dict["l_r"], car_params_dict["L"]
    beta = arctan((lr / L) * tan(delta))
    denom = (1 - kappa_ref * n) 
    
    ds = (v * cos(e + beta)) / denom
    dn = v * sin(e + beta)
    dpsi = (v / L) * cos(beta) * tan(delta)
    de = dpsi - kappa_ref * ds
    
    f_expl = vertcat(ds, dn, de, a, delta_dot)

    # --- 4. Cost Definition ---
    err_x = vertcat(s-s_ref, n-n_ref, e-e_ref, v-v_ref, delta-delta_ref)
    
    cost_track = 0
    for i in range(5): cost_track += Q_diag[i] * err_x[i]**2
    cost_input = R_diag[0]*a**2 + R_diag[1]*delta_dot**2
    
    expr_ext_cost = cost_track + cost_input 

    # --- 5. Constraints Logic (ELLIPSE CBF) ---
    constraint_exprs = []
    
    # A. Lateral Acceleration
    a_lat = v * dpsi
    constraint_exprs.append(a_lat) 
    margin = 0.85
    constraint_exprs.append(n - n_max_p + margin)  # n <= n_max_p - margin
    constraint_exprs.append(n_min_p + margin - n)  # n >= n_min_p + margin

    # B. CBF Logic
    Ts = mpc_params_dict["Tf"] / mpc_params_dict["N"]
    
    # Helper function to compute dynamics at given state
    def compute_dynamics(s_val, n_val, e_val, v_val, delta_val):
        beta_val = arctan((lr / L) * tan(delta_val))
        denom_val = (1 - kappa_ref * n_val)
        ds_val = (v_val * cos(e_val + beta_val)) / denom_val
        dn_val = v_val * sin(e_val + beta_val)
        dpsi_val = (v_val / L) * cos(beta_val) * tan(delta_val)
        de_val = dpsi_val - kappa_ref * ds_val
        return vertcat(ds_val, dn_val, de_val, a, delta_dot)
    
    # 1. Symbolic Lookahead (x_k1) using RK4
    k1 = f_expl
    x_k2 = x + (Ts/2) * k1
    k2 = compute_dynamics(x_k2[0], x_k2[1], x_k2[2], x_k2[3], x_k2[4])
    x_k3 = x + (Ts/2) * k2
    k3 = compute_dynamics(x_k3[0], x_k3[1], x_k3[2], x_k3[3], x_k3[4])
    x_k4 = x + Ts * k3
    k4 = compute_dynamics(x_k4[0], x_k4[1], x_k4[2], x_k4[3], x_k4[4])
    x_k1 = x + (Ts/6) * (k1 + 2*k2 + 2*k3 + k4) 
    

    # --- HELPER: h_ellipse_sym  ---
    def h_ellipse_sym(state_vec, obs_vec):
        s_curr, n_curr = state_vec[0], state_vec[1]
        v_curr = state_vec[3]
        
        s_obs_val, n_obs_val = obs_vec[0], obs_vec[1]
        v_obs_val = obs_vec[2]
        
        a_min_mag = abs(constraints_dict["acceleration_min"])
        
        v_rel = fmax(0.0, v_curr - v_obs_val)
        stopping_dist = (v_rel**2) / (2.0 * a_min_mag)
        
        a_axis_base = constraints_dict["ellipse_height"] / 2.0
        a_axis_dynamic = a_axis_base + stopping_dist
        
        b_axis = constraints_dict["ellipse_width"] / 2.0
        
        s_diff = s_curr - s_obs_val
        n_diff = n_curr - n_obs_val
        
        margin = 0.5
        
        h = (s_diff / a_axis_dynamic)**2 + (n_diff / b_axis)**2 - (1.0 + margin)
        return h

    for i in range(n_obstacles):
        # Extract Obs_k
        idx = i * 3
        s_obs_k = p_obs[idx]
        n_obs_k = p_obs[idx+1]
        v_obs   = p_obs[idx+2]
        
        obs_k = vertcat(s_obs_k, n_obs_k, v_obs)

        # Predict Obs_k1 (Constant Velocity)
        s_obs_k1 = s_obs_k + v_obs * Ts
        n_obs_k1 = n_obs_k 
        
        obs_k1 = vertcat(s_obs_k1, n_obs_k1, v_obs)

        # Compute CBF Condition
        h_k = h_ellipse_sym(x, obs_k)
        h_k1 = h_ellipse_sym(x_k1, obs_k1)

        # Constraint: (1-gamma)*h_k - h_k1 <= 0
        cbf_expr = (1 - gamma_p) * h_k - h_k1
        constraint_exprs.append(cbf_expr)
        
    con_h_expr = vertcat(*constraint_exprs)

    # --- Pack Model ---
    model = SimpleNamespace()
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x; model.xdot = xdot; model.u = u; model.p = p
    model.cost_expr_ext_cost = expr_ext_cost
    model.cost_expr_ext_cost_e = cost_track 
    model.con_h_expr = con_h_expr
    model.name = model_name
    model.x0 = x_0

    constraint = SimpleNamespace()
    constraint.alat_min = constraints_dict["alat_min"]
    constraint.alat_max = constraints_dict["alat_max"]
    constraint.delta_min = constraints_dict["delta_min"]
    constraint.delta_max = constraints_dict["delta_max"]
    constraint.a_min = constraints_dict["acceleration_min"]
    constraint.a_max = constraints_dict["acceleration_max"]
    constraint.ddelta_min = constraints_dict["delta_dot_min"]
    constraint.ddelta_max = constraints_dict["delta_dot_max"]

    return model, constraint


def acados_settings(n_obstacles=2):
    
    model, constraint = FrenetFrameKinematicModel(n_obstacles=n_obstacles)
    
    ocp = AcadosOcp()
    
    # Fill AcadosModel
    model_ac = AcadosModel()
    model_ac.f_impl_expr = model.f_impl_expr
    model_ac.f_expl_expr = model.f_expl_expr
    model_ac.x = model.x
    model_ac.xdot = model.xdot
    model_ac.u = model.u
    model_ac.p = model.p
    model_ac.cost_expr_ext_cost = model.cost_expr_ext_cost
    model_ac.cost_expr_ext_cost_e = model.cost_expr_ext_cost_e
    model_ac.con_h_expr = model.con_h_expr
    model_ac.name = model.name
    ocp.model = model_ac

    # Dimensions
    nh = model.con_h_expr.size1()

    ocp.dims.N = mpc_params_dict["N"]
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    # Hard Bounds (Controls & States)
    ocp.constraints.lbu = np.array([constraint.a_min, constraint.ddelta_min])
    ocp.constraints.ubu = np.array([constraint.a_max, constraint.ddelta_max])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lbx = np.array([constraint.delta_min])
    ocp.constraints.ubx = np.array([constraint.delta_max])
    ocp.constraints.idxbx = np.array([4])

    # Nonlinear Constraints (h)
    # [a_lat, cbf_0, cbf_1...]
    
    # Bounds for CBF logic: val <= 0
    # Lower: -Infinity (-1e9)
    # Upper: 0.0
    
    
    lh = np.concatenate((
        [constraint.alat_min],   # a_lat
        [-1e9, -1e9],            # n_max, n_min (effectively no lower bound for these <= 0 forms)
        -1e9 * np.ones(n_obstacles) # CBFs
    ))

    # 2. Upper Bounds (uh)
    # All must be <= 0 (except a_lat which is <= alat_max)
    uh = np.concatenate((
        [constraint.alat_max],   # a_lat
        [0.0, 0.0],              # n_max, n_min
        np.zeros(n_obstacles)    # CBFs
    ))
    
    ocp.constraints.lh = lh
    ocp.constraints.uh = uh
    
    # Soft Constraints (Slack)
    ocp.constraints.lsh = np.zeros(nh)
    ocp.constraints.ush = np.zeros(nh)
    ocp.constraints.idxsh = np.array(range(nh))
    #ocp.cost.zl = mpc_params_dict["z_low"] * np.ones(nh)
    #ocp.cost.zu = mpc_params_dict["z_up"] * np.ones(nh)
    #ocp.cost.Zl = mpc_params_dict["Z_low"] * np.ones(nh)
    #ocp.cost.Zu = mpc_params_dict["Z_up"] * np.ones(nh)


    
    # --- PRIORITY 3: Lateral Acceleration (Index 0) ---
    Z_val = np.zeros(nh)
    z_val = np.zeros(nh)
    # Lowest Priority: We prefer "uncomfortable" jerks over crashing.
    # Cost: Low
    # Relaxed compared to baseline
    z_val[0] = 4000 
    Z_val[0] = 4000 

    # --- PRIORITY 2: Out of Bounds (Indices 1 & 2) ---
    # Your working baseline
    z_val[1:3] = 35000 #5500
    Z_val[1:3] = 60000

    # --- PRIORITY 1: Obstacles (Indices 3 to End) ---
    # Slightly higher than baseline, but low enough to maintain matrix conditioning
    z_val[3:] = 90000 #90000
    Z_val[3:] = 90000

    # --- Apply to Acados ---
    ocp.cost.zl = z_val
    ocp.cost.zu = z_val
    ocp.cost.Zl = Z_val
    ocp.cost.Zu = Z_val


    ocp.constraints.x0 = x_0

    # Solver Options
    ocp.solver_options.tf = mpc_params_dict["Tf"]
    ocp.solver_options.qp_solver = mpc_params_dict["qp_solver"]
    ocp.solver_options.nlp_solver_type = mpc_params_dict["nlp_solver_type"]
    ocp.solver_options.hessian_approx = mpc_params_dict["hessian_approx"]
    ocp.solver_options.integrator_type = mpc_params_dict["integrator_type"]
    ocp.solver_options.nlp_solver_max_iter = mpc_params_dict["nlp_solver_max_iter"]
    ocp.solver_options.tol = mpc_params_dict["tol"]
    ocp.solver_options.sim_method_num_stages = mpc_params_dict["sim_method_num_stages"]
    ocp.solver_options.sim_method_num_steps = mpc_params_dict["sim_method_num_steps"]

    # Initial Parameter Vector (Placeholder)
    p_size = 6 + 2 + 5 + 2 + 1 + n_obstacles * 3
    p0 = np.zeros(p_size)
    p0[0:6] = [0,0,0,1,0,0] # Refs 
    p0[6:8] = [constraints_dict["n_max"], constraints_dict["n_min"]] # Limits
    p0[8:13] = mpc_params_dict["Q"]
    p0[13:15] = mpc_params_dict["R"]
    p0[15] = constraints_dict["gamma"] # gamma
    # Obstacles (far away)
    for i in range(n_obstacles):
        idx = 16 + i*3
        p0[idx] = 1000 + i*10 # s
    
    ocp.parameter_values = p0

    # Check if generated code already exists
    json_file = "acados_ocp.json"
    model_name = ocp.model.name
    so_file = Path("c_generated_code") / f"libacados_ocp_solver_{model_name}.so"
    build = not so_file.exists()
    
    # Generate
    if rebuild:
        acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")
    else:
        acados_solver = AcadosOcpSolver(ocp, json_file=json_file, build=build)


    return constraint, model, acados_solver