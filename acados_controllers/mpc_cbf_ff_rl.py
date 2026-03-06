# Prevent Qt conflicts with opencv - unset opencv's Qt plugin path before any cv2 import
import os
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
os.environ['QT_PLUGIN_PATH'] = ''
# Tell opencv to not use Qt
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


from pathlib import Path
import sys
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from acados_controllers import mpc_cbf_ff_setup
import numpy as np
from casadi import *
import gurobipy as gp
from gurobipy import GRB
import time
import torch
GAMMA_BOUNDS = (0.05, 1.0)
REWARD_WEIGHTS_VEC = [0.31, 0.45, 0.05, 0.04, 0.15]

max_obstacles = 4

class KinematicModelMIQPMPCinFF:
    def __init__(self, config):
        # Handle both dict and object config
        if isinstance(config, dict) and 'miqp' in config:
            self.miqp_params = config['miqp']
            self.frenet_mpc_params = config['frenet_mpc']
            self.obstacles_on = config['obstacles_on']

        else:
            self.miqp_params = config.miqp
            self.frenet_mpc_params = config.frenet_mpc
            self.obstacles_on = config.obstacles_on

           

        if isinstance(self.frenet_mpc_params, dict):
            self.Ts = self.frenet_mpc_params['Ts']                      # Sampling time - step
            self.T_horizon = self.frenet_mpc_params['T_horizon']        # Prediction horizon time steps
            self.delta_limit = self.frenet_mpc_params['delta_limit']    # Delta angle limit
            self.gamma = self.frenet_mpc_params['gamma']                # CBF parameter (may be updated by RL)
            self.gamma_default = float(self.gamma)                      # Fixed reference gamma for visualization
            self.ellipse_width = self.frenet_mpc_params.get('ellipse_width', 2.0)  # Ellipse width (lateral dimension)
            self.ellipse_height = self.frenet_mpc_params.get('ellipse_height', 6.0)  # Ellipse height (longitudinal dimension)
            self.R = np.diag(self.frenet_mpc_params['R'])  # Convert [2x1] vector to 2x2 diagonal matrix
            self.Q = np.diag(self.frenet_mpc_params['Q'])
            self.R_diag = np.array(self.frenet_mpc_params['R'])  # 1D array for parameters
            self.Q_diag = np.array(self.frenet_mpc_params['Q'])  # 1D array for parameters
            self.max_alat = self.frenet_mpc_params['max_alat']    # Max lateral acceleration limit
        else:
            self.Ts = self.frenet_mpc_params.Ts                      # Sampling time - step
            self.T_horizon = self.frenet_mpc_params.T_horizon        # Prediction horizon time steps
            self.delta_limit = self.frenet_mpc_params.delta_limit    # Steering angle limit
            self.gamma = self.frenet_mpc_params.gamma                # CBF parameter (may be updated by RL)
            self.gamma_default = float(self.gamma)                   # Fixed reference gamma for visualization
            self.ellipse_width = getattr(self.frenet_mpc_params, 'ellipse_width', 2.0)  # Ellipse width (lateral dimension)
            self.ellipse_height = getattr(self.frenet_mpc_params, 'ellipse_height', 6.0)  # Ellipse height (longitudinal dimension)
            self.R = np.diag(self.frenet_mpc_params.R)  # Convert [2x1] vector to 2x2 diagonal matrix
            self.Q = np.diag(self.frenet_mpc_params.Q)
            self.R_diag = np.array(self.frenet_mpc_params.R)  # 1D array for parameters
            self.Q_diag = np.array(self.frenet_mpc_params.Q)  # 1D array for parameters
            self.max_alat = getattr(self.frenet_mpc_params, 'max_alat', 8.0)    # Max lateral acceleration limit

        if self.obstacles_on:
            self.obs = None               # Obstacles

        self.tvp_cache = None
        self.previous_kappa = 0.0
        self.last_successful_trajectory_frenet = None
        self.last_successful_trajectory_cartesian = None
        self.prev_miqp_path = None  # Initialize previous MIQP path
        self.last_rl_computation_time = 0.0  # Initialize RL inference computation time
        self.miqp_step_counter = 0

        self.h_activate = self.ellipse_width**2


        self.last_miqp_success = False

        # Low pass filter for gamma (time constant in seconds)
        gamma_filter_tau = 0.1  # Make filter nearly transparent (very short time constant)
        self.gamma_filter_alpha = self.Ts / (gamma_filter_tau + self.Ts) if gamma_filter_tau > 0 else 1.0  # Alpha close to 1
        self.gamma_filtered = float(self.gamma)  # Initialize filtered gamma

        # Load PASTA RL model
        model_path = project_root / "rl_pasta" / "logs" / "final_model.pt" 
        try:
            print(f"Loading PASTA model from: {model_path}")
            
            if not model_path.exists():
                raise FileNotFoundError(f"PASTA model not found at {model_path}")
            
            import rl_pasta.pasta as pasta_module
            import sys
            sys.modules['pasta'] = pasta_module
            # Handle numpy version mismatch (checkpoint saved with numpy 2.x)
            import numpy.core
            if 'numpy._core' not in sys.modules:
                sys.modules['numpy._core'] = numpy.core
                sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
            self.rl_model = torch.load(str(model_path), map_location="cpu", weights_only=False)
            self.rl_model.device = torch.device("cpu")
            self.rl_model.actor.eval()
            self.rl_model.set_preference(np.array(REWARD_WEIGHTS_VEC, dtype=np.float32))
            print(f"PASTA model loaded successfully from {model_path}")
            print(f"RL agent will adaptively update gamma based on current state")
        except Exception as e:
            print(f"Warning: Could not load PASTA model: {e}")
            import traceback
            traceback.print_exc()
            print(f"Falling back to fixed gamma={self.gamma}")
            self.rl_model = None

        # Initialize MPC components with error handling
        
        try:
            self.constraint, self.model, self.acados_solver = mpc_cbf_ff_setup.acados_settings(n_obstacles=max_obstacles)
            print("MPC Acados CBF FF controller initialized successfully!")
        except Exception as e:
            print(f"Error initializing MPC Acados CBF FF controller: {e}")
            raise


    def get_current_state_action(self, x_frenet, target_trajectory_frenet=None):
        """
        Get gamma from RL model based on current state.
        
        Args:
            x_frenet: [s, n, e, v, delta] - current state in Frenet frame
            target_trajectory_frenet: target trajectory in Frenet frame (optional)
        
        Returns:
            gamma: CBF parameter value
        """
        if self.rl_model is None:
            # Return default gamma if RL model not loaded
            if not hasattr(self, '_rl_warning_printed'):
                print(f"Warning: RL model not available, using fixed gamma={self.gamma}")
                self._rl_warning_printed = True
            return self.gamma
        
        # Extract ego state
        n_ego = x_frenet[1]  # Lateral offset
        e_ego = x_frenet[2]  # Heading error
        v_ego = x_frenet[3]  # Velocity
        steering = x_frenet[4]  # Steering angle
        
        # Extract obstacle states (up to 2 obstacles)
        s_obs1, s_obs2 = 500.0, 500.0  # matches training sentinel for "no obstacle nearby"
        n_obs1, n_obs2 = 0.0, 0.0
        v_obs1, v_obs2 = 0.0, 0.0
        
        if self.obs is not None and len(self.obs) > 0:
            obstacles_frenet = np.array(self.obs)
            if obstacles_frenet.ndim == 1:
                obstacles_frenet = obstacles_frenet.reshape(1, -1)
            
            # Filter obstacles with s_obs >= -20 and sort by s_obs
            if obstacles_frenet.shape[0] > 0 and obstacles_frenet.shape[1] >= 3:
                valid_mask = obstacles_frenet[:, 0] >= -20.0
                valid_obstacles = obstacles_frenet[valid_mask]
                
                if len(valid_obstacles) > 0:
                    sort_indices = np.argsort(valid_obstacles[:, 0])
                    sorted_obstacles = valid_obstacles[sort_indices]
                    
                    if len(sorted_obstacles) >= 1:
                        s_obs1 = float(sorted_obstacles[0, 0])
                        n_obs1 = float(sorted_obstacles[0, 1])
                        v_obs1 = v_ego - float(sorted_obstacles[0, 2])  # Relative velocity
                    
                    if len(sorted_obstacles) >= 2:
                        s_obs2 = float(sorted_obstacles[1, 0])
                        n_obs2 = float(sorted_obstacles[1, 1])
                        v_obs2 = v_ego - float(sorted_obstacles[1, 2])  # Relative velocity
        
        # Average curvature from target trajectory
        if target_trajectory_frenet is not None and target_trajectory_frenet.shape[0] > 0:
            if target_trajectory_frenet.shape[1] > 5:
                curvatures = target_trajectory_frenet[:, 5]  # kappa column
                avg_curvature = float(np.mean(curvatures))
            else:
                avg_curvature = 0.0
        else:
            avg_curvature = 0.0
        
        # MIQP n value (from previous path if available)
        miqp_n_value = 0.0
        if self.prev_miqp_path is not None and len(self.prev_miqp_path) > 0:
            miqp_n_value = float(self.prev_miqp_path[0])
        
        # Boundary distances from target trajectory
        if target_trajectory_frenet is not None and target_trajectory_frenet.shape[0] > 0:
            if target_trajectory_frenet.shape[1] > 7:
                # Use first point's boundary distances
                nearest_inner_boundary_distance = float(target_trajectory_frenet[0, 6])
                nearest_outer_boundary_distance = float(abs(target_trajectory_frenet[0, 7]))
            else:
                nearest_inner_boundary_distance = 2.0
                nearest_outer_boundary_distance = 2.0
        else:
            nearest_inner_boundary_distance = 2.0
            nearest_outer_boundary_distance = 2.0
        
        # Minimum h value (compute from current state and obstacles)
        min_h_value = 0.0
        if self.obs is not None and len(self.obs) > 0:
            h_values_current = []
            for obs in self.obs:
                if len(obs) >= 3:
                    # Compute h for current state
                    if hasattr(self, 'ellipse_width') and hasattr(self, 'ellipse_height'):
                        h_val = self._compute_h_ellipse_numeric(x_frenet, obs)
                    else:
                        h_val = self._compute_h_circle_numeric(x_frenet, obs)
                    h_values_current.append(h_val)
            if len(h_values_current) > 0:
                min_h_value = float(np.min(h_values_current))
        # Fallback to previous values if available
        elif hasattr(self, 'current_h_values') and self.current_h_values is not None:
            if len(self.current_h_values) > 0:
                min_h_value = float(np.min(self.current_h_values))
        
        # Build state vector
        state = np.array([
            n_ego, 
            # e_ego, 
            v_ego, steering,
            s_obs1, s_obs2, n_obs1, n_obs2, v_obs1, v_obs2,
            avg_curvature, 
            # miqp_n_value,
            nearest_inner_boundary_distance, nearest_outer_boundary_distance,
            # min_h_value
        ], dtype=np.float32)
        
        # Predict action (gamma) via PASTA - output is in [0,1] via sigmoid
        action, _, _ = self.rl_model.select_action(state, deterministic=True)
        
        # Map from [0, 1] to [GAMMA_BOUNDS[0], GAMMA_BOUNDS[1]]
        gamma = float(GAMMA_BOUNDS[0] + action[0] * (GAMMA_BOUNDS[1] - GAMMA_BOUNDS[0]))
        gamma = float(np.clip(gamma, GAMMA_BOUNDS[0], GAMMA_BOUNDS[1]))
        
        # Apply low pass filter to gamma
        if not hasattr(self, 'gamma_filtered'):
            self.gamma_filtered = float(gamma)
        else:
            self.gamma_filtered = self.gamma_filter_alpha * float(gamma) + (1.0 - self.gamma_filter_alpha) * self.gamma_filtered
        
        # Update self.gamma (do NOT touch gamma_default)
        self.gamma = float(self.gamma_filtered)
        
        
        return gamma


    def get_control_input(self, x, target_trajectory, obs=None, gamma=None, simulator_dt=None):
        """Computes the optimal control input using Acados MPC.

        Inputs:
          - x: The current state vector [5x1] = [s, n, e, v, delta] (Cartesian)
          - target_trajectory: A 2D numpy array of shape (N+1, 8)
          - obs: Obstacle array (Cartesian)
        """
        import time
        mpctiming = {}
        t0_start = time.time()
        
        # 1. --- Pre-processing (Cartesian to Frenet) ---
        t0 = time.time()
        
        # Ensure target_trajectory is a numpy array
        target_trajectory_cartesian = np.array(target_trajectory)
        
        # Validate shape (MIQP horizon check)
        expected_length = self.miqp_params['planning_horizon'] + 1
        if target_trajectory_cartesian.shape[0] < expected_length:
             print(f"Warning: Trajectory length {target_trajectory_cartesian.shape[0]} < expected {expected_length}")

        # Process Obstacles (Cartesian -> Frenet)
        if self.obstacles_on and obs is not None and len(obs) > 0:
            self._precompute_curve_geometry(target_trajectory_cartesian)
            obs_frenet = np.zeros((len(obs), 3))
            for i in range(len(obs)):
                # Convert each obstacle to Frenet [s, n, v]
                obs_frenet[i] = self.cartesian_to_frenet_obstacle(obs[i], target_trajectory_cartesian)
            self.obs = obs_frenet 
        else:
            self.obs = []
            
        mpctiming['obstacle_setup'] = (time.time() - t0) * 1000
        
        t0 = time.time()
        
        # Create Reference Trajectory (Frenet)
        miqp_target_trajectory_frenet = self.create_target_trajectory_frenet(target_trajectory_cartesian)
        
        # Convert Current State to Frenet
        # Note: Ensure this returns exactly [s, n, e, v, delta]
        x_frenet = self.cartesian_to_frenet(x, target_trajectory_cartesian)

        # -------------------------- MIQP LOGIC (Optional, runs at 1:2 MPC freq) --------------------------
        self.last_miqp_computation_time = 0.0  # Default: no MIQP this step
        if self.obstacles_on and self.miqp_params['use_miqp']:
            self.miqp_step_counter += 1
            run_miqp_now = (self.miqp_step_counter % 2 == 1) or (self.last_successful_trajectory_cartesian is None)

            if run_miqp_now:
                target_trajectory_frenet_long, target_trajectory_cartesian_miqp = self.miqp_traj_generation(
                    miqp_target_trajectory_frenet, target_trajectory_cartesian, x_frenet, prev_path_n=self.prev_miqp_path
                )
                if self.last_miqp_success:
                    self.last_successful_trajectory_frenet = target_trajectory_frenet_long.copy()
                    self.last_successful_trajectory_cartesian = target_trajectory_cartesian_miqp.copy() if target_trajectory_cartesian_miqp is not None else None
                elif self.last_successful_trajectory_frenet is not None:
                    target_trajectory_frenet_long = self.last_successful_trajectory_frenet
                    target_trajectory_cartesian_miqp = self.last_successful_trajectory_cartesian
            else:
                # Skip MIQP: re-project previous Cartesian MIQP path into new Frenet frame
                target_trajectory_frenet_long = self._retransform_miqp_to_new_frame(
                    self.last_successful_trajectory_cartesian,
                    miqp_target_trajectory_frenet,
                    target_trajectory_cartesian
                )
                target_trajectory_cartesian_miqp = self.last_successful_trajectory_cartesian
        else:
            target_trajectory_frenet_long = miqp_target_trajectory_frenet
            target_trajectory_cartesian_miqp = None
        # ---------------------------------------------------------------------------

        # Slice trajectory to MPC Horizon
        # Use self.N (class attribute) or extract from params dict
        N_mpc = mpc_cbf_ff_setup.mpc_params_dict['N'] 
        target_trajectory_frenet = target_trajectory_frenet_long[:N_mpc + 1, :]
        
        # Store for visualization if needed
        self.current_target_trajectory = target_trajectory_frenet 
        
        mpctiming['traj_setup'] = (time.time() - t0) * 1000


        # Get gamma from RL model (or keep default if RL not available)
        t0 = time.time()
        if gamma is None:
            gamma = float(self.get_current_state_action(x_frenet, target_trajectory_frenet))
        rl_inference_time = time.time() - t0
        mpctiming['gamma_setup'] = rl_inference_time * 1000
        # Store RL inference time for external tracking
        self.last_rl_computation_time = rl_inference_time
        
        # Verify gamma is valid
        if gamma is None or np.isnan(gamma):
            print(f"WARNING: Invalid gamma={gamma}, using default={self.gamma_default}")
            gamma = self.gamma_default

        # Debug Print
        # print(f"State: s={x_frenet[0]:.1f}, n={x_frenet[1]:.2f}, e={x_frenet[2]:.2f}, v={x_frenet[3]:.1f}")
        
        # 2. --- Acados Solver Setup ---
        t0 = time.time()
        
        # Initialize horizon on the very first run to prevent initial jitter
        if not hasattr(self, 'mpc_initialized') or not self.mpc_initialized:
            for k in range(N_mpc + 1):
                # Seed the state with the target reference trajectory
                ref_state = target_trajectory_frenet[k, 0:5]
                self.acados_solver.set(k, "x", ref_state)
                # Seed controls with zeros (steady state)
                if k < N_mpc:
                    self.acados_solver.set(k, "u", np.array([0.0, 0.0]))
            self.mpc_initialized = True
        else:
            # Shift internal states and controls forward by one step for RTI warm-starting
            for k in range(N_mpc):
                self.acados_solver.set(k, "x", self.acados_solver.get(k + 1, "x"))
            for k in range(N_mpc - 1):
                self.acados_solver.set(k, "u", self.acados_solver.get(k + 1, "u"))
            
            # Duplicate the last node to fill the end of the shifted horizon
            self.acados_solver.set(N_mpc, "x", self.acados_solver.get(N_mpc, "x"))
            if N_mpc > 0:
                self.acados_solver.set(N_mpc - 1, "u", self.acados_solver.get(N_mpc - 1, "u"))
        
        # A. Update Initial State (Constraint)
        # In Acados, x0 is enforced via bounds on the first node (shooting node 0)
        self.acados_solver.set(0, "lbx", x_frenet)
        self.acados_solver.set(0, "ubx", x_frenet)
        
        # B. Update Parameters (The "TVP" Loop)
        # Parameter vector p structure:
        # [0-5]: Refs (s, n, e, v, delta, kappa)
        # [6-7]: Limits (n_max, n_min)
        # [8-12]: Q_diag
        # [13-14]: R_diag
        # [15]: gamma
        # [16+]: Obstacles (s, n, v) * N_obs
        
        Ts = mpc_cbf_ff_setup.mpc_params_dict['Tf'] / N_mpc
        n_obs_max = getattr(self, 'max_obstacles', max_obstacles) # Default to 10
        
        # Pre-allocate parameter vector
        p_len = 16 + n_obs_max * 3
        p_val = np.zeros(p_len)

        # Set Weights (Constant across horizon)
        p_val[8:13] = self.Q_diag
        p_val[13:15] = self.R_diag
        p_val[15] = gamma

        for k in range(N_mpc + 1):
            # 1. Reference & Limits
            # target_trajectory_frenet row: [s, n, e, v, delta, kappa, inner_dist, outer_dist]
            ref_k = target_trajectory_frenet[k]
            
            p_val[0:6] = ref_k[0:6]  # Refs: s, n, e, v, delta, kappa
            p_val[6]   = ref_k[6]    # n_max (Inner boundary)
            p_val[7]   = ref_k[7]    # n_min (Outer boundary)
            
            # 2. Obstacles (Prediction)
            t_predict = k * Ts
            
            for i in range(n_obs_max):
                idx = 16 + i * 3
                if i < len(self.obs):
                    # Active Obstacle: Predict position
                    s0, n0, v0 = self.obs[i]
                    
                    # Simple linear prediction
                    s_pred = s0 + v0 * t_predict 
                    
                    p_val[idx]   = s_pred
                    p_val[idx+1] = n0
                    p_val[idx+2] = v0
                else:
                    # Inactive Obstacle: Move far away
                    p_val[idx]   = 99999.0 + i*100
                    p_val[idx+1] = 0.0
                    p_val[idx+2] = 0.0

            # Set parameter for stage k
            self.acados_solver.set(k, "p", p_val)

        mpctiming['update_params'] = (time.time() - t0) * 1000

        # 3. --- Solve ---
        t0 = time.time()
        
        status = self.acados_solver.solve()
        
        mpctiming['solve'] = (time.time() - t0) * 1000
        
        mpc_success = (status == 0)
        if not mpc_success:
            print(f"WARNING: Acados returned status {status}")
            # Optional: self.acados_solver.print_statistics()

        # 4. --- Extract Results ---
        t0 = time.time()
        
        # Get optimal control at k=0
        # u = [a, delta_dot]
        u_opt = self.acados_solver.get(0, "u")
        a = u_opt[0]
        delta_dot = u_opt[1]
        
        # Get Predicted Trajectory (k=0 to N) for visualization
        pred_x_list = []
        for k in range(N_mpc + 1):
            x_k = self.acados_solver.get(k, "x")
            pred_x_list.append(x_k) # [s, n, e, v, delta]
            
        pred_x_array = np.array(pred_x_list)
        # Extract only [s, n, e, v, delta] for conversion
        predicted_trajectory_frenet = pred_x_array[:, :5] 

        # Calculate Delta Command
        delta = x_frenet[4]
        dt_int = simulator_dt if simulator_dt is not None else Ts
        # delta_cmd = delta + dt_int * delta_dot
        # Directly extract the steering command integrated by Acados for the next step
        delta_cmd = self.acados_solver.get(1, "x")[4]
        
        # Clip
        delta_cmd = np.clip(delta_cmd, -self.delta_limit, self.delta_limit)

        # Convert Prediction to Cartesian
        predicted_trajectory_cartesian = self.frenet_to_cartesian(predicted_trajectory_frenet, target_trajectory_cartesian)
        
        mpctiming['extract'] = (time.time() - t0) * 1000

        # 5. --- CBF Value Extraction (Optional Debug) ---
        h_values = None
        if self.obstacles_on and hasattr(self, 'obs') and self.obs is not None and predicted_trajectory_frenet is not None:
            num_obs = len(self.obs)
            # Only check up to what we have in prediction
            num_steps = min(predicted_trajectory_frenet.shape[0], N_mpc+1)
            h_values = np.zeros((num_steps, num_obs))
            
            for k in range(num_steps):
                x_pred = predicted_trajectory_frenet[k, :]
                for i in range(num_obs):
                    obs_state = self.obs[i]
                    h_val = self._compute_h_ellipse_numeric(x_pred, obs_state)
                    h_values[k, i] = h_val
        
        # Print timing breakdown
        total_time_ms = (time.time() - t0_start) * 1000
        if total_time_ms > 2.0:  # Only print if > 2ms
            print(f"  MPC breakdown (Total: {total_time_ms:.2f}ms):")
            for key, val in mpctiming.items():
                print(f"    {key:20s}: {val:.3f}ms ({val/total_time_ms*100:.1f}%)")

        # Always return gamma used for this solve (not whatever self.gamma might be later)
        return np.array([a, delta_cmd]), predicted_trajectory_cartesian, h_values, mpc_success, target_trajectory_cartesian_miqp, gamma
 
    
    def _compute_h_ellipse_numeric(self, x, obstacle_params):
        """Numeric version of h_ellipse for computing values."""
        s_obs = obstacle_params[0]
        n_obs = obstacle_params[1]
        a_axis = self.ellipse_height / 2.0
        b_axis = self.ellipse_width / 2.0
        s_diff = x[0] - s_obs
        n_diff = x[1] - n_obs
        margin = 0.0
        h_raw = (s_diff / a_axis)**2 + (n_diff / b_axis)**2 - (1.0 + margin)
        
        # Normalize and scale: handle wide range (0 to 1e6+)
        # Normalize by large factor to bring to reasonable range for tanh
        normalization_factor = 10000.0  # Normalize large values (1e6 -> 1e3)
        h_normalized = h_raw / normalization_factor
        
        # Apply tanh to smooth and bound between -1 and 1
        h_tanh = np.tanh(h_normalized)
        
        # Scale output to useful range (preserves sign, bounds extreme values)
        output_scale = 1.0  # Output range: [-10, 10]
        h = output_scale * h_tanh
        
        return h
    

    def create_target_trajectory_frenet(self, target_trajectory_cartesian):
        """Creates a target trajectory in Frenet coordinates.
        
        Inputs:
          - target_trajectory_cartesian: A 2D numpy array of shape (N+1, 8),
                               where N can be 2*self.T_horizon for MIQP.
                               Each row is [x, y, yaw, velocity, delta, curvature, inner_dist, outer_dist].
        Returns:
          - target_trajectory_frenet: A 2D numpy array of shape (N+1, 8),
                               matching the input length.
                               Each row is [s, 0, 0, velocity, delta, curvature, inner_dist, outer_dist].

        """
        traj_length = target_trajectory_cartesian.shape[0]
        target_trajectory_frenet = np.zeros((traj_length, 8))
        s_previous = 0
        s = 0
        for i in range(traj_length):
            n = 0
            e = 0
            velocity = target_trajectory_cartesian[i, 3]
            delta = target_trajectory_cartesian[i, 4] # Later will be replaced by feedforward delta
            curvature = target_trajectory_cartesian[i, 5]
            inner_dist = target_trajectory_cartesian[i, 6]
            outer_dist = target_trajectory_cartesian[i, 7]
            target_trajectory_frenet[i, 0] = s
            target_trajectory_frenet[i, 1] = n
            target_trajectory_frenet[i, 2] = e
            target_trajectory_frenet[i, 3] = velocity
            target_trajectory_frenet[i, 4] = delta
            target_trajectory_frenet[i, 5] = curvature
            target_trajectory_frenet[i, 6] = inner_dist # This is in n direction (positive is inside)
            target_trajectory_frenet[i, 7] = -outer_dist # This in n direction (negative is outside)
            s_previous = s
            s = s_previous + target_trajectory_cartesian[i, 3] * self.Ts # needs to be edited later. 

        return target_trajectory_frenet

    def cartesian_to_frenet(self, x_cartesian, target_trajectory_cartesian):
        # Uses index 0 of target as the reference frame origin
        ref_x = target_trajectory_cartesian[0, 0]
        ref_y = target_trajectory_cartesian[0, 1]
        ref_yaw = target_trajectory_cartesian[0, 2]
        
        dx = x_cartesian[0] - ref_x
        dy = x_cartesian[1] - ref_y
        
        cos_theta = np.cos(ref_yaw)
        sin_theta = np.sin(ref_yaw)
        
        # Rotate into path frame
        s = dx * cos_theta + dy * sin_theta
        n = -dx * sin_theta + dy * cos_theta
        
        e = x_cartesian[2] - ref_yaw
        e = (e + np.pi) % (2 * np.pi) - np.pi
        
        return np.array([s, n, e, x_cartesian[3], x_cartesian[4]])

    def _precompute_curve_geometry(self, traj):
        """Precompute segment geometry once per control step."""
        self._traj_A      = traj[:-1, :2]
        self._traj_seg_d  = np.diff(traj[:, :2], axis=0)
        self._traj_seg_len = np.linalg.norm(self._traj_seg_d, axis=1)
        S = np.zeros(traj.shape[0])
        S[1:] = np.cumsum(self._traj_seg_len)
        self._traj_S      = S
        L_sq = np.sum(self._traj_seg_d**2, axis=1)
        self._traj_L_sq_safe = np.where(L_sq > 1e-12, L_sq, 1.0)
        self._traj_psi    = traj[:, 2]

    def cartesian_to_frenet_obstacle(self, obstacle_cartesian, target_trajectory_cartesian):
        """Cartesian → Frenet. Returns [99999, 0, v] if obstacle is outside trajectory span."""
        obs_xy = obstacle_cartesian[:2]
        obs_v  = obstacle_cartesian[3]

        A       = self._traj_A
        seg_d   = self._traj_seg_d
        seg_len = self._traj_seg_len
        S       = self._traj_S
        safe    = self._traj_L_sq_safe
        psi_all = self._traj_psi

        d_obs = obs_xy - A
        t_raw = np.sum(d_obs * seg_d, axis=1) / safe

        interior = (t_raw >= 0.0) & (t_raw <= 1.0)
        if not np.any(interior):
            return np.array([99999.0, 0.0, obs_v])

        t_clmp  = np.clip(t_raw, 0.0, 1.0)
        proj    = A + t_clmp[:, None] * seg_d
        dist_sq = np.sum((obs_xy - proj)**2, axis=1)

        dist_sq_int = np.where(interior, dist_sq, np.inf)
        best_i = int(np.argmin(dist_sq_int))
        best_t = t_raw[best_i]

        best_s = S[best_i] + best_t * seg_len[best_i]

        psi_i    = psi_all[best_i]
        psi_next = psi_all[best_i + 1]
        dpsi     = ((psi_next - psi_i) + np.pi) % (2 * np.pi) - np.pi
        psi      = psi_i + best_t * dpsi

        foot   = proj[best_i]
        best_n = -(obs_xy[0] - foot[0]) * np.sin(psi) + \
                  (obs_xy[1] - foot[1]) * np.cos(psi)

        return np.array([best_s, best_n, obs_v])


    def frenet_to_cartesian(self, pred_frenet, target_cartesian):
        """
        Converts Frenet prediction back to Cartesian by wrapping it around the reference path.
        """
        if pred_frenet is None:
            return None

        num_points = pred_frenet.shape[0]
        pred_cartesian = np.zeros((num_points, 6))
        
        # We assume the prediction horizon steps align roughly with the target trajectory steps
        # This is a safe assumption for visualization if T_step is consistent
        max_target_idx = target_cartesian.shape[0] - 1

        for k in range(num_points):
            # 1. Get the predicted Frenet state
            # s_pred = pred_frenet[k, 0] # Not strictly needed for this simplified method
            n_pred = pred_frenet[k, 1]
            e_pred = pred_frenet[k, 2]
            
            # 2. Get the reference point on the curve at this step
            # Clamp index to avoid crashing if horizon > target points
            idx = min(k, max_target_idx)
            
            ref_x = target_cartesian[idx, 0]
            ref_y = target_cartesian[idx, 1]
            ref_yaw = target_cartesian[idx, 2]
        
            
            # Note: This "wraps" the straight Frenet line back onto the curve
            pred_cartesian[k, 0] = ref_x - n_pred * np.sin(ref_yaw)
            pred_cartesian[k, 1] = ref_y + n_pred * np.cos(ref_yaw)
            
            # 4. Calculate predicted Yaw
            pred_cartesian[k, 2] = (ref_yaw + e_pred + np.pi) % (2 * np.pi) - np.pi
            
            # Copy other stats for debugging if needed
            pred_cartesian[k, 3] = target_cartesian[idx, 3] 

        return pred_cartesian


    def _retransform_miqp_to_new_frame(self, prev_cartesian_miqp, new_frenet_ref, new_cartesian_ref):
        """Re-project previous MIQP Cartesian path into the current Frenet frame.
        
        For each new reference waypoint, finds the closest old MIQP Cartesian
        point and computes its lateral offset n in the new frame.
        """
        N = new_frenet_ref.shape[0]
        result = new_frenet_ref.copy()
        prev_xy = prev_cartesian_miqp[:, :2]

        for k in range(N):
            ref_idx = min(k, new_cartesian_ref.shape[0] - 1)
            ref_x = new_cartesian_ref[ref_idx, 0]
            ref_y = new_cartesian_ref[ref_idx, 1]
            ref_yaw = new_cartesian_ref[ref_idx, 2]

            # Closest old MIQP waypoint to this reference point
            dists = np.linalg.norm(prev_xy - np.array([ref_x, ref_y]), axis=1)
            closest = np.argmin(dists)

            dx = prev_cartesian_miqp[closest, 0] - ref_x
            dy = prev_cartesian_miqp[closest, 1] - ref_y
            n_new = -dx * np.sin(ref_yaw) + dy * np.cos(ref_yaw)

            # Clip to track bounds (col 6 = inner/max, col 7 = outer/min)
            n_new = np.clip(n_new, new_frenet_ref[k, 7], new_frenet_ref[k, 6])
            result[k, 1] = n_new

        return result

    def miqp_traj_generation(self, miqp_target_trajectory_frenet, target_trajectory_cartesian, ego_car_frenet, prev_path_n=None):
        """
        MIQP with Spatiotemporal Collision Checking, Stability Anchoring, and Soft Constraints.
        """
        start_time = time.perf_counter()

        # --- 0. Downsample for Speed (Half Resolution) ---
        s_vals_full = miqp_target_trajectory_frenet[:, 0]
        N_full = len(s_vals_full)
        indices = np.arange(0, N_full, 2)
        
        data_ds = miqp_target_trajectory_frenet[indices]
        s_vals = data_ds[:, 0]
        n_ref = data_ds[:, 1]
        n_max = data_ds[:, 6]
        n_min = data_ds[:, 7]
        N = len(s_vals)

        # Estimate dt for predictions (assuming constant velocity profile if not provided)
        # If speed info isn't available, default to 0.1s steps
        dt = mpc_cbf_ff_setup.mpc_params_dict['Tf'] / mpc_cbf_ff_setup.mpc_params_dict['N']
        if hasattr(self, 'dt'): dt = self.dt

        # --- 1. Gurobi Initialization ---
        try:
            m = gp.Model("Raceline_MIQP")
            m.setParam('OutputFlag', 0)
            m.setParam('MIPFocus', 3)       # Focus on finding feasible solutions
            m.setParam('MIPGap', 0.3)
            m.setParam('TimeLimit', 0.02)
            m.setParam('Heuristics', 0.05)
            m.setParam('PreSparsify', 0)
            m.setParam('Method', -1)
            m.setParam('Threads', 1)
            m.setParam('Cuts', 1)

            # --- 2. Variables ---
            d = m.addVars(N, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="d")
            
            # Adaptive warm start: Shift previous path to match current time
            shifted_prev = None
            if prev_path_n is not None:
                # Interpolate previous path to current s_vals for accuracy
                # or just shift if grids match
                shifted_prev = np.roll(prev_path_n, -1) if len(prev_path_n) == N_full else prev_path_n
                # Downsample for warm start
                shifted_prev_ds = shifted_prev[indices] if len(shifted_prev) == N_full else None
                
                if shifted_prev_ds is not None and len(shifted_prev_ds) == N:
                    for k in range(N):
                        d[k].Start = shifted_prev_ds[k]

            # --- 3. Constraints ---
            
            # A. Track Bounds (Hard)
            margin = 0.95
            m.addConstrs((d[k] >= n_min[k] + margin for k in range(N)), "BndMin")
            m.addConstrs((d[k] <= n_max[k] - margin for k in range(N)), "BndMax")


            # B. Ego Continuity (Soft)
            ego_n = np.clip(ego_car_frenet[1], n_min[0], n_max[0])
            m.addConstr(d[0] >= ego_n - 0.2)
            m.addConstr(d[0] <= ego_n + 0.2)

            # C. Dynamic Obstacle Avoidance (Soft & Time-Indexed)
            obj_obs_slack = 0.0
            obj_binary_stab = 0.0
            
            # Constants
            obs_w = mpc_cbf_ff_setup.constraints_dict.get('ellipse_width', 2.0)
            obs_h = mpc_cbf_ff_setup.constraints_dict.get('ellipse_height', 6.0)
            safe_lon = obs_h/2 + self.miqp_params['safety_margin_lon']
            safe_lat = obs_w/2 + self.miqp_params['safety_margin_lat']
            big_M = self.miqp_params.get('big_M', 20.0)

            if self.obs is not None and len(self.obs) > 0:
                # ---------- NEW CODE: obstacle clustering for binary consistency ----------

                num_obs = len(self.obs)

                # Extract obstacle longitudinal info
                obs_s0 = []
                obs_vs = []

                for obs in self.obs:
                    obs_s0.append(obs[0])
                    obs_vs.append(obs[2] if len(obs) > 2 else 0.0)

                # Prediction horizon length (account for downsampling)
                T_horizon = N * dt * 2

                # Build adjacency list for "close-in-s" obstacles
                adj = {i: set() for i in range(num_obs)}

                for i in range(num_obs):
                    for j in range(i + 1, num_obs):
                        delta_s_min = abs(obs_s0[i] - obs_s0[j]) - abs(obs_vs[i] - obs_vs[j]) * T_horizon
                        if delta_s_min < 2.0 * safe_lon:
                            adj[i].add(j)
                            adj[j].add(i)

                # Find connected components (clusters)
                clusters = []
                visited = set()

                for i in range(num_obs):
                    if i in visited:
                        continue
                    stack = [i]
                    comp = []
                    while stack:
                        u = stack.pop()
                        if u in visited:
                            continue
                        visited.add(u)
                        comp.append(u)
                        stack.extend(adj[u] - visited)
                    clusters.append(comp)

                # ---------- END NEW CODE ----------

                for i, obs in enumerate(self.obs):
                    obs_s_start = obs[0]
                    obs_n = obs[1]
                    # Attempt to get velocity, default to 0 if missing
                    obs_vs = obs[2] if len(obs) > 2 else 0.0
                    
                    # Create one binary variable per obstacle (Decision Consistency)
                    # We assume the decision (Left/Right) holds for the whole pass
                    # b_var = m.addVar(vtype=GRB.BINARY, name=f"b_{i}")
                    if i == 0:
                        b_vars = {}
                    b_var = m.addVar(vtype=GRB.BINARY, name=f"b_{i}")
                    b_vars[i] = b_var

                    # Binary Stability: Bias b_var towards previous path's side
                    if shifted_prev_ds is not None:
                        # Check "average" side of previous path relative to obstacle
                        avg_n_prev = np.mean(shifted_prev_ds)
                        target_b = 1.0 if avg_n_prev > obs_n else 0.0
                        # Soft penalty for switching sides
                        obj_binary_stab += 500.0 * (b_var - target_b)**2

               
                    # Iterate through time steps
                    for k in range(N):
                        # PROJECT OBSTACLE POSITION FORWARD
                        obs_s_k = obs_s_start + obs_vs * (k * dt * 2) # *2 because downsampled
                        ego_s_k = s_vals[k]

                        # CHECK COLLISION AT THIS TIME STEP
                        # If future ego and future obstacle are close longitudinally
                        if abs(ego_s_k - obs_s_k) < safe_lon:
                            
                            # Add slack for feasibility
                            sl = m.addVar(lb=0, name=f"sl_{i}_{k}")
                            obj_obs_slack += sl
                            
                            # Lateral constraints (Left/Right)
                            # b=1 (Left): d >= obs + safe
                            # b=0 (Right): d <= obs - safe
                            m.addConstr(d[k] <= (obs_n - safe_lat) + big_M * b_var + sl)
                            m.addConstr(d[k] >= (obs_n + safe_lat) - big_M * (1 - b_var) - sl)
        

            # ---------- NEW CODE: enforce binary consistency within clusters ----------

            for comp in clusters:
                if len(comp) <= 1:
                    continue
                base = comp[0]
                for j in comp[1:]:
                    m.addConstr(b_vars[j] == b_vars[base], name=f"b_cons_{base}_{j}")

            # ---------- END NEW CODE ----------

            # --- 4. Objective ---
            # Deviation
            obj_dev = gp.quicksum(self.miqp_params['deviation_weight'] * (d[k] - n_ref[k])**2 for k in range(N))
            
            # Smoothness
            obj_smooth = gp.quicksum(self.miqp_params['smoothness_weight'] * (d[k+1] - d[k])**2 for k in range(N-1))
            
            # Path Stability (anchor to previous solution)
            obj_path_stab = 0.0
            if shifted_prev_ds is not None:
                 obj_path_stab = gp.quicksum(self.miqp_params.get('stability_weight', 5.0) * (d[k] - shifted_prev_ds[k])**2 for k in range(N))

            # Total Objective
            m.setObjective(
                obj_dev + 
                obj_smooth + 
                obj_path_stab + 
                obj_binary_stab + 
                (self.miqp_params.get('obstacle_slack_penalty', 2000.0) * obj_obs_slack) +
                GRB.MINIMIZE
            )

            # --- 5. Solve ---
            m.optimize()
            
            if m.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
                miqp_path_n_ds = np.array([d[k].X for k in range(N)])
                miqp_success = True
            else:
                miqp_path_n_ds = n_ref
                miqp_success = False

        except Exception:
            miqp_path_n_ds = n_ref
            miqp_success = False

        # --- 6. Post Processing (Upsample) ---
        miqp_path_n = np.interp(s_vals_full, s_vals, miqp_path_n_ds)
        
        self.prev_miqp_path = miqp_path_n
        
        miqp_target_trajectory_frenet = miqp_target_trajectory_frenet.copy()
        miqp_target_trajectory_frenet[:, 1] = miqp_path_n
        
        miqp_path_frenet = np.zeros((len(miqp_path_n), 6))
        miqp_path_frenet[:, 0] = s_vals_full
        miqp_path_frenet[:, 1] = miqp_path_n
        
        miqp_path_cartesian = self.frenet_to_cartesian(miqp_path_frenet, target_trajectory_cartesian)

        self.last_miqp_computation_time = time.perf_counter() - start_time
        self.last_miqp_success = miqp_success
    
        return miqp_target_trajectory_frenet, miqp_path_cartesian