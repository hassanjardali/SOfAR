# Prevent Qt conflicts with opencv - unset opencv's Qt plugin path before any cv2 import
import os
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
os.environ['QT_PLUGIN_PATH'] = ''
# Tell opencv to not use Qt
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

from pathlib import Path
import sys
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend instead of Qt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
import time 
from datetime import datetime
from helpers.car_models import CarKinematicModel
from acados_controllers.mpc_cbf_ff_rl import KinematicModelMIQPMPCinFF
import yaml
import random
from helpers.sim_helpers import generate_moving_obstacles, update_obstacle_positions, compute_cte, create_periodic_splines, check_collision, check_boundary_violation, update_overtake_tracking, compute_s_obs_for_obstacles
padding_car = True
focus_on_car = False
GAMMA_BOUNDS = (0.05, 1.0)

def generate_target_trajectory(car, path_points, inner_boundaries_distances, outer_boundaries_distances, closest_idx, horizon, dt, params_simulator):
    """Professional spline-based trajectory generation for MPC.
    
    Solves the critical discontinuity issue at raceline start/end using:
    - Periodic cubic splines for smooth wrapping
    - Velocity-dependent trajectory planning
    - Curvature-aware velocity control
    - Professional spline interpolation
    """
    offset = 0 if params_simulator['raceline_path'] == './racelines/raceline_Monza.csv' or params_simulator['raceline_path'] == './racelines/raceline_LS.csv' else 2
    closest_idx = closest_idx + offset
    target_trajectory = []
    target_velocity = params_simulator['target_velocity']

    
    # PROFESSIONAL CONTROL PARAMETERS
    current_velocity = car.velocity
    
    # CRITICAL: Create periodic splines to solve discontinuity at start/end (cached)
    global trajectory_splines_x, trajectory_splines_y, trajectory_splines_curvature, trajectory_splines_x_deriv, trajectory_splines_y_deriv, trajectory_splines_inner_dist, trajectory_splines_outer_dist, trajectory_splines_velocity, cached_path_points, spline_path_length
    if 'trajectory_splines_x' not in globals() or 'cached_path_points' not in globals() or not np.array_equal(cached_path_points, path_points):
        trajectory_splines_x, trajectory_splines_y, trajectory_splines_curvature = create_periodic_splines(path_points)
        # Pre-compute derivative functions
        trajectory_splines_x_deriv = trajectory_splines_x.derivative()
        trajectory_splines_y_deriv = trajectory_splines_y.derivative()
        # Create periodic splines for inner and outer distances
        from scipy.interpolate import CubicSpline
        n_points = len(inner_boundaries_distances)
        # CRITICAL: Ensure periodic condition: first and last values must match perfectly
        inner_dist_periodic = inner_boundaries_distances.copy()
        outer_dist_periodic = outer_boundaries_distances.copy()
        # Force perfect closure by averaging first and last
        inner_dist_periodic[0] = inner_dist_periodic[-1] = (inner_dist_periodic[0] + inner_dist_periodic[-1]) / 2.0
        outer_dist_periodic[0] = outer_dist_periodic[-1] = (outer_dist_periodic[0] + outer_dist_periodic[-1]) / 2.0
        t = np.linspace(0, 1, n_points, endpoint=False)
        trajectory_splines_inner_dist = CubicSpline(t, inner_dist_periodic, bc_type='periodic')
        trajectory_splines_outer_dist = CubicSpline(t, outer_dist_periodic, bc_type='periodic')
        # Create periodic spline for velocity if raceline has velocity data
        if path_points.shape[1] >= 3:
            velocity_periodic = path_points[:, 2].copy()
            velocity_periodic[0] = velocity_periodic[-1] = (velocity_periodic[0] + velocity_periodic[-1]) / 2.0
            trajectory_splines_velocity = CubicSpline(t, velocity_periodic, bc_type='periodic')
        else:
            trajectory_splines_velocity = None
        cached_path_points = path_points.copy()
        # Store spline path length for parameter conversion
        spline_path_length = len(path_points)
    spline_x, spline_y, spline_curvature = trajectory_splines_x, trajectory_splines_y, trajectory_splines_curvature
    spline_x_deriv, spline_y_deriv = trajectory_splines_x_deriv, trajectory_splines_y_deriv
    spline_inner_dist, spline_outer_dist = trajectory_splines_inner_dist, trajectory_splines_outer_dist
    spline_velocity = trajectory_splines_velocity if 'trajectory_splines_velocity' in globals() else None
    
    # CRITICAL: Use spline_path_length (from closed path) for parameter conversion
    # This ensures we use the correct number of points that the spline was created with
    if 'spline_path_length' not in globals():
        spline_path_length = len(path_points)
    path_length_for_spline = spline_path_length
            
    # Pre-compute all spline parameters once (vectorized)
    distance_per_step = current_velocity * dt
    points_per_step = distance_per_step / 2.0
    
    # Vectorized cumulative distance computation
    i_range = np.arange(horizon + 1)
    # cumulative_distances = points_per_step * i_range * (1.0 + 0.025 * np.maximum(i_range - 1, 0))
    # cumulative_distances[0] = 0  # Fix first element
    min_spatial_step = 1.5  # meters
    distance_per_step = max(current_velocity * dt, min_spatial_step)
    cumulative_distances = distance_per_step * i_range / 2
    
    # CRITICAL: Convert path index to spline parameter (0 to 1)
    # Use modulo to handle wrapping, and normalize by path_length_for_spline
    # The spline parameter goes from 0 to 1, where 0 and 1 are the same point (periodic)
    spline_parameters = (closest_idx + cumulative_distances) / path_length_for_spline
    spline_parameters = spline_parameters % 1.0
    
    # Vectorized spline evaluation (much faster!)
    spline_params_array = spline_parameters
    target_xs = spline_x(spline_params_array)
    target_ys = spline_y(spline_params_array)
    dx_dts = spline_x_deriv(spline_params_array)
    dy_dts = spline_y_deriv(spline_params_array)
    raw_yaws = np.arctan2(dy_dts, dx_dts)
    
    # Vectorized curvature computation
    curvatures = np.array([spline_curvature(s) for s in spline_parameters])
    
    # Vectorized inner and outer distances evaluation
    inner_dists = spline_inner_dist(spline_params_array)
    outer_dists = spline_outer_dist(spline_params_array)
    
    # Vectorized angle unwrapping
    yaws_unwrapped = np.zeros_like(raw_yaws)
    yaws_unwrapped[0] = raw_yaws[0]
    for i in range(1, len(raw_yaws)):
        angle_diff = raw_yaws[i] - raw_yaws[i-1]
        angle_diff = ((angle_diff + np.pi) % (2 * np.pi)) - np.pi
        yaws_unwrapped[i] = yaws_unwrapped[i-1] + angle_diff
    
    # Vectorized velocity planning
    prediction_times = i_range * dt
    
    # Initialize target velocities
    use_raceline_velocity = params_simulator.get('use_raceline_velocity', False)
    if use_raceline_velocity and spline_velocity is not None:
        # Use raceline velocity from spline, scaled by difficulty factor
        velocity_difficulty = params_simulator.get('use_raceline_velocity_difficulty', 1.0)
        target_vels = spline_velocity(spline_params_array) * velocity_difficulty
    else:
        # Use constant target velocity
        target_vels = np.full_like(prediction_times, target_velocity)
    
    # Compute feedforward delta from curvature: delta_ff = arctan(L * kappa)
    L = car.wheel_base
    target_deltas = np.arctan(L * curvatures)
    # target_deltas = np.zeros_like(prediction_times)
    # Build trajectory array
    target_trajectory = np.column_stack([target_xs, target_ys, raw_yaws, target_vels, target_deltas, curvatures, inner_dists, outer_dists]) 
    
    return np.array(target_trajectory)

def shortest_distance_to_boundary(point, boundary):
    """Calculate the shortest distance from a point to a boundary."""
    point_2d = point[:2] if len(point) > 2 else point
    differences = boundary - point_2d
    distances = np.linalg.norm(differences, axis=1)
    min_idx = np.argmin(distances)
    return distances[min_idx], boundary[min_idx]

def animate_simulation(car, path, boundaries, mpc_controller, params_simulator, save_data_csv=False):
    """Animate simulation with real-time visualization.
    
    Args:
        car: CarKinematicModel instance
        path: Raceline dataframe
        boundaries: Boundaries dataframe
        mpc_controller: MPC controller instance
        params_simulator: Simulator parameters dict
        save_data_csv: If True, save all summary plot data to CSV (default: False)
    """
    N_pred = mpc_controller.T_horizon  # MPC prediction horizon 

    # Create a figure with GridSpec for the layout
    if save_data_csv:
        # Only show main visualization when saving CSV
        fig = plt.figure(figsize=(16, 12))
        ax1 = fig.add_subplot(111)  # Single subplot taking entire figure
        ax2 = ax3 = ax4 = ax5 = ax6 = ax7 = None  # No other plots
    else:
        # Full visualization with all plots
        if mpc_controller.obstacles_on:
            fig = plt.figure(figsize=(24, 10))  # Adjust the figure size as needed
            gs = GridSpec(3, 5, height_ratios=[4, 1, 1])  # 3 rows, 5 columns - larger first row for raceline/car/obstacles
        else:
            fig = plt.figure(figsize=(24, 8))  # Adjust the figure size as needed
            gs = GridSpec(2, 5, height_ratios=[4, 1])  # 2 rows, 5 columns - larger first row for raceline/car
        ax1 = fig.add_subplot(gs[0, :])

    path_points = path.to_numpy()
    boundaries_points = boundaries.to_numpy()
    inner_boundaries_points = boundaries_points[:, :2]
    outer_boundaries_points = boundaries_points[:, 2:]

    # Calculate the distance from each point in path to the inner and outer boundaries, put in global variables, do it once only
    global inner_boundaries_distances, outer_boundaries_distances
    inner_boundaries_distances = np.zeros(len(path_points))
    outer_boundaries_distances = np.zeros(len(path_points))
    for i in range(len(path_points)):
        inner_boundaries_distances[i], _ = shortest_distance_to_boundary(path_points[i], inner_boundaries_points)
        outer_boundaries_distances[i], _ = shortest_distance_to_boundary(path_points[i], outer_boundaries_points)

    # Check if obstacles should be static or dynamic
    static_obstacles_flag = params_simulator['obstacles']['static_obstacles']
    max_velocity = params_simulator['obstacles']['max_velocity']
    random_obstacle_position = params_simulator['obstacles'].get('random_obstacle_position', False)
    num_obs = params_simulator['obstacles']['num_obstacles']
    target_velocity = params_simulator['target_velocity']
    
    # Check if random obstacle position is enabled
    if random_obstacle_position:
        # Generate random obstacle indices from raceline
        obstacle_indices = random.sample(range(len(path_points)), min(num_obs, len(path_points)))
        # Generate random n_deviation from -2 to +2
        n_deviation = [random.uniform(-2.0, 2.0) for _ in range(num_obs)]
        print(f"Using RANDOM obstacle positions: {obstacle_indices}")
        print(f"Using RANDOM n_deviation: {n_deviation}")
    else:
        # Use obstacle_indices from config - ensures reproducibility
        obstacle_indices = params_simulator['obstacles'].get('obstacle_indices', None)
        # Use n_deviation from config - ensures reproducibility
        n_deviation = params_simulator['obstacles'].get('n_deviation', None)
        if obstacle_indices is not None:
            num_obs = len(obstacle_indices)
            print(f"Using FIXED obstacle positions from config: {obstacle_indices}")
            print(f"Using FIXED n_deviation from config: {n_deviation}")
        else:
            print("WARNING: No obstacle_indices specified in config and random_obstacle_position is False!")
    
    if static_obstacles_flag:
        # Static obstacles: set velocities to 0
        obstacle_velocities = [0.0] * num_obs
        print("Using STATIC obstacles (velocities = 0)")
    else:
        # Dynamic obstacles
        use_raceline_velocity = params_simulator.get('use_raceline_velocity', False)
        if use_raceline_velocity and path_points.shape[1] >= 3:
            # Use raceline velocity at obstacle positions (works for both fixed and random positions)
            # Apply difficulty factor: 1.0 = same as raceline, < 1.0 = slower (easier)
            difficulty = params_simulator['obstacles'].get('difficulty', 1.0)
            obstacle_velocities = [path_points[idx, 2] * difficulty for idx in obstacle_indices]
            print(f"Using DYNAMIC obstacles with RACELINE velocities (difficulty={difficulty:.2f}): {[f'{v:.1f}' for v in obstacle_velocities]} m/s")
        elif random_obstacle_position:
            # Random velocities from 0 to target_velocity - 10
            max_vel = max(0.0, target_velocity - 10.0)
            obstacle_velocities = [random.uniform(0.0, max_vel) for _ in range(num_obs)]
            print(f"Using DYNAMIC obstacles with RANDOM velocities (0 to {max_vel:.1f} m/s)")
        else:
            # Use max_velocity for all obstacles
            obstacle_velocities = [max_velocity for _ in range(num_obs)]
            print(f"Using DYNAMIC obstacles (max velocity = {max_velocity} m/s)")
    
    # Generate obstacles
    obstacles = generate_moving_obstacles(
        path_points, 
        num_obs, 
        params_simulator['obstacles']['circle_radius'],
        obstacle_velocities,
        obstacle_indices,
        n_deviation
    )
    
    # Store obstacles globally for controller access
    global static_obstacles, global_obstacles, static_obstacles_flag_global, global_n_deviation
    static_obstacles = obstacles
    global_obstacles = obstacles
    static_obstacles_flag_global = static_obstacles_flag
    global_n_deviation = n_deviation
    
    # Print obstacle data for verification
    obs_type = "STATIC" if static_obstacles_flag else "DYNAMIC"
    print(f"Generated {len(obstacles)} {obs_type.lower()} obstacles:")
    for i, obstacle in enumerate(obstacles):
        print(f"Obstacle {i+1}: Position=({obstacle[0]:.2f}, {obstacle[1]:.2f}), Radius={obstacle[2]:.2f}m, Velocity={obstacle[3]:.2f}m/s")

    # Plot raceline on the main subplot
    # ax1.plot(path_points[:, 0], path_points[:, 1], 'r--', label='Raceline')
    raceline_line, = ax1.plot(path_points[:, 0], path_points[:, 1], 'r--', label='Raceline')
    boundary_inner_line, = ax1.plot(boundaries_points[:, 0], boundaries_points[:, 1], 'g--', label='Boundary Inner', linewidth=3.0)
    boundary_outer_line, = ax1.plot(boundaries_points[:, 2], boundaries_points[:, 3], 'g--', label='Boundary Outer', linewidth=3.0)

    # Create obstacle patches and store them
    obstacle_patches = []
    # Create obstacle center markers (line plots for blit support)
    obstacle_centers, = ax1.plot([], [], 'o', c='darkred', markersize=5, alpha=0.8, 
                                  label='Obstacle Centers', zorder=5)
    
    # Get obstacle shape from config
    obstacle_shape = params_simulator['obstacles'].get('obstacle_shape', 'circle')
    
    if len(obstacles) > 0:
        obstacle_x = obstacles[:, 0]
        obstacle_y = obstacles[:, 1]
        
        # Plot obstacles based on shape
        for i, obs_data in enumerate(obstacles):
            x, y = obs_data[0], obs_data[1]
            label_text = f'{obs_type} Obstacle' if i == 0 else ""
            
            if obstacle_shape == 'ellipse':
                ellipse_width = params_simulator['obstacles'].get('ellipse_width', 0.5)
                ellipse_height = params_simulator['obstacles'].get('ellipse_height', 1.5)
                # Get heading from raceline at obstacle's path index
                path_idx = int(obs_data[4]) if len(obs_data) > 4 else 0
                next_idx = (path_idx + 1) % len(path_points)
                dx = path_points[next_idx, 0] - path_points[path_idx, 0]
                dy = path_points[next_idx, 1] - path_points[path_idx, 1]
                heading_rad = np.arctan2(dy, dx)
                heading_deg = np.degrees(heading_rad)
                # width = longitudinal (along raceline), height = lateral (perpendicular to raceline)
                ellipse = Ellipse((x, y), ellipse_height, ellipse_width, angle=heading_deg, color='red', alpha=0.7, label=label_text)
                ellipse.is_obstacle = True
                ellipse.path_idx = path_idx  # Store path index for angle updates
                ax1.add_patch(ellipse)
                obstacle_patches.append(ellipse)
            else:  # circle
                r = obs_data[2]
                circle = plt.Circle((x, y), r, color='red', alpha=0.7, label=label_text)
                circle.is_obstacle = True
                ax1.add_patch(circle)
                obstacle_patches.append(circle)
    car_path, = ax1.plot([], [], 'b-', label='Car Path')
    # Create car as an ellipse (larger visualization)
    car_width = 3.0  # Lateral width in meters
    car_length = 6.0  # Longitudinal length in meters
    car_ellipse = Ellipse((car.x, car.y), car_length, car_width, angle=np.degrees(car.yaw), 
                         color='blue', alpha=0.8, zorder=10, label='Car')
    ax1.add_patch(car_ellipse)
    orientation_quiver = ax1.quiver(car.x, car.y, np.cos(car.yaw), np.sin(car.yaw), scale=50, color='blue')
    
    # Add a plot for target trajectory (always shown)
    target_trajectory_line, = ax1.plot([], [], 'g--', label='Target Trajectory', alpha=0.7)
    
    # Add a plot for MIQP trajectory
    miqp_trajectory_line, = ax1.plot([], [], 'c:', label='Target Trajectory MIQP (Cartesian)', linewidth=2, alpha=0.7)

    # Initialize predicted orientation quiver with zeros
    if N_pred > 0: 
        x_pred_initial = np.zeros(N_pred + 1)  # +1 because horizon includes current step
        y_pred_initial = np.zeros(N_pred + 1)
        u_pred_initial = np.zeros(N_pred + 1)
        v_pred_initial = np.zeros(N_pred + 1)
        predicted_orientation_quiver = ax1.quiver(
            x_pred_initial, y_pred_initial, u_pred_initial, v_pred_initial,
            scale=80, color='cyan', label='Predicted Orientation'
        )
        # Add a plot for predicted positions
        predicted_line, = ax1.plot([], [], 'c-', label='Predicted Path')


    # Set axis limits and labels for raceline plot
    if not focus_on_car:
        ax1.set_xlim(min(path_points[:, 0]) - 10, max(path_points[:, 0]) + 10)
        ax1.set_ylim(min(path_points[:, 1]) - 10, max(path_points[:, 1]) + 10)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax1.set_title("Real-Time Car Simulation using Kinematic Model and MPC Controller")
    ax1.set_aspect('equal')  # Ensure circles appear as circles
    
    # Add time display text element
    time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=28,
                         verticalalignment='top', horizontalalignment='left',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                         fontweight='bold')

    # Lower row for velocity, steering angle, cross-track error, lateral acceleration, and gamma (side-by-side plots)
    # Only create these plots if not saving CSV
    if not save_data_csv:
        ax2 = fig.add_subplot(gs[1, 0])  # First column for velocity
        ax3 = fig.add_subplot(gs[1, 1])  # Second column for steering angle
        ax4 = fig.add_subplot(gs[1, 2])  # Third column for cross-track error
        ax6 = fig.add_subplot(gs[1, 3])  # Fourth column for lateral acceleration
        ax7 = fig.add_subplot(gs[1, 4])  # Fifth column for gamma
    else:
        ax2 = ax3 = ax4 = ax6 = ax7 = None

    # Data lists for velocity, steering angle, cross-track error (CTE), lateral acceleration, and gamma
    time_data = []
    current_velocity_data = []
    target_velocity_data = []
    steering_angle_data = []
    cte_data = []
    alat_data = []
    gamma_data = []
    gamma_data_full = []  # Backup that never gets cleared for summary plot
    average_curvature_data = []
    average_curvature_data_full = []  # Backup that never gets cleared for summary plot
    h_values_history = []
    h_values_history_full = []  # Backup that never gets cleared for summary plot

    # Initialize plot lines (only if not saving CSV)
    velocity_line = target_velocity_line = steering_angle_line = cte_line = alat_line = gamma_line = None
    h_lines = []
    ax5 = None
    
    if not save_data_csv:
        # Velocity plot
        velocity_line, = ax2.plot([], [], 'b-', label='Current Velocity (m/s)')
        target_velocity_line, = ax2.plot([], [], 'r-', label='Target Velocity (m/s)')
        ax2.set_xlim(0, params_simulator['max_time'])
        ax2.set_ylim(0, max(path_points[:, 2]) + 25)  # Adjust y-limit based on target velocity
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity (m/s)")
        ax2.legend()
        ax2.set_title("Velocity Tracking")

        # Steering angle plot
        steering_angle_line, = ax3.plot([], [], 'g-', label='Steering Angle (rad)')
        ax3.set_xlim(0, params_simulator['max_time'])
        ax3.set_ylim(-0.1, 0.1)  
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Steering Angle (rad)")
        ax3.legend()
        ax3.set_title("Steering Angle")

        # Cross-track error plot
        cte_line, = ax4.plot([], [], 'm-', label='Cross-Track Error (m)')
        ax4.set_xlim(0, params_simulator['max_time'])
        ax4.set_ylim(-2, 2)  
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("CTE (m)")
        ax4.legend()
        ax4.set_title("Cross-Track Error")

        # Lateral acceleration plot
        max_alat = mpc_controller.max_alat
        alat_line, = ax6.plot([], [], 'orange', label='Lateral Acceleration (m/s²)', linewidth=2)
        ax6.axhline(y=max_alat, color='r', linestyle='--', alpha=0.7, label=f'Upper Limit ({max_alat:.1f} m/s²)')
        ax6.axhline(y=-max_alat, color='r', linestyle='--', alpha=0.7, label=f'Lower Limit (-{max_alat:.1f} m/s²)')
        ax6.set_xlim(0, params_simulator['max_time'])
        ax6.set_ylim(-max_alat - 2, max_alat + 2)
        ax6.set_xlabel("Time (s)")
        ax6.set_ylabel("a_lat (m/s²)")
        ax6.legend()
        ax6.set_title("Lateral Acceleration")
        ax6.grid(True, alpha=0.3)

        # Gamma (CBF parameter) plot
        gamma_line, = ax7.plot([], [], 'purple', label='Gamma (CBF Parameter)', linewidth=2)
        if hasattr(mpc_controller, 'gamma_default'):
            gamma_default = mpc_controller.gamma_default
            ax7.axhline(y=gamma_default, color='gray', linestyle='--', alpha=0.7, label=f'Default ({gamma_default:.3f})')
        ax7.axhline(y=GAMMA_BOUNDS[0], color='r', linestyle=':', alpha=0.5, label=f'Min ({GAMMA_BOUNDS[0]:.3f})')
        ax7.axhline(y=GAMMA_BOUNDS[1], color='r', linestyle=':', alpha=0.5, label=f'Max ({GAMMA_BOUNDS[1]:.3f})')
        ax7.set_xlim(0, params_simulator['max_time'])
        ax7.set_ylim(GAMMA_BOUNDS[0] - 0.1, GAMMA_BOUNDS[1] + 0.1)
        ax7.set_xlabel("Time (s)")
        ax7.set_ylabel("Gamma")
        ax7.legend()
        ax7.set_title("CBF Parameter (Gamma)")
        ax7.grid(True, alpha=0.3)

        # H values plot (only if obstacles are enabled)
        h_window_size = 10.0  # Show last 10 seconds in rolling window
        if mpc_controller.obstacles_on and len(obstacles) > 0:
            ax5 = fig.add_subplot(gs[2, :])  # Third row, spans all columns
            num_obs = len(obstacles)
            for i in range(num_obs):
                h_line, = ax5.plot([], [], label=f'Obstacle {i+1} h-value', linewidth=2)
                h_lines.append(h_line)
            ax5.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Safety Threshold (h=0)')
            ax5.set_xlim(0, min(h_window_size, params_simulator['max_time']))
            ax5.set_xlabel("Time (s)")
            ax5.set_ylabel("h value")
            ax5.legend()
            ax5.set_title("Control Barrier Function (CBF) Values")
            ax5.grid(True, alpha=0.3)

    # History of car path (limit for performance)
    history = []
    max_history_points = 200  # Limit history for performance
    
    # Initialize tracking variables
    mpc_failure_count = 0
    computation_times = []
    miqp_failure_count = 0
    miqp_computation_times = []
    rl_computation_times = []
    collision_count = 0
    out_of_bounds_count = 0
    h_violation_count = 0
    car_in_bounds = None  # Track current boundary state (None = uninitialized)
    car_radius = 0.75  # Approximate car radius in meters
    prev_gamma = None  # Track previous gamma to detect changes
    # Track previous boundary state for jump detection
    prev_car_dist_to_path = None
    prev_in_bounds = None
    prev_ignored_jump = False  # Track if previous frame had an ignored jump
    obstacle_shape = params_simulator['obstacles'].get('obstacle_shape', 'circle')
    ellipse_width = params_simulator['obstacles'].get('ellipse_width', 2.0) if obstacle_shape == 'ellipse' else None
    ellipse_height = params_simulator['obstacles'].get('ellipse_height', 6.0) if obstacle_shape == 'ellipse' else None
    freeze_on_violation = params_simulator.get('freeze_on_violation', True)  # Freeze animation on violation/collision
    h_values_frame_offset = 0  # Track frame offset when h_values_history is cleared
    
    # Overtake tracking: track state for each obstacle
    # State: 0 = not started, 1 = tracking (s >= 20), 2 = passed 0 (s < 0), 3 = completed (s <= -10)
    # Only count overtake if full sequence: s >= 20 -> s < 0 -> s <= -10 (tight racing threshold)
    num_obstacles = len(obstacles)
    overtake_states = np.zeros(num_obstacles, dtype=int)  # Track state for each obstacle
    prev_s_obs = np.full(num_obstacles, np.nan)  # Track previous s_obs for each obstacle
    successful_overtakes = 0  # Total count of successful overtakes
    animation_frozen = False  # Flag to freeze animation on track violation
    last_return_value = None  # Store last return value for frozen animation
    animation_obj = [None]  # Store animation object to allow stopping from nested function
    
    # Distance tracking for lap calculation
    total_distance_traversed = 0.0  # Total distance traveled in meters
    prev_car_x = car.x  # Previous car position for distance calculation
    prev_car_y = car.y
    prev_car_velocity = car.velocity  # Previous car velocity for acceleration calculation
    
    # Frame-by-frame data for CSV export
    frame_data_list = [] if save_data_csv else None

    def update(frame):
        nonlocal mpc_failure_count, computation_times, miqp_failure_count, miqp_computation_times, rl_computation_times, collision_count, out_of_bounds_count, h_violation_count, car_in_bounds, overtake_states, prev_s_obs, successful_overtakes, prev_car_dist_to_path, prev_in_bounds, prev_ignored_jump, animation_frozen, last_return_value, h_values_frame_offset, total_distance_traversed, prev_car_x, prev_car_y, prev_car_velocity, prev_gamma, frame_data_list
        
        # Freeze animation if track violation occurred
        if animation_frozen:
            if last_return_value is not None:
                return last_return_value
            return ()
        global last_closest_idx, obstacles, static_obstacles, global_obstacles, steering_angle  # Global variables
        if 'last_closest_idx' not in globals():
            last_closest_idx = 0
        if 'obstacles' not in globals():
            obstacles = static_obstacles  # Use static obstacles as fallback
        else:
            obstacles = global_obstacles
        if 'steering_angle' not in globals():
            steering_angle = 0.0  # Initialize steering angle
        a_lat = 0.0  # Initialize lateral acceleration
        avg_curvature = 0.0  # Initialize average curvature
        gamma = None  # Initialize gamma
        start_time = time.time()
        timing = {}
        predicted_positions = None
        h_values = None
        target_trajectory_cartesian_miqp = None
        
        # Update obstacle positions along the raceline
        t0 = time.time()
        if 'static_obstacles_flag_global' in globals() and not static_obstacles_flag_global:
            # Only update obstacle positions if they are dynamic
            n_dev = global_n_deviation if 'global_n_deviation' in globals() else None
            obstacles = update_obstacle_positions(obstacles, path_points, params_simulator['dt'], n_dev)
            
            # Update obstacle velocities based on raceline velocity at their current position
            use_raceline_velocity = params_simulator.get('use_raceline_velocity', False)
            if use_raceline_velocity and path_points.shape[1] >= 3:
                difficulty = params_simulator['obstacles'].get('difficulty', 1.0)
                for i in range(len(obstacles)):
                    path_idx = int(obstacles[i, 4])  # Get current path index
                    obstacles[i, 3] = path_points[path_idx, 2] * difficulty  # Update velocity
            
            static_obstacles = obstacles  # Update global obstacles for controller
            global_obstacles = obstacles  # Update global obstacles for visualization
        timing['obstacle_update'] = (time.time() - t0) * 1000

        # Determine the search range based on the last closest index
        t0 = time.time()
        search_start = max(last_closest_idx - params_simulator['search_window'], 0)
        search_end = min(last_closest_idx + params_simulator['search_window'], len(path_points))

        # Calculate distances within the search range only
        distances = np.linalg.norm(path_points[search_start:search_end, :2] - np.array([car.x, car.y]), axis=1)
        closest_idx = search_start + np.argmin(distances)  # Get the index within the entire path

        if min(distances) > 10:
            closest_idx = np.argmin(np.linalg.norm(path_points[:, :2] - np.array([car.x, car.y]), axis=1))


        # Update last closest index for the next frame
        last_closest_idx = closest_idx
        timing['find_closest'] = (time.time() - t0) * 1000

        # Extract a subset of path points to send to the controller
        t0 = time.time()
        # Wrap the start and end indices for a closed-loop path
        total_points = len(path_points)
        start_idx = (closest_idx - 1) % total_points
        end_idx = (closest_idx + params_simulator['num_points']) % total_points

        # Compute cross-track error (CTE)
        t0 = time.time()
        cte = compute_cte(car, path_points, closest_idx)
        timing['cte_compute'] = (time.time() - t0) * 1000

        # Generate target trajectory for MPC
        t0 = time.time()
        target_trajectory = generate_target_trajectory(car, path_points, inner_boundaries_distances, outer_boundaries_distances, closest_idx, mpc_controller.miqp_params['planning_horizon'], params_simulator['dt'], params_simulator)
        # target_trajectory = generate_target_trajectory(car, path_points, inner_boundaries_distances, outer_boundaries_distances, closest_idx, mpc_controller.T_horizon, mpc_controller.Ts, params_simulator)
        timing['trajectory_gen'] = (time.time() - t0) * 1000
        
        # Calculate average curvature for target trajectory (curvatures are at column index 5)
        avg_curvature = np.mean(target_trajectory[:, 5]) if target_trajectory.shape[0] > 0 else 0.0
        
        # Track overtakes: compute s_obs for each obstacle using arc length along full raceline
        # Overtake sequence: obstacle >= 20m ahead -> passes ego (s < 0) -> >= 10m behind (s <= -10) for tight racing
        if len(obstacles) > 0:
            s_obs_values = compute_s_obs_for_obstacles(obstacles, car, path_points, closest_idx)
            overtake_states, prev_s_obs, successful_overtakes = update_overtake_tracking(
                obstacles, s_obs_values, overtake_states, prev_s_obs, successful_overtakes
            )
            
            # Move obstacles that have completed overtake (state 3) to prevent re-overtaking
            # Use -10m threshold for tight racing (moved from -20m to count sooner)
            # Place them far ahead (200 points ahead) so they won't re-overtake
            ahead_distance = 200  # Points ahead
            completion_threshold = -2.0  # Count overtake at -2m for tight racing (was -20m)
            n_dev = global_n_deviation if 'global_n_deviation' in globals() else None
            
            for obs_idx in range(len(obstacles)):
                # Move obstacles that have completed the overtake (state 3)
                # Also move obstacles in state 2 that are behind threshold to prevent comeback
                should_move = False
                if obs_idx < len(overtake_states):
                    if overtake_states[obs_idx] == 3:
                        # Completed overtake - move it
                        should_move = True
                    elif overtake_states[obs_idx] == 2 and s_obs_values[obs_idx] <= completion_threshold:
                        # In state 2 (passed) and behind threshold - count and move to prevent comeback
                        # Manually count this overtake since state machine might not have transitioned yet
                        if s_obs_values[obs_idx] <= completion_threshold:
                            # Count the overtake (obstacle was in state 2, meaning it already passed)
                            successful_overtakes += 1
                            overtake_states[obs_idx] = 3
                            should_move = True
                
                if should_move and s_obs_values[obs_idx] <= completion_threshold:
                    # Move obstacle far ahead to prevent re-overtaking
                    new_path_idx = (closest_idx + ahead_distance) % len(path_points)
                    
                    # Get n_deviation for this obstacle (preserve if it exists)
                    n_dev_obs = 0.0
                    if n_dev is not None:
                        if isinstance(n_dev, (list, np.ndarray)) and obs_idx < len(n_dev):
                            n_dev_obs = n_dev[obs_idx]
                        elif isinstance(n_dev, (int, float)):
                            n_dev_obs = float(n_dev)
                    
                    # If no n_deviation from config, try to compute from current position
                    if abs(n_dev_obs) < 1e-6 and obstacles.shape[1] > 4:
                        current_path_idx = int(obstacles[obs_idx, 4])
                        obs_x, obs_y = obstacles[obs_idx, 0], obstacles[obs_idx, 1]
                        path_x, path_y = path_points[current_path_idx, 0], path_points[current_path_idx, 1]
                        next_idx = (current_path_idx + 1) % len(path_points)
                        dx = path_points[next_idx, 0] - path_x
                        dy = path_points[next_idx, 1] - path_y
                        tangent_length = np.sqrt(dx*dx + dy*dy)
                        if tangent_length > 1e-6:
                            tangent = np.array([dx / tangent_length, dy / tangent_length])
                            normal = np.array([-tangent[1], tangent[0]])
                            rel_pos = np.array([obs_x - path_x, obs_y - path_y])
                            n_dev_obs = np.dot(rel_pos, normal)
                    
                    # Update obstacle position to new path index
                    new_x = path_points[new_path_idx, 0]
                    new_y = path_points[new_path_idx, 1]
                    
                    # Apply n_deviation if it exists
                    if abs(n_dev_obs) > 1e-6:
                        next_idx = (new_path_idx + 1) % len(path_points)
                        dx = path_points[next_idx, 0] - new_x
                        dy = path_points[next_idx, 1] - new_y
                        tangent_length = np.sqrt(dx*dx + dy*dy)
                        if tangent_length > 1e-6:
                            tangent = np.array([dx / tangent_length, dy / tangent_length])
                            normal = np.array([-tangent[1], tangent[0]])
                            new_x += n_dev_obs * normal[0]
                            new_y += n_dev_obs * normal[1]
                    
                    # Update obstacle: [x, y, radius, velocity, path_index]
                    obstacles[obs_idx, 0] = new_x
                    obstacles[obs_idx, 1] = new_y
                    if obstacles.shape[1] > 4:
                        obstacles[obs_idx, 4] = new_path_idx
                    else:
                        # Extend obstacle array if needed (add path_index column)
                        path_indices = np.full(len(obstacles), new_path_idx, dtype=float)
                        obstacles = np.column_stack([obstacles, path_indices])
                        obstacles[obs_idx, 4] = new_path_idx
                    
                    # Reset overtake state for this obstacle since it's been moved
                    if obs_idx < len(overtake_states):
                        overtake_states[obs_idx] = 0
                        prev_s_obs[obs_idx] = np.nan
            
            # Update global obstacles after moving them
            global_obstacles = obstacles
            static_obstacles = obstacles
        
        # Current state vector [x, y, yaw, velocity]
        current_state = np.array([car.x, car.y, car.yaw, car.velocity, steering_angle])
        
        # Get optimal control from MPC
        t0 = time.time()
        mpc_solve_time = 0.0
        try:
            if mpc_controller.obstacles_on:
                # Extract obstacle data in correct format [x, y, radius] for MPC
                obs_for_mpc = obstacles[:, [0, 1, 2, 3]]  # Take only x, y, radius, velocity columns
                # result = mpc_controller.get_control_input(current_state, target_trajectory, obs_for_mpc, params_simulator['dt'])
                result = mpc_controller.get_control_input(current_state, target_trajectory, obs_for_mpc)
            else:
                # result = mpc_controller.get_control_input(current_state, target_trajectory, None, params_simulator['dt'])
                result = mpc_controller.get_control_input(current_state, target_trajectory)

            
            # Track MIQP stats if available
            if hasattr(mpc_controller, 'last_miqp_computation_time'):
                miqp_computation_times.append(mpc_controller.last_miqp_computation_time)
                mpc_solve_time = time.time() - t0 - mpc_controller.last_miqp_computation_time - mpc_controller.last_rl_computation_time
                if not mpc_controller.last_miqp_success:
                    miqp_failure_count += 1
            else:
                mpc_solve_time = time.time() - t0
            
            # Track RL inference time if available
            if hasattr(mpc_controller, 'last_rl_computation_time'):
                rl_computation_times.append(mpc_controller.last_rl_computation_time)
            timing['mpc_solve'] = mpc_solve_time * 1000
            computation_times.append(mpc_solve_time)
            # Check if MPC solve time exceeds dt (critical for real-time control)
            if mpc_solve_time > params_simulator['dt']:
                print(f"WARNING: MPC solve time ({mpc_solve_time*1000:.2f}ms) exceeds dt ({params_simulator['dt']*1000:.2f}ms)! "
                      f"Control computed at t={frame*params_simulator['dt']:.3f}s will be applied with delay. "
                      f"Consider reducing MPC horizon or complexity.")
            
            # Unpack result: [throttle, steering_angle], predicted_trajectory, h_values, mpc_success, target_trajectory_cartesian_miqp, gamma
            if len(result) == 6:
                u_optimal, predicted_trajectory_mpc, h_values, mpc_success, target_trajectory_cartesian_miqp, gamma = result
            elif len(result) == 5:
                u_optimal, predicted_trajectory_mpc, h_values, mpc_success, target_trajectory_cartesian_miqp = result
                gamma = None
            elif len(result) == 4:
                u_optimal, predicted_trajectory_mpc, h_values, mpc_success = result
                target_trajectory_cartesian_miqp = None
                gamma = None
            elif len(result) == 3:
                u_optimal, predicted_trajectory_mpc, h_values = result
                mpc_success = True
                target_trajectory_cartesian_miqp = None
                gamma = None
            elif len(result) == 2:
                u_optimal, predicted_trajectory_mpc = result
                h_values = None
                mpc_success = True
                target_trajectory_cartesian_miqp = None
                gamma = None
            else:
                u_optimal = result
                predicted_trajectory_mpc = None
                h_values = None
                mpc_success = True
                target_trajectory_cartesian_miqp = None
                gamma = None
            

            if not mpc_success:
                mpc_failure_count += 1
                throttle = 0.0
                steering_angle = 0.0
                predicted_positions = None
            else:
                throttle = u_optimal[0]  # First control input (throttle)
                steering_angle = u_optimal[1]  # Second control input (steering angle)
                
                # Use MPC predicted trajectory if available, otherwise use target trajectory
                if predicted_trajectory_mpc is not None:
                    predicted_positions = predicted_trajectory_mpc  # [x, y, yaw] from MPC
            
            print(f"Throttle: {throttle:.3f}, Steering Angle: {steering_angle:.3f}, Gamma: {gamma:.4f}" if gamma is not None else f"Throttle: {throttle:.3f}, Steering Angle: {steering_angle:.3f}, Gamma: N/A")
            
            # Track gamma changes
            if gamma is not None and prev_gamma is not None:
                if abs(gamma - prev_gamma) > 0.05:  # Significant change threshold
                    print(f"  -> Gamma changed: {prev_gamma:.4f} -> {gamma:.4f} (RL agent adaptation)")
            prev_gamma = gamma if gamma is not None else prev_gamma
            
            # Compute lateral acceleration: a_lat = (v^2 / wheel_base) * tan(delta)
            # This is an approximation for the kinematic model
            a_lat = (car.velocity**2 / car.wheel_base) * np.tan(steering_angle) if abs(steering_angle) > 1e-6 else 0.0
            
        except Exception as e:
            print(f"MPC failed: {e}")
            mpc_failure_count += 1
            timing['mpc_solve'] = 0.0
            # Fallback to simple control
            throttle = 0.0
            steering_angle = 0.0
            predicted_positions = None
            h_values = None
            target_trajectory_cartesian_miqp = None
            gamma = None
            a_lat = 0.0
        
        # Check for violations BEFORE updating car state to freeze exactly at violation frame
        t0_violation_check = time.time()
        violation_detected = False
        collision_detected = False
        out_of_bounds_detected = False
        h_violation_detected = False
        colliding_indices = np.array([], dtype=int)
        
        # Check for collisions using current car position (before update)
        if len(obstacles) > 0 and freeze_on_violation:
            collision_detected, colliding_indices = check_collision(
                car, obstacles, path_points, car_radius, obstacle_shape, ellipse_width, ellipse_height
            )
        
        # Check for boundary violations using current car position (before update)
        if freeze_on_violation:
            trajectory_splines_inner_dist_global = globals().get('trajectory_splines_inner_dist', None)
            trajectory_splines_outer_dist_global = globals().get('trajectory_splines_outer_dist', None)
            spline_path_length_global = globals().get('spline_path_length', None)
            
            out_of_bounds_detected, actual_in_bounds, should_ignore, car_dist_to_path, inner_dist_at_path, outer_dist_at_path = check_boundary_violation(
                car, cte, closest_idx, inner_boundaries_distances, outer_boundaries_distances,
                car_radius, trajectory_splines_inner_dist_global, trajectory_splines_outer_dist_global,
                spline_path_length_global, prev_car_dist_to_path, prev_in_bounds, prev_ignored_jump,
                jump_threshold=1.0, frame=frame, dt=params_simulator['dt']
            )
        
        # Check for h violation (negative h values indicate safety constraint violation)
        if freeze_on_violation and h_values is not None and mpc_controller.obstacles_on:
            if h_values.shape[0] > 0 and h_values.shape[1] > 0:
                # Check if any h value in the current state (first step) is negative
                if np.any(h_values[0, :] < 0.0):
                    h_violation_detected = True
        
        # Freeze on the first violation detected (collision takes priority if both occur)
        if freeze_on_violation:
            if collision_detected:
                if not animation_frozen:
                    collision_count += len(colliding_indices)
                    violation_detected = True
                    animation_frozen = True
                    print(f"\n*** ANIMATION FROZEN: Collision detected at t={frame*params_simulator['dt']:.3f}s ***")
                    # Stop the animation
                    if animation_obj[0] is not None:
                        animation_obj[0].event_source.stop()
            elif out_of_bounds_detected:
                if not animation_frozen:
                    out_of_bounds_count += 1
                    violation_detected = True
                    animation_frozen = True
                    print(f"\n*** ANIMATION FROZEN: Track violation detected at t={frame*params_simulator['dt']:.3f}s ***")
                    # Stop the animation
                    if animation_obj[0] is not None:
                        animation_obj[0].event_source.stop()
            elif h_violation_detected:
                if not animation_frozen:
                    # Don't count here - will be counted in the counting section below to match no-animation behavior
                    violation_detected = True
                    animation_frozen = True
                    negative_h_obs = np.where(h_values[0, :] < 0.0)[0] if h_values is not None and h_values.shape[0] > 0 else []
                    print(f"\n*** ANIMATION FROZEN: CBF violation (h < 0) detected at t={frame*params_simulator['dt']:.3f}s ***")
                    if len(negative_h_obs) > 0:
                        print(f"  Negative h values for obstacles: {negative_h_obs + 1}")
                        print(f"  h values: {h_values[0, negative_h_obs]}")
                    # Stop the animation
                    if animation_obj[0] is not None:
                        animation_obj[0].event_source.stop()
        
        timing['violation_check'] = (time.time() - t0_violation_check) * 1000
        
        # Initialize longitudinal acceleration (will be computed if car state updates)
        longitudinal_acceleration = 0.0
        
        # If violation detected, freeze and skip car state update
        if violation_detected and animation_frozen:
            # Skip car state update to show exact violation position
            timing['car_update'] = 0.0  # No update performed
            longitudinal_acceleration = 0.0  # No acceleration when frozen
        else:
            # Update the car state
            # NOTE: If MPC took longer than dt, the control is stale but we still apply it
            # The actual elapsed time since last update may be > dt, but we use dt for consistency
            t0 = time.time()
            car.update_state(throttle, steering_angle, params_simulator['dt'])
            timing['car_update'] = (time.time() - t0) * 1000
            
            # Calculate distance traveled this frame
            dx = car.x - prev_car_x
            dy = car.y - prev_car_y
            frame_distance = np.sqrt(dx*dx + dy*dy)
            total_distance_traversed += frame_distance
            
            # Calculate longitudinal acceleration from velocity change
            longitudinal_acceleration = (car.velocity - prev_car_velocity) / params_simulator['dt'] if params_simulator['dt'] > 0 else 0.0
            
            # Update previous position and velocity for next frame
            prev_car_x = car.x
            prev_car_y = car.y
            prev_car_velocity = car.velocity
        
        # Efficient collision detection (for counting, even if frozen)
        if len(obstacles) > 0:
            _, colliding_indices = check_collision(
                car, obstacles, path_points, car_radius, obstacle_shape, ellipse_width, ellipse_height
            )
            
            # Count collisions at each iteration (every frame when collision occurs)
            # Only count if not already counted in pre-check (to avoid double-counting when frozen)
            if len(colliding_indices) > 0 and not (freeze_on_violation and violation_detected):
                collision_count += len(colliding_indices)
                # Note: Freezing already handled before car state update
        
        # Simple boundary violation detection using closest_idx and boundary distances
        t0_boundary = time.time()
        
        trajectory_splines_inner_dist_global = globals().get('trajectory_splines_inner_dist', None)
        trajectory_splines_outer_dist_global = globals().get('trajectory_splines_outer_dist', None)
        spline_path_length_global = globals().get('spline_path_length', None)
        
        out_of_bounds, actual_in_bounds, should_ignore, car_dist_to_path, inner_dist_at_path, outer_dist_at_path = check_boundary_violation(
            car, cte, closest_idx, inner_boundaries_distances, outer_boundaries_distances,
            car_radius, trajectory_splines_inner_dist_global, trajectory_splines_outer_dist_global,
            spline_path_length_global, prev_car_dist_to_path, prev_in_bounds, prev_ignored_jump,
            jump_threshold=1.0, frame=frame, dt=params_simulator['dt']
        )
        
        # Count h violations at each iteration (every frame when h < 0, not just on transition)
        # Count every frame when h < 0, regardless of freezing (freezing is just for visualization)
        if h_values is not None and mpc_controller.obstacles_on:
            if h_values.shape[0] > 0 and h_values.shape[1] > 0:
                if np.any(h_values[0, :] < 0.0):
                    h_violation_count += 1
                    # Note: Freezing is handled separately for visualization, but we count every violation
        
        # Count out-of-bounds at each iteration (every frame when out of bounds, not just on transition)
        # But ignore sudden jumps unless they persist
        # Only count if not already counted in pre-check (to avoid double-counting when frozen)
        if out_of_bounds and not (freeze_on_violation and violation_detected):
            out_of_bounds_count += 1
            # Note: Freezing already handled before car state update
            if frame % 10 == 0:  # Print every 10 frames to avoid spam
                if abs(cte) < 1e-6:
                    side = "on path"
                    boundary_info = f"inner={inner_dist_at_path:.3f}m, outer={outer_dist_at_path:.3f}m"
                else:
                    side = "inner" if cte > 0 else "outer"
                    boundary_dist = inner_dist_at_path if cte > 0 else outer_dist_at_path
                    boundary_info = f"{side}_boundary={boundary_dist:.3f}m"
                jump_info = " (persistent after ignored jump)" if prev_ignored_jump else ""
                print(f"OUT OF BOUNDS at t={frame*params_simulator['dt']:.3f}s (count={out_of_bounds_count}){jump_info}: "
                      f"car_dist_to_path={car_dist_to_path:.3f}m, "
                      f"car on {side}, "
                      f"{boundary_info}, "
                      f"car_edge_dist={car_dist_to_path + car_radius:.3f}m, "
                      f"CTE={cte:.3f}m")
        
        # Update previous values for next iteration (use actual state, not ignored state)
        prev_car_dist_to_path = car_dist_to_path
        prev_in_bounds = actual_in_bounds  # Store actual state for next frame's comparison
        prev_ignored_jump = should_ignore  # Track if we ignored a jump this frame
        
        timing['boundary_check'] = (time.time() - t0_boundary) * 1000

        # Record history for visualization (limit for performance)
        history.append((car.x, car.y))
        if len(history) > max_history_points:
            history.pop(0)  # Remove oldest point

        # Update obstacle angles for ellipses (position updates happen later with shift)
        if len(obstacles) > 0 and obstacle_shape == 'ellipse':
            for i, obs_data in enumerate(obstacles):
                if i < len(obstacle_patches):
                    patch = obstacle_patches[i]
                    if hasattr(patch, 'path_idx'):
                        path_idx = int(obs_data[4]) if len(obs_data) > 4 else patch.path_idx
                        next_idx = (path_idx + 1) % len(path_points)
                        dx = path_points[next_idx, 0] - path_points[path_idx, 0]
                        dy = path_points[next_idx, 1] - path_points[path_idx, 1]
                        heading_rad = np.arctan2(dy, dx)
                        heading_deg = np.degrees(heading_rad)
                        patch.angle = heading_deg
                        patch.path_idx = path_idx
        
        # Extract predicted positions and yaw angles (for later use with shifting)
        if predicted_positions is not None:
            x_pred = predicted_positions[:, 0]
            y_pred = predicted_positions[:, 1]
            yaw_pred = predicted_positions[:, 2]
            # Compute the components for the quiver arrows
            u_pred = np.cos(yaw_pred)
            v_pred = np.sin(yaw_pred)
        else:
            x_pred = None
            y_pred = None
            u_pred = None
            v_pred = None
        
        # FIXED: Always update data arrays to keep them synchronized
        time_data.append(frame * params_simulator['dt'])
        current_velocity_data.append(car.velocity)
        # Use actual target velocity from trajectory if raceline velocity is enabled, otherwise use constant target_velocity
        if params_simulator.get('use_raceline_velocity', False) and target_trajectory is not None and len(target_trajectory) > 0:
            target_velocity_data.append(target_trajectory[0, 3])  # Column 3 is velocity
        else:
            target_velocity_data.append(params_simulator['target_velocity'])
        steering_angle_data.append(steering_angle)
        cte_data.append(cte)  # Add CTE data to keep arrays synchronized
        alat_data.append(a_lat)  # Add lateral acceleration data
        # Add gamma data (use default gamma if None)
        gamma_value = gamma if gamma is not None else (mpc_controller.gamma_default if hasattr(mpc_controller, 'gamma_default') else 0.0)
        gamma_data.append(gamma_value)
        gamma_data_full.append(gamma_value)  # Keep full backup for summary plot
        average_curvature_data.append(avg_curvature)  # Add average curvature data
        average_curvature_data_full.append(avg_curvature)  # Keep full backup for summary plot
        h_values_history.append(h_values)
        h_values_history_full.append(h_values)  # Keep full backup for summary plot
        
        # Update data plots every frame for smooth animation (only if plots exist)
        if velocity_line is not None:
            # Update velocity plot
            velocity_line.set_data(time_data, current_velocity_data)
            target_velocity_line.set_data(time_data, target_velocity_data)

        if steering_angle_line is not None:
            # Update steering angle plot
            steering_angle_line.set_data(time_data, steering_angle_data)

        if cte_line is not None:
            # Update cross-track error plot
            cte_line.set_data(time_data, cte_data)

        if alat_line is not None:
            # Update lateral acceleration plot
            alat_line.set_data(time_data, alat_data)
        
        if gamma_line is not None:
            # Update gamma plot
            gamma_line.set_data(time_data, gamma_data)
        
        # Update h values plot
        if mpc_controller.obstacles_on and ax5 is not None and len(h_lines) > 0:
            num_obs = len(h_lines)
            current_time = frame * params_simulator['dt']
            
            # Adaptive x-axis: rolling window
            x_min = max(0, current_time - h_window_size)
            x_max = min(current_time + 1.0, params_simulator['max_time'])
            ax5.set_xlim(x_min, x_max)
            
            # Collect all h values for y-axis limits
            all_h_vals = []
            
            for obs_idx in range(num_obs):
                h_times = []
                h_vals = []
                # Use time_data to get actual times (handles data clearing)
                for frame_idx in range(len(h_values_history)):
                    if frame_idx < len(time_data):
                        h_frame = h_values_history[frame_idx]
                        if h_frame is not None and h_frame.shape[1] > obs_idx:
                            # Use the first step's h value (current state)
                            if h_frame.shape[0] > 0:
                                t = time_data[frame_idx]  # Use actual time from time_data
                                # Only include data in visible window (with small buffer for continuity)
                                if x_min - 0.5 <= t <= x_max + 0.5:
                                    h_times.append(t)
                                    h_val = h_frame[0, obs_idx]
                                    h_vals.append(h_val)
                                    # Only add to all_h_vals if in strict window for y-axis limits
                                    if x_min <= t <= x_max:
                                        all_h_vals.append(h_val)
                
                if len(h_vals) > 0:
                    h_lines[obs_idx].set_data(h_times, h_vals)
                else:
                    # Clear the line if no data
                    h_lines[obs_idx].set_data([], [])
            
            # Adaptive y-axis limits based on visible data
            if len(all_h_vals) > 0:
                h_min = min(all_h_vals)
                h_max = max(all_h_vals)
                # Add padding
                h_range = h_max - h_min
                if h_range < 0.1:  # If range is too small, add fixed padding
                    h_padding = 0.5
                else:
                    h_padding = h_range * 0.1
                ax5.set_ylim(h_min - h_padding, h_max + h_padding)
            else:
                # Default limits if no data yet - try to use all available data
                all_available_h_vals = []
                for frame_idx in range(len(h_values_history)):
                    if frame_idx < len(time_data):
                        h_frame = h_values_history[frame_idx]
                        if h_frame is not None:
                            for obs_idx in range(min(num_obs, h_frame.shape[1])):
                                if h_frame.shape[0] > 0:
                                    all_available_h_vals.append(h_frame[0, obs_idx])
                
                if len(all_available_h_vals) > 0:
                    h_min = min(all_available_h_vals)
                    h_max = max(all_available_h_vals)
                    h_range = h_max - h_min
                    if h_range < 0.1:
                        h_padding = 0.5
                    else:
                        h_padding = h_range * 0.1
                    ax5.set_ylim(h_min - h_padding, h_max + h_padding)
                else:
                    # Fallback default limits
                    ax5.set_ylim(-2, 2)
        
        # Keep aspect equal so the car doesn't get stretched
        ax1.set_aspect('equal')


        # Record end time
        end_time = time.time()
        
        # Calculate and print computation time
        computation_time = end_time - start_time
        # Note: MPC solve time is already tracked above, this is total frame time
        # print(f"\nFrame {frame} timing breakdown:")
        print(f"  Total: {computation_time*1000:.2f}ms")
        for key, val in timing.items():
            pct = (val / computation_time / 10) * 100 if computation_time > 0 else 0
            print(f"  {key:20s}: {val:.2f}ms ({pct:.1f}%)")
        print(f"  Others: {computation_time*1000 - sum(timing.values()):.2f}ms")
        
        # Collect frame data for CSV export
        if save_data_csv:
            current_time = frame * params_simulator['dt']
            
            # Get computation times safely
            mpc_time_ms = None
            miqp_time_ms = None
            rl_time_ms = None
            try:
                if 'mpc_solve_time' in locals():
                    mpc_time_ms = mpc_solve_time * 1000
            except:
                pass
            try:
                if hasattr(mpc_controller, 'last_miqp_computation_time') and mpc_controller.last_miqp_computation_time is not None:
                    miqp_time_ms = mpc_controller.last_miqp_computation_time * 1000
            except:
                pass
            try:
                if hasattr(mpc_controller, 'last_rl_computation_time') and mpc_controller.last_rl_computation_time is not None:
                    rl_time_ms = mpc_controller.last_rl_computation_time * 1000
            except:
                pass
            
            # Compute target velocity (same logic as for plotting)
            if params_simulator.get('use_raceline_velocity', False) and target_trajectory is not None and len(target_trajectory) > 0:
                current_target_velocity = target_trajectory[0, 3]  # Column 3 is velocity
            else:
                current_target_velocity = params_simulator['target_velocity']
            
            frame_data = {
                'frame': frame,
                'time_s': current_time,
                'mpc_computation_time_ms': mpc_time_ms,
                'miqp_computation_time_ms': miqp_time_ms,
                'rl_computation_time_ms': rl_time_ms,
                'collision_count_cumulative': collision_count,
                'out_of_bounds_count_cumulative': out_of_bounds_count,
                'h_violation_count_cumulative': h_violation_count,
                'successful_overtakes_cumulative': successful_overtakes,
                'mpc_failure_count_cumulative': mpc_failure_count,
                'miqp_failure_count_cumulative': miqp_failure_count,
                'car_x': car.x,
                'car_y': car.y,
                'car_yaw': car.yaw,
                'car_velocity': car.velocity,
                'target_velocity': current_target_velocity,
                'steering_angle': steering_angle,
                'throttle': throttle if 'throttle' in locals() else None,
                'lateral_acceleration': a_lat if 'a_lat' in locals() else None,
                'longitudinal_acceleration': longitudinal_acceleration if 'longitudinal_acceleration' in locals() else None,
                'gamma': gamma if 'gamma' in locals() and gamma is not None else None,
                'average_curvature': avg_curvature if 'avg_curvature' in locals() else None,
                'closest_idx': closest_idx,
                'cte': cte if 'cte' in locals() else None
            }
            
            # Add obstacle positions (x, y), heading, and closest_idx for each obstacle
            if len(obstacles) > 0:
                for obs_idx in range(len(obstacles)):
                    frame_data[f'obs_{obs_idx}_x'] = obstacles[obs_idx, 0]
                    frame_data[f'obs_{obs_idx}_y'] = obstacles[obs_idx, 1]
                    
                    # Get closest path index (path_idx) for this obstacle
                    if obstacles.shape[1] > 4:
                        path_idx = int(obstacles[obs_idx, 4])
                        frame_data[f'obs_{obs_idx}_closest_idx'] = path_idx
                        
                        # Compute heading from raceline at obstacle's path index
                        next_idx = (path_idx + 1) % len(path_points)
                        dx = path_points[next_idx, 0] - path_points[path_idx, 0]
                        dy = path_points[next_idx, 1] - path_points[path_idx, 1]
                        heading_rad = np.arctan2(dy, dx)
                        frame_data[f'obs_{obs_idx}_heading'] = heading_rad
                    else:
                        # If path_idx not available, compute closest point
                        obs_x, obs_y = obstacles[obs_idx, 0], obstacles[obs_idx, 1]
                        distances = np.linalg.norm(path_points[:, :2] - np.array([obs_x, obs_y]), axis=1)
                        path_idx = int(np.argmin(distances))
                        frame_data[f'obs_{obs_idx}_closest_idx'] = path_idx
                        
                        # Compute heading from raceline at closest point
                        next_idx = (path_idx + 1) % len(path_points)
                        dx = path_points[next_idx, 0] - path_points[path_idx, 0]
                        dy = path_points[next_idx, 1] - path_points[path_idx, 1]
                        heading_rad = np.arctan2(dy, dx)
                        frame_data[f'obs_{obs_idx}_heading'] = heading_rad
            
            # Add h values - only save the first h-value (step_0) for each obstacle
            if 'h_values' in locals() and h_values is not None and isinstance(h_values, np.ndarray):
                if h_values.ndim == 2:
                    # 2D array: [horizon_steps, num_obstacles]
                    for obs_idx in range(h_values.shape[1]):
                        # Only save step_0 (first h-value)
                        frame_data[f'h_obs_{obs_idx}'] = h_values[0, obs_idx]
                elif h_values.ndim == 1:
                    # 1D array: [num_obstacles]
                    for obs_idx in range(len(h_values)):
                        frame_data[f'h_obs_{obs_idx}'] = h_values[obs_idx]
            else:
                # Add None for h values if not available
                if len(obstacles) > 0:
                    for obs_idx in range(len(obstacles)):
                        frame_data[f'h_obs_{obs_idx}'] = None
            
            frame_data_list.append(frame_data)

        # Check if data lists have reached the maximum limit
        if len(time_data) > params_simulator['max_data_length']:
            # Track frame offset: next frame will be stored at index 0, so offset = frame + 1
            h_values_frame_offset = frame + 1
            # Clear all data lists
            time_data.clear()
            current_velocity_data.clear()
            target_velocity_data.clear()
            steering_angle_data.clear()
            cte_data.clear()
            alat_data.clear()
            gamma_data.clear()
            average_curvature_data.clear()
            h_values_history.clear()
            print(f"Data lists cleared at frame {frame} to prevent overflow.")

        # 1. Calculate the Shift (Where is the car?)
        if padding_car:
            shift_x = car.x
            shift_y = car.y
        else:
            shift_x = 0
            shift_y = 0

        # 2. Update all visualization elements with shift applied (single update, no duplicates)
        # Move the Raceline relative to the car (The track moves opposite to the car)
        raceline_line.set_data(path_points[:, 0] - shift_x, path_points[:, 1] - shift_y)
        boundary_inner_line.set_data(boundaries_points[:, 0] - shift_x, boundaries_points[:, 1] - shift_y)
        boundary_outer_line.set_data(boundaries_points[:, 2] - shift_x, boundaries_points[:, 3] - shift_y)

        # 3. Move the Car Path
        if len(history) > 0:
            car_path_x = [p[0] - shift_x for p in history]
            car_path_y = [p[1] - shift_y for p in history]
            car_path.set_data(car_path_x, car_path_y)
        
        # Update car ellipse position and orientation
        if padding_car:
            car_ellipse.center = (0, 0)
        else:
            car_ellipse.center = (car.x - shift_x, car.y - shift_y)
        car_ellipse.angle = np.degrees(car.yaw)
        
        # 4. Move the Obstacles
        if len(obstacles) > 0:
            # Update obstacle centers (dots)
            obstacle_centers.set_data(obstacles[:, 0] - shift_x, obstacles[:, 1] - shift_y)
            
            # Update obstacle patches (circles or ellipses) - position relative to car
            for i, patch in enumerate(obstacle_patches):
                if i < len(obstacles):
                    obs_x = obstacles[i, 0] - shift_x
                    obs_y = obstacles[i, 1] - shift_y
                    patch.center = (obs_x, obs_y)

        # 5. Move Target & Predicted Trajectories
        target_trajectory_line.set_data(target_trajectory[:, 0] - shift_x, target_trajectory[:, 1] - shift_y)
        
        # Update MIQP trajectory if available
        if target_trajectory_cartesian_miqp is not None and len(target_trajectory_cartesian_miqp) > 0:
            if target_trajectory_cartesian_miqp.ndim == 2 and target_trajectory_cartesian_miqp.shape[1] >= 2:
                miqp_trajectory_line.set_data(target_trajectory_cartesian_miqp[:, 0] - shift_x, target_trajectory_cartesian_miqp[:, 1] - shift_y)
            else:
                miqp_trajectory_line.set_data([], [])
        else:
            miqp_trajectory_line.set_data([], [])
        
        if predicted_positions is not None and x_pred is not None:
            predicted_line.set_data(predicted_positions[:, 0] - shift_x, predicted_positions[:, 1] - shift_y)
            # Move predicted arrows
            predicted_orientation_quiver.set_offsets(np.c_[x_pred - shift_x, y_pred - shift_y])
            predicted_orientation_quiver.set_UVC(u_pred, v_pred)

        # 6. Update Car Orientation Arrow
        if padding_car:
            orientation_quiver.set_offsets(np.array([[0, 0]]))
        else:
            orientation_quiver.set_offsets(np.array([[car.x - shift_x, car.y - shift_y]]))
        orientation_quiver.set_UVC(np.cos(car.yaw), np.sin(car.yaw))

        # 7. Optionally keep view focused and zoomed on the car
        if focus_on_car:
            view_half_width = params_simulator.get('focus_view_half_width', 40.0)
            view_half_height = params_simulator.get('focus_view_half_height', 25.0)
            if padding_car:
                center_x, center_y = 0.0, 0.0
            else:
                center_x, center_y = car.x - shift_x, car.y - shift_y
            ax1.set_xlim(center_x - view_half_width, center_x + view_half_width)
            ax1.set_ylim(center_y - view_half_height, center_y + view_half_height)

        # 8. Update time display
        current_time = frame * params_simulator['dt']
        time_text.set_text(f'Time: {current_time:.2f} s')
        
        # IMPORTANT: Make sure raceline_line is added to the return tuple!
        base_return = tuple(obstacle_patches) + (raceline_line, boundary_inner_line, boundary_outer_line, obstacle_centers, car_path, car_ellipse, target_trajectory_line, miqp_trajectory_line, orientation_quiver, time_text)
        
        # Add plot lines only if they exist (not in CSV save mode)
        if velocity_line is not None:
            base_return = base_return + (velocity_line, target_velocity_line, steering_angle_line, cte_line, alat_line, gamma_line)
        
        if predicted_positions is not None:
            base_return = base_return + (predicted_line, predicted_orientation_quiver)
        
        if mpc_controller.obstacles_on and ax5 is not None and len(h_lines) > 0:
            base_return = base_return + tuple(h_lines)
        
        # Store return value for frozen animation
        last_return_value = base_return
        return base_return

    # Ensure all plots are fully initialized and rendered before starting animation
    print("Initializing plots...")
    if not save_data_csv:
        plt.tight_layout(rect=[0, 0, 0.95, 1])  # Leave space on the right for legend
    else:
        plt.tight_layout()  # Simple layout for single plot
    
    # Force matplotlib to render everything to ensure plots are ready
    fig.canvas.draw()
    plt.pause(0.1)  # Small pause to ensure rendering completes
    
    # Initialize the first frame to set up all plot data before animation starts
    print("Preparing first frame...")
    try:
        update(0)
        fig.canvas.draw()
        plt.pause(0.1)
    except Exception as e:
        print(f"Warning: Initial frame setup had issues: {e}")
    
    print("Starting animation...")
    
    # Real-time simulation: Match the physics dt to the display
    # NOTE: update(0) was already called above for initialization, so we start FuncAnimation from frame 1
    # to avoid processing frame 0 twice. This ensures we process exactly num_steps frames.
    frame_interval_ms = params_simulator['dt'] * 1000  # Convert dt to milliseconds
    num_frames = int(params_simulator["max_time"] / params_simulator['dt'])
    # Start from frame 1 since frame 0 was already processed in initialization
    ani = FuncAnimation(fig, update, frames=range(1, num_frames), 
                        # interval=frame_interval_ms, blit=True, repeat=False)  # Real-time synchronization
                        interval=2, blit=True, repeat=False)  # Real-time synchronization
    animation_obj[0] = ani  # Store animation object for stopping from update function
    plt.show()
    
    # Create summary figure (skip if saving CSV)
    if not save_data_csv:
        summary_fig = plt.figure(figsize=(16, 12))
        gs_summary = GridSpec(4, 3, figure=summary_fig, hspace=0.3, wspace=0.3)
        
        # 1. Computation time histogram (MPC, MIQP, and RL)
        ax_comp = summary_fig.add_subplot(gs_summary[0, 0])
        if len(computation_times) > 0:
            mpc_times_ms = np.array(computation_times) * 1000
            ax_comp.hist(mpc_times_ms, bins=50, alpha=0.7, label=f'MPC (avg: {np.mean(mpc_times_ms):.2f}ms)', color='blue')
        if len(miqp_computation_times) > 0:
            miqp_times_ms = np.array(miqp_computation_times) * 1000
            ax_comp.hist(miqp_times_ms, bins=50, alpha=0.7, label=f'MIQP (avg: {np.mean(miqp_times_ms):.2f}ms)', color='orange')
        if len(rl_computation_times) > 0:
            rl_times_ms = np.array(rl_computation_times) * 1000
            ax_comp.hist(rl_times_ms, bins=50, alpha=0.7, label=f'RL (avg: {np.mean(rl_times_ms):.2f}ms)', color='green')
        ax_comp.set_xlabel('Computation Time (ms)')
        ax_comp.set_ylabel('Frequency')
        ax_comp.set_title('Computation Time Distribution')
        ax_comp.legend()
        ax_comp.grid(True, alpha=0.3)
        
        # 2. Computation time over time
        ax_comp_time = summary_fig.add_subplot(gs_summary[0, 1])
        if len(computation_times) > 0:
            mpc_frame_times = np.arange(len(computation_times)) * params_simulator['dt']
            ax_comp_time.plot(mpc_frame_times, np.array(computation_times) * 1000, 'b-', alpha=0.6, label='MPC+MIQP', linewidth=0.5)
        if len(miqp_computation_times) > 0:
            miqp_frame_times = np.arange(len(miqp_computation_times)) * params_simulator['dt']
            ax_comp_time.plot(miqp_frame_times, np.array(miqp_computation_times) * 1000, 'orange', alpha=0.6, label='MIQP', linewidth=0.5)
        if len(rl_computation_times) > 0:
            rl_frame_times = np.arange(len(rl_computation_times)) * params_simulator['dt']
            ax_comp_time.plot(rl_frame_times, np.array(rl_computation_times) * 1000, 'g-', alpha=0.6, label='RL', linewidth=0.5)
        ax_comp_time.set_xlabel('Time (s)')
        ax_comp_time.set_ylabel('Computation Time (ms)')
        ax_comp_time.set_title('Computation Time Over Time')
        ax_comp_time.legend()
        ax_comp_time.grid(True, alpha=0.3)
        
        # 3. Violations bar chart
        ax_viol = summary_fig.add_subplot(gs_summary[0, 2])
        violation_types = ['Collisions', 'Boundary\nViolations', 'CBF\nViolations\n(h < 0)']
        violation_counts = [collision_count, out_of_bounds_count, h_violation_count]
        colors_viol = ['red', 'orange', 'purple']
        bars = ax_viol.bar(violation_types, violation_counts, color=colors_viol, alpha=0.7)
        ax_viol.set_ylabel('Count')
        ax_viol.set_title('Violations Summary')
        ax_viol.grid(True, alpha=0.3, axis='y')
        for i, (bar, count) in enumerate(zip(bars, violation_counts)):
            height = bar.get_height()
            ax_viol.text(bar.get_x() + bar.get_width()/2., height, f'{int(count)}', 
                        ha='center', va='bottom', fontweight='bold')
        
        # 4. Overtakes and failures
        ax_stats = summary_fig.add_subplot(gs_summary[1, 0])
        stats_types = ['Overtakes', 'MPC\nFailures', 'MIQP\nFailures']
        stats_counts = [successful_overtakes, mpc_failure_count, miqp_failure_count]
        colors_stats = ['green', 'red', 'orange']
        bars_stats = ax_stats.bar(stats_types, stats_counts, color=colors_stats, alpha=0.7)
        ax_stats.set_ylabel('Count')
        ax_stats.set_title('Performance Statistics')
        ax_stats.grid(True, alpha=0.3, axis='y')
        for bar, count in zip(bars_stats, stats_counts):
            height = bar.get_height()
            ax_stats.text(bar.get_x() + bar.get_width()/2., height, f'{int(count)}', 
                         ha='center', va='bottom', fontweight='bold')
        
        # 5. H values over time for each obstacle (complete horizon)
        ax_h = summary_fig.add_subplot(gs_summary[1, 1:])
        if mpc_controller.obstacles_on and len(h_values_history_full) > 0 and len(obstacles) > 0:
            num_obs = len(obstacles)
            colors_h = plt.cm.tab10(np.linspace(0, 1, num_obs))
            horizon_steps = None
            # Find horizon_steps from first available frame
            for frame_idx in range(len(h_values_history_full)):
                h_frame = h_values_history_full[frame_idx]
                if h_frame is not None and h_frame.shape[0] > 0:
                    horizon_steps = h_frame.shape[0]
                    break
            
            # Determine x-axis range from computation_times to match computation time plot
            max_time_comp = len(computation_times) * params_simulator['dt'] if len(computation_times) > 0 else 0
            max_time_h = len(h_values_history_full) * params_simulator['dt'] if len(h_values_history_full) > 0 else 0
            max_time = max(max_time_comp, max_time_h)
            
            if horizon_steps is not None:
                for obs_idx in range(num_obs):
                    # Plot each horizon step with different alpha
                    for h_step in range(horizon_steps):
                        h_times = []
                        h_vals = []
                        # Use all available h_values_history_full, compute time from frame index
                        for frame_idx in range(len(h_values_history_full)):
                            h_frame = h_values_history_full[frame_idx]
                            if h_frame is not None and h_frame.shape[1] > obs_idx and h_frame.shape[0] > h_step:
                                # Compute time from frame index (no offset needed since we never cleared)
                                t = frame_idx * params_simulator['dt']
                                h_times.append(t)
                                h_vals.append(h_frame[h_step, obs_idx])
                        if len(h_vals) > 0:
                            alpha = 0.2 + 0.6 * (h_step + 1) / horizon_steps  # Fade from light to dark
                            label = f'Obs {obs_idx+1} h{h_step}' if h_step == 0 else ''
                            ax_h.plot(h_times, h_vals, label=label, color=colors_h[obs_idx], 
                                    linewidth=1.0, alpha=alpha)
            # Set x-axis limits to match computation time plot
            if max_time > 0:
                ax_h.set_xlim(0, max_time)
            ax_h.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1, label='Safety Threshold (h=0)')
            ax_h.set_xlabel('Time (s)')
            ax_h.set_ylabel('h value')
            ax_h.set_title('Control Barrier Function (CBF) Values Over Time - Complete Horizon')
            ax_h.legend(loc='best', fontsize=7, ncol=min(num_obs + 1, 5))
            ax_h.grid(True, alpha=0.3)
        else:
            # Set x-axis limits to match computation time plot even if no h data
            max_time_comp = len(computation_times) * params_simulator['dt'] if len(computation_times) > 0 else 0
            if max_time_comp > 0:
                ax_h.set_xlim(0, max_time_comp)
            ax_h.text(0.5, 0.5, 'No obstacle data available', ha='center', va='center', transform=ax_h.transAxes)
            ax_h.set_title('Control Barrier Function (CBF) Values')
        
        # 6. Average curvature over time
        ax_curv = summary_fig.add_subplot(gs_summary[2, :2])
        if len(average_curvature_data_full) > 0:
            curvature_times = np.arange(len(average_curvature_data_full)) * params_simulator['dt']
            ax_curv.plot(curvature_times, average_curvature_data_full, 'purple', alpha=0.7, linewidth=1.5, label='Average Curvature')
            ax_curv.set_xlabel('Time (s)')
            ax_curv.set_ylabel('Average Curvature (1/m)')
            ax_curv.set_title('Average Curvature per Target Trajectory Over Time')
            ax_curv.legend()
            ax_curv.grid(True, alpha=0.3)
            # Set x-axis limits to match other plots
            max_time_curv = len(average_curvature_data_full) * params_simulator['dt'] if len(average_curvature_data_full) > 0 else 0
            if max_time_curv > 0:
                ax_curv.set_xlim(0, max_time_curv)
        else:
            ax_curv.text(0.5, 0.5, 'No curvature data available', ha='center', va='center', transform=ax_curv.transAxes)
            ax_curv.set_title('Average Curvature per Target Trajectory')
        
        # 7. Gamma (CBF parameter) over time
        ax_gamma_summary = summary_fig.add_subplot(gs_summary[2, 2])
        if len(gamma_data_full) > 0:
            gamma_times = np.arange(len(gamma_data_full)) * params_simulator['dt']
            ax_gamma_summary.plot(gamma_times, gamma_data_full, 'purple', alpha=0.7, linewidth=1.5, label='Gamma (CBF Parameter)')
            if hasattr(mpc_controller, 'gamma_default'):
                gamma_default = mpc_controller.gamma_default
                ax_gamma_summary.axhline(y=gamma_default, color='gray', linestyle='--', alpha=0.7, label=f'Default ({gamma_default:.3f})')
            ax_gamma_summary.axhline(y=GAMMA_BOUNDS[0], color='r', linestyle=':', alpha=0.5, label=f'Min ({GAMMA_BOUNDS[0]:.3f})')
            ax_gamma_summary.axhline(y=GAMMA_BOUNDS[1], color='r', linestyle=':', alpha=0.5, label=f'Max ({GAMMA_BOUNDS[1]:.3f})')
            ax_gamma_summary.set_xlabel('Time (s)')
            ax_gamma_summary.set_ylabel('Gamma')
            ax_gamma_summary.set_title('CBF Parameter (Gamma) Over Time')
            ax_gamma_summary.legend(fontsize=8)
            ax_gamma_summary.grid(True, alpha=0.3)
            # Set x-axis limits to match other plots
            max_time_gamma = len(gamma_data_full) * params_simulator['dt'] if len(gamma_data_full) > 0 else 0
            if max_time_gamma > 0:
                ax_gamma_summary.set_xlim(0, max_time_gamma)
            ax_gamma_summary.set_ylim(GAMMA_BOUNDS[0] - 0.1, GAMMA_BOUNDS[1] + 0.1)
        else:
            ax_gamma_summary.text(0.5, 0.5, 'No gamma data available', ha='center', va='center', transform=ax_gamma_summary.transAxes)
            ax_gamma_summary.set_title('CBF Parameter (Gamma)')
        
        # 8. Summary statistics text
        ax_text = summary_fig.add_subplot(gs_summary[3, :])
        ax_text.axis('off')
        stats_text = []
        stats_text.append(f"MPC: min={min(computation_times)*1000:.2f}ms, max={max(computation_times)*1000:.2f}ms, avg={np.mean(computation_times)*1000:.2f}ms" if len(computation_times) > 0 else "MPC: No data")
        stats_text.append(f"MIQP: min={min(miqp_computation_times)*1000:.2f}ms, max={max(miqp_computation_times)*1000:.2f}ms, avg={np.mean(miqp_computation_times)*1000:.2f}ms" if len(miqp_computation_times) > 0 else "MIQP: No data")
        stats_text.append(f"RL: min={min(rl_computation_times)*1000:.2f}ms, max={max(rl_computation_times)*1000:.2f}ms, avg={np.mean(rl_computation_times)*1000:.2f}ms" if len(rl_computation_times) > 0 else "RL: No data")
        stats_text.append(f"Violations: {collision_count} collisions, {out_of_bounds_count} boundary violations, {h_violation_count} CBF violations (h < 0)")
        stats_text.append(f"Overtakes: {successful_overtakes}")
        stats_text.append(f"Failures: {mpc_failure_count} MPC failures, {miqp_failure_count} MIQP failures")
        
        # Calculate laps completed
        path_length = None
        path_total_length_cache = globals().get('path_total_length_cache')
        if path_total_length_cache is not None and path_total_length_cache > 0:
            path_length = path_total_length_cache
        else:
            # Compute path length if not cached
            path_xy = path_points[:, :2]
            n_path = len(path_xy)
            if n_path > 0:
                cumulative_dist = 0.0
                for i in range(n_path - 1):
                    dx = path_xy[i+1, 0] - path_xy[i, 0]
                    dy = path_xy[i+1, 1] - path_xy[i, 1]
                    cumulative_dist += np.sqrt(dx*dx + dy*dy)
                # Close the loop
                dx = path_xy[0, 0] - path_xy[n_path-1, 0]
                dy = path_xy[0, 1] - path_xy[n_path-1, 1]
                path_length = cumulative_dist + np.sqrt(dx*dx + dy*dy)
        
        if path_length is not None and path_length > 0:
            num_laps = total_distance_traversed / path_length
            stats_text.append(f"Distance Traversed: {total_distance_traversed:.2f} m")
            stats_text.append(f"Laps Completed: {num_laps:.3f} (Path Length: {path_length:.2f} m)")
        else:
            stats_text.append(f"Distance Traversed: {total_distance_traversed:.2f} m")
            stats_text.append(f"Laps Completed: N/A (Path length not available)")
        
        if len(average_curvature_data_full) > 0:
            stats_text.append(f"Average Curvature: min={min(average_curvature_data_full):.6f} 1/m, max={max(average_curvature_data_full):.6f} 1/m, avg={np.mean(average_curvature_data_full):.6f} 1/m")
        if len(gamma_data_full) > 0:
            # Use all gamma values for statistics
            stats_text.append(f"Gamma: min={min(gamma_data_full):.6f}, max={max(gamma_data_full):.6f}, avg={np.mean(gamma_data_full):.6f}")
            if hasattr(mpc_controller, 'gamma_default'):
                stats_text.append(f"Gamma Default: {mpc_controller.gamma_default:.6f}")
        ax_text.text(0.5, 0.5, '\n'.join(stats_text), ha='center', va='center', 
                    transform=ax_text.transAxes, fontsize=11, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        summary_fig.suptitle('Simulation Summary', fontsize=14, fontweight='bold')
        plt.show()
    
    # Save CSV data if requested
    if save_data_csv and frame_data_list is not None and len(frame_data_list) > 0:
        timestamp_csv = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f'simulation_data_{timestamp_csv}.csv'
        csv_path = csv_filename
        
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(frame_data_list)
        df.to_csv(csv_path, index=False)
        print(f"Saved simulation data CSV ({len(frame_data_list)} rows) to: {csv_path}")
    
    # Print statistics after animation completes
    print(f"\nMPC Performance Statistics:")
    if len(computation_times) > 0:
        min_computation_time = min(computation_times) * 1000
        max_computation_time = max(computation_times) * 1000
        avg_computation_time = np.mean(computation_times) * 1000
        print(f"  Minimum computation time with MIQP: {min_computation_time:.2f} ms")
        print(f"  Maximum computation time with MIQP: {max_computation_time:.2f} ms")
        print(f"  Average computation time with MIQP: {avg_computation_time:.2f} ms")
    else:
        print(f"  Minimum computation time with MIQP: N/A (no successful MPC solves)")
        print(f"  Maximum computation time with MIQP: N/A (no successful MPC solves)")
        print(f"  Average computation time with MIQP: N/A (no successful MPC solves)")
    print(f"  MPC failures: {mpc_failure_count}")
    
    print(f"\nMIQP Performance Statistics:")
    if len(miqp_computation_times) > 0:
        min_miqp_time = min(miqp_computation_times) * 1000
        max_miqp_time = max(miqp_computation_times) * 1000
        avg_miqp_time = np.mean(miqp_computation_times) * 1000
        print(f"  Minimum computation time: {min_miqp_time:.2f} ms")
        print(f"  Maximum computation time: {max_miqp_time:.2f} ms")
        print(f"  Average computation time: {avg_miqp_time:.2f} ms")
    else:
        print(f"  Minimum computation time: N/A (no MIQP solves)")
        print(f"  Maximum computation time: N/A (no MIQP solves)")
        print(f"  Average computation time: N/A (no MIQP solves)")
    print(f"  MIQP failures: {miqp_failure_count}")
    
    print(f"\nRL Performance Statistics:")
    if len(rl_computation_times) > 0:
        min_rl_time = min(rl_computation_times) * 1000
        max_rl_time = max(rl_computation_times) * 1000
        avg_rl_time = np.mean(rl_computation_times) * 1000
        print(f"  Minimum inference time: {min_rl_time:.2f} ms")
        print(f"  Maximum inference time: {max_rl_time:.2f} ms")
        print(f"  Average inference time: {avg_rl_time:.2f} ms")
    else:
        print(f"  Minimum inference time: N/A (no RL inference)")
        print(f"  Maximum inference time: N/A (no RL inference)")
        print(f"  Average inference time: N/A (no RL inference)")
    
    print(f"\nCollision Statistics:")
    print(f"  Total collisions: {collision_count}")
    
    print(f"\nBoundary Violation Statistics:")
    print(f"  Total out-of-bounds violations: {out_of_bounds_count}")
    
    print(f"\nCBF Violation Statistics:")
    print(f"  Total CBF violations (h < 0): {h_violation_count}")
    
    print(f"\nOvertake Statistics:")
    print(f"  Total successful overtakes: {successful_overtakes}")
    if len(overtake_states) > 0:
        for obs_idx in range(len(overtake_states)):
            state = overtake_states[obs_idx]
            state_str = ["Not started", "Tracking", "Passed 0", "Completed"][state]
            print(f"  Obstacle {obs_idx+1}: {state_str}")
    
    print(f"\nDistance and Laps Statistics:")
    path_length = None
    path_total_length_cache = globals().get('path_total_length_cache')
    if path_total_length_cache is not None and path_total_length_cache > 0:
        path_length = path_total_length_cache
    else:
        # Compute path length if not cached
        path_xy = path_points[:, :2]
        n_path = len(path_xy)
        if n_path > 0:
            cumulative_dist = 0.0
            for i in range(n_path - 1):
                dx = path_xy[i+1, 0] - path_xy[i, 0]
                dy = path_xy[i+1, 1] - path_xy[i, 1]
                cumulative_dist += np.sqrt(dx*dx + dy*dy)
            # Close the loop
            dx = path_xy[0, 0] - path_xy[n_path-1, 0]
            dy = path_xy[0, 1] - path_xy[n_path-1, 1]
            path_length = cumulative_dist + np.sqrt(dx*dx + dy*dy)
    
    if path_length is not None and path_length > 0:
        num_laps = total_distance_traversed / path_length
        print(f"  Total distance traversed: {total_distance_traversed:.2f} m")
        print(f"  Path length: {path_length:.2f} m")
        print(f"  Laps completed: {num_laps:.3f}")
    else:
        print(f"  Total distance traversed: {total_distance_traversed:.2f} m")
        print(f"  Path length: N/A (not computed)")
        print(f"  Laps completed: N/A")




# Main Function to run the simulation
def main():
    # Load parameters
    with open('configs/params.yaml', 'r') as file:
        params = yaml.safe_load(file)
    
    # Set random seed for reproducibility (if specified in config)
    # This ensures that when randomness is enabled, results are reproducible
    random_seed = params.get('simulator', {}).get('random_seed', None)
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        print(f"Random seed set to: {random_seed}")

    # Load raceline
    raceline = pd.read_csv(params['simulator']['raceline_path'])
    boundaries = pd.read_csv(params['simulator']['boundary_path'])

    # Handle start position: use random if enabled, otherwise use config values
    raceline_points = raceline.to_numpy()
    if params['simulator'].get('random_start_position', False):
        random_idx = random.randint(0, len(raceline_points)-1)
        next_idx = (random_idx + 1) % len(raceline_points)
        dx = raceline_points[next_idx, 0] - raceline_points[random_idx, 0]
        dy = raceline_points[next_idx, 1] - raceline_points[random_idx, 1]
        yaw = np.arctan2(dy, dx)
        params['vehicle_models']['kinematic_model']['initial_state']['initial_x'] = raceline_points[random_idx, 0]
        params['vehicle_models']['kinematic_model']['initial_state']['initial_y'] = raceline_points[random_idx, 1]
        params['vehicle_models']['kinematic_model']['initial_state']['initial_yaw'] = yaw
        params['vehicle_models']['kinematic_model']['initial_state']['initial_velocity'] = random.randint(0, int(params['simulator']['target_velocity']))
        print(f"Random start position: {params['vehicle_models']['kinematic_model']['initial_state']['initial_x']}, {params['vehicle_models']['kinematic_model']['initial_state']['initial_y']}, yaw: {yaw:.3f}")
    else:
        # Use start_idx from config to set position and calculate yaw from next index
        start_idx = params['simulator'].get('start_idx', 0)
        start_idx = start_idx % len(raceline_points)  # Ensure valid index
        next_idx = (start_idx + 1) % len(raceline_points)
        dx = raceline_points[next_idx, 0] - raceline_points[start_idx, 0]
        dy = raceline_points[next_idx, 1] - raceline_points[start_idx, 1]
        yaw = np.arctan2(dy, dx)
        params['vehicle_models']['kinematic_model']['initial_state']['initial_x'] = raceline_points[start_idx, 0]
        params['vehicle_models']['kinematic_model']['initial_state']['initial_y'] = raceline_points[start_idx, 1]
        params['vehicle_models']['kinematic_model']['initial_state']['initial_yaw'] = yaw
        print(f"Using fixed start position from config (start_idx={start_idx}): "
              f"x={params['vehicle_models']['kinematic_model']['initial_state']['initial_x']}, "
              f"y={params['vehicle_models']['kinematic_model']['initial_state']['initial_y']}, "
              f"yaw={yaw:.3f}, "
              f"velocity={params['vehicle_models']['kinematic_model']['initial_state']['initial_velocity']}")

    # Initial car state
    kinematic_car = CarKinematicModel(params['vehicle_models']['kinematic_model'])
    
    # Controller
    mpc = KinematicModelMIQPMPCinFF(params['acados_miqp_mpc_params'])

    # Run simulation
    animate_simulation(kinematic_car, raceline, boundaries, mpc, params_simulator=params['simulator'], save_data_csv=False)
    
    # Print final obstacle data for controller use
    if 'static_obstacles' in globals():
        print("\nFinal obstacle data for controller:")
        print("static_obstacles = np.array([")
        for obstacle in static_obstacles:
            print(f"    [{obstacle[0]:.6f}, {obstacle[1]:.6f}, {obstacle[2]:.6f}],")
        print("])")


# Run the main function
if __name__ == "__main__":
    main()
