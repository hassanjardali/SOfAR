import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import time 
import yaml
import random

def generate_moving_obstacles(path_points, num_obstacles, max_radius, velocities):
    """Generate moving obstacles on the raceline with random radii and velocities.
    
    Args:
        path_points: Array of raceline points [x, y, ...]
        num_obstacles: Number of obstacles to generate
        max_radius: Maximum radius for obstacles in meters
        velocities: Array of velocities for each obstacle (m/s)
    
    Returns:
        obstacles: Array of [x, y, radius, velocity, path_index] for each obstacle
    """
    obstacles = []
    path_length = len(path_points)
    
    # Generate random indices for obstacle placement
    obstacle_indices = random.sample(range(path_length), min(num_obstacles, path_length))
    
    for i, idx in enumerate(obstacle_indices):
        x = path_points[idx, 0]
        y = path_points[idx, 1]
        radius = max_radius
        velocity = velocities[i] if i < len(velocities) else 5.0  # Default velocity
        obstacles.append([x, y, radius, velocity, idx])
    
    return np.array(obstacles)

def generate_moving_obstacles(path_points, num_obstacles, max_radius, velocities, obstacle_indices=None, n_deviation=None):
    """Generate moving obstacles on the raceline with random radii and velocities.
    
    Args:
        path_points: Array of raceline points [x, y, ...]
        num_obstacles: Number of obstacles to generate
        max_radius: Maximum radius for obstacles in meters
        velocities: Array of velocities for each obstacle (m/s)
        obstacle_indices: Optional list of raceline indices for obstacle placement. If None, random indices are used.
        n_deviation: Optional list of lateral offsets from raceline in meters. Positive = outer, negative = inner.
                    Can be single value or list. If None, obstacles are placed on raceline.
    
    Returns:
        obstacles: Array of [x, y, radius, velocity, path_index] for each obstacle
    """
    obstacles = []
    path_length = len(path_points)
    
    # Process n_deviation: convert to list if single value
    if n_deviation is not None:
        if isinstance(n_deviation, (int, float)):
            n_deviations = [float(n_deviation)] * num_obstacles
        else:
            n_deviations = [float(d) for d in n_deviation]
            # Pad or truncate to match number of obstacles
            if len(n_deviations) < num_obstacles:
                n_deviations.extend([n_deviations[-1] if n_deviations else 0.0] * (num_obstacles - len(n_deviations)))
            elif len(n_deviations) > num_obstacles:
                n_deviations = n_deviations[:num_obstacles]
    else:
        n_deviations = [0.0] * num_obstacles
    
    # Use provided indices or generate random indices for obstacle placement
    if obstacle_indices is not None:
        # Validate and use provided indices
        obstacle_indices = [int(idx) % path_length for idx in obstacle_indices]  # Wrap indices if needed
        num_obstacles = min(len(obstacle_indices), num_obstacles)
    else:
        # Generate random indices for obstacle placement
        obstacle_indices = random.sample(range(path_length), min(num_obstacles, path_length))
    
    for i, idx in enumerate(obstacle_indices[:num_obstacles]):
        x = path_points[idx, 0]
        y = path_points[idx, 1]
        
        # Apply lateral offset (n_deviation) if specified
        n_dev = n_deviations[i] if i < len(n_deviations) else 0.0
        if abs(n_dev) > 1e-6:
            # Compute tangent vector at this point
            next_idx = (idx + 1) % path_length
            dx = path_points[next_idx, 0] - path_points[idx, 0]
            dy = path_points[next_idx, 1] - path_points[idx, 1]
            tangent_length = np.sqrt(dx*dx + dy*dy)
            if tangent_length > 1e-6:
                # Normalize tangent
                tangent = np.array([dx / tangent_length, dy / tangent_length])
                # Compute normal vector (90 degrees counterclockwise: [-dy, dx])
                normal = np.array([-tangent[1], tangent[0]])
                # Offset position along normal
                x += n_dev * normal[0]
                y += n_dev * normal[1]
        
        radius = max_radius
        velocity = velocities[i] if i < len(velocities) else 5.0  # Default velocity
        obstacles.append([x, y, radius, velocity, idx])
    
    reset_obstacle_splines()  # Reset spline cache for new simulation
    return np.array(obstacles)

# ---------------------------------------------------------------------------
# Spline-based obstacle motion system
# ---------------------------------------------------------------------------
_obs_spline_data = None   # Cached arc-length parametrized periodic cubic splines
_obs_arc_positions = None  # Per-obstacle continuous arc-length positions

def reset_obstacle_splines():
    """Reset obstacle spline cache. Called automatically by generate_moving_obstacles."""
    global _obs_spline_data, _obs_arc_positions
    _obs_spline_data = None
    _obs_arc_positions = None

def _ensure_obstacle_splines(path_points):
    """Create or retrieve cached arc-length parametrized periodic cubic splines for obstacle paths."""
    global _obs_spline_data
    if _obs_spline_data is not None:
        return _obs_spline_data

    from scipy.interpolate import CubicSpline

    n = len(path_points)

    # Vectorized cumulative arc-length computation
    diffs = np.diff(path_points[:, :2], axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    arc = np.zeros(n)
    arc[1:] = np.cumsum(seg_lens)

    # Close the loop
    close_len = np.linalg.norm(path_points[0, :2] - path_points[-1, :2])
    total_length = arc[-1] + close_len

    # Normalized parameter t ∈ [0, 1)  mapped from arc-length
    t = arc / total_length

    # Append first point at t=1 for periodic boundary condition
    t_ext = np.append(t, 1.0)
    x_ext = np.append(path_points[:, 0], path_points[0, 0])
    y_ext = np.append(path_points[:, 1], path_points[0, 1])

    spline_x = CubicSpline(t_ext, x_ext, bc_type='periodic')
    spline_y = CubicSpline(t_ext, y_ext, bc_type='periodic')

    # Velocity spline (if raceline carries velocity in 3rd column)
    spline_vel = None
    if path_points.shape[1] >= 3:
        v_ext = np.append(path_points[:, 2], path_points[0, 2])
        spline_vel = CubicSpline(t_ext, v_ext, bc_type='periodic')

    _obs_spline_data = {
        'spline_x': spline_x,
        'spline_y': spline_y,
        'spline_vel': spline_vel,
        'total_length': total_length,
        'arc_lengths': arc,
        'n_points': n,
    }
    return _obs_spline_data


def update_obstacle_positions(obstacles, path_points, dt, n_deviation=None):
    """Update obstacle positions along the raceline using cubic spline interpolation.

    Uses arc-length parametrized periodic cubic splines (C2-smooth) instead of
    discrete waypoint hopping with linear interpolation.  This eliminates:
      - position jaggedness at waypoint boundaries
      - discontinuous tangent / normal vectors for lateral offset
      - broken wrap-around at the loop closure point
      - searchsorted index bugs

    Args:
        obstacles: Array of [x, y, radius, velocity, path_index] per obstacle
        path_points: Array of raceline points [x, y, (velocity, ...)]
        dt: Time step
        n_deviation: Optional lateral offsets from raceline in meters.
                     Positive = outer, negative = inner.

    Returns:
        Updated obstacles array (same 5-column format)
    """
    global _obs_arc_positions

    if len(obstacles) == 0:
        return obstacles

    n_obs = len(obstacles)
    sp = _ensure_obstacle_splines(path_points)

    spline_x = sp['spline_x']
    spline_y = sp['spline_y']
    total_length = sp['total_length']
    arc_lengths = sp['arc_lengths']
    n_points = sp['n_points']

    # --- initialise arc-length state from path indices on first call ----------
    if _obs_arc_positions is None or len(_obs_arc_positions) != n_obs:
        _obs_arc_positions = np.array([
            arc_lengths[int(obstacles[i, 4]) % n_points] for i in range(n_obs)
        ])
    else:
        # Detect external repositioning (e.g. obstacle moved ahead after overtake)
        # Compare stored arc-position against the path_idx the simulator wrote
        for i in range(n_obs):
            ext_idx = int(obstacles[i, 4]) % n_points
            expected_idx = int(np.clip(
                np.searchsorted(arc_lengths, _obs_arc_positions[i], side='right') - 1,
                0, n_points - 1))
            if abs(ext_idx - expected_idx) > 5 and abs(ext_idx - expected_idx) < n_points - 5:
                # Obstacle was teleported — re-sync arc position from its new path_idx
                _obs_arc_positions[i] = arc_lengths[ext_idx]

    # --- process n_deviation into a numpy array ------------------------------
    if n_deviation is not None:
        if isinstance(n_deviation, (int, float)):
            n_devs = np.full(n_obs, float(n_deviation))
        else:
            n_devs = np.zeros(n_obs)
            nd = list(n_deviation)
            length = min(len(nd), n_obs)
            for i in range(length):
                n_devs[i] = float(nd[i])
            if len(nd) < n_obs:
                n_devs[len(nd):] = float(nd[-1]) if nd else 0.0
    else:
        n_devs = np.zeros(n_obs)

    # --- advance arc-length: s += v·dt,  wrap mod total_length ---------------
    _obs_arc_positions = (_obs_arc_positions + obstacles[:, 3] * dt) % total_length

    # --- evaluate spline for smooth (x, y) -----------------------------------
    t_params = _obs_arc_positions / total_length          # normalised ∈ [0,1)
    new_x = spline_x(t_params)
    new_y = spline_y(t_params)

    # --- nearest integer path index (backward-compat for heading / vel lookup)
    new_path_idx = np.searchsorted(arc_lengths, _obs_arc_positions, side='right') - 1
    new_path_idx = np.clip(new_path_idx, 0, n_points - 1)

    # --- smooth lateral offset via analytic spline derivatives ----------------
    has_dev = np.abs(n_devs) > 1e-6
    if np.any(has_dev):
        dx_dt = spline_x.derivative()(t_params)
        dy_dt = spline_y.derivative()(t_params)
        tang_len = np.maximum(np.sqrt(dx_dt**2 + dy_dt**2), 1e-8)

        # Normal = 90° CCW rotation of unit tangent
        normal_x = -dy_dt / tang_len
        normal_y =  dx_dt / tang_len

        mask = has_dev.astype(float)
        new_x += n_devs * normal_x * mask
        new_y += n_devs * normal_y * mask

    # --- build result (preserve velocity — managed externally) ----------------
    return np.column_stack([
        new_x, new_y, obstacles[:, 2], obstacles[:, 3], new_path_idx.astype(float)
    ])



def smooth_angles_with_wrapping(angles, sigma):
    """Professional angle smoothing that handles wrapping correctly.
    
    This is critical for preventing angle discontinuities at boundaries.
    """
    from scipy.ndimage import gaussian_filter1d
    
    # Convert angles to complex numbers for proper wrapping
    complex_angles = np.exp(1j * angles)
    
    # Smooth the complex representation
    smoothed_real = gaussian_filter1d(np.real(complex_angles), sigma=sigma)
    smoothed_imag = gaussian_filter1d(np.imag(complex_angles), sigma=sigma)
    
    # Convert back to angles
    smoothed_angles = np.angle(smoothed_real + 1j * smoothed_imag)
    
    return smoothed_angles

def preprocess_raceline_for_smoothness(path_points):
    """Professional raceline preprocessing to eliminate CSV discontinuities.
    
    This solves the critical issue where CSV raceline data has a sudden jump
    between the last point and first point, causing CTE spikes.
    
    Professional approach:
    - Detects discontinuity at start/end boundary
    - Applies smoothing filter to eliminate the jump
    - Ensures smooth transition for MPC controller
    """
    n_points = len(path_points)
    
    # PROFESSIONAL: Detect discontinuity at start/end boundary
    # Calculate the jump between last and first points
    last_point = path_points[-1, :2]  # [x, y]
    first_point = path_points[0, :2]  # [x, y]
    boundary_jump = np.linalg.norm(last_point - first_point)
    
    print(f"Raceline boundary jump detected: {boundary_jump:.6f}")
    
    if boundary_jump > 0.1:  # Significant discontinuity
        print("Applying professional raceline smoothing to eliminate discontinuity...")
        
        # PROFESSIONAL: Create extended raceline for smoothing
        # Duplicate the raceline to create a seamless loop for filtering
        extended_path = np.vstack([path_points, path_points, path_points])  # Triple the path
        
        # PROFESSIONAL: Apply Gaussian smoothing filter
        # This eliminates the discontinuity while preserving the track shape
        from scipy.ndimage import gaussian_filter1d
        
        # Smooth x and y coordinates separately
        sigma = 2.0  # Smoothing parameter (adjust for desired smoothness)
        smoothed_x = gaussian_filter1d(extended_path[:, 0], sigma=sigma)
        smoothed_y = gaussian_filter1d(extended_path[:, 1], sigma=sigma)
        
        # CRITICAL: Handle angle wrapping in velocity column if present
        if path_points.shape[1] > 2:
            # Check if the third column contains angles (yaw) or velocity
            # If it's angles, we need to handle wrapping
            col3_data = extended_path[:, 2]
            if np.max(np.abs(col3_data)) > 2.0:  # Likely angles (radians)
                # Smooth angles with proper wrapping
                smoothed_angles = smooth_angles_with_wrapping(col3_data, sigma)
                smoothed_path_extended = np.column_stack([smoothed_x, smoothed_y, smoothed_angles])
            else:  # Likely velocity
                smoothed_velocity = gaussian_filter1d(col3_data, sigma=sigma)
                smoothed_path_extended = np.column_stack([smoothed_x, smoothed_y, smoothed_velocity])
        else:
            smoothed_path_extended = np.column_stack([smoothed_x, smoothed_y])
        
        # PROFESSIONAL: Extract the middle section (original raceline)
        # This eliminates edge effects from the smoothing
        start_idx = n_points
        end_idx = 2 * n_points
        smoothed_path = smoothed_path_extended[start_idx:end_idx]
        
        # Verify the smoothing worked
        new_boundary_jump = np.linalg.norm(smoothed_path[-1, :2] - smoothed_path[0, :2])
        print(f"Boundary jump after smoothing: {new_boundary_jump:.6f}")
        
        return smoothed_path
        
    else:
        print("Raceline is already smooth - no preprocessing needed")
        return path_points

def create_periodic_splines(path_points):
    """Create periodic cubic splines to solve discontinuity at raceline start/end.
    
    This is the PROFESSIONAL solution to the critical discontinuity problem:
    - Preprocesses raceline to eliminate CSV discontinuities
    - Creates smooth periodic splines that wrap around seamlessly
    - Eliminates sudden jumps at index 0
    - Provides smooth derivatives everywhere
    - Enables continuous trajectory planning
    """
    from scipy.interpolate import CubicSpline
    
    # PROFESSIONAL: Preprocess raceline to eliminate CSV discontinuities
    smoothed_path = preprocess_raceline_for_smoothness(path_points)
    
    n_points = len(smoothed_path)
    
    # CRITICAL: Ensure raceline is properly closed for periodicity
    # Check if first and last points are identical (within tolerance)
    first_point = smoothed_path[0, :2]  # [x, y]
    last_point = smoothed_path[-1, :2]  # [x, y]
    closure_error = np.linalg.norm(first_point - last_point)
    
    if closure_error > 1e-6:  # If raceline is not closed
        # PROFESSIONAL: Close the raceline by adding the first point at the end
        closed_path = np.vstack([smoothed_path, smoothed_path[0:1]])  # Add first point at end
        n_points = len(closed_path)
        print(f"Raceline closure error: {closure_error:.6f} - Automatically closed raceline")
    else:
        closed_path = smoothed_path
        print(f"Raceline is properly closed (error: {closure_error:.6f})")
    
    # CRITICAL: Create periodic parameterization
    # Parameter t goes from 0 to 1, representing the full raceline
    t = np.linspace(0, 1, n_points, endpoint=False)  # endpoint=False for periodicity
    
    # Extract x and y coordinates from closed path
    x_coords = closed_path[:, 0]
    y_coords = closed_path[:, 1]
    
    # Create periodic cubic splines
    # This ensures smoothness at the boundary (t=0 and t=1 are identical)
    spline_x = CubicSpline(t, x_coords, bc_type='periodic')
    spline_y = CubicSpline(t, y_coords, bc_type='periodic')
    
    # PROFESSIONAL: Create curvature spline from spline derivatives
    # Curvature = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
    def compute_curvature_spline(t_param):
        # First derivatives
        dx_dt = spline_x.derivative()(t_param)
        dy_dt = spline_y.derivative()(t_param)
        
        # Second derivatives
        d2x_dt2 = spline_x.derivative(2)(t_param)
        d2y_dt2 = spline_y.derivative(2)(t_param)
        
        # Curvature formula
        # numerator = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2)
        numerator = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2)
        denominator = (dx_dt**2 + dy_dt**2)**(3/2)
        
        # Avoid division by zero
        curvature = np.where(denominator > 1e-6, numerator / denominator, 0.0)
        return curvature

    def compute_curvature(spline_x, spline_y, t_param):
        dx = spline_x.derivative(1)(t_param)
        dy = spline_y.derivative(1)(t_param)
        ddx = spline_x.derivative(2)(t_param)
        ddy = spline_y.derivative(2)(t_param)
        
        # numerator = np.abs(dx * ddy - dy * ddx)
        numerator = (dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2)**1.5
        
        return np.where(denominator > 1e-6, numerator / denominator, 0.0)
        
    
    # Create curvature spline for efficient evaluation
    curvature_values = compute_curvature_spline(t)
    # curvature_values = [compute_curvature(spline_x, spline_y, t_param) for t_param in t]
    spline_curvature = CubicSpline(t, curvature_values, bc_type='periodic')  # Options: 'periodic', 'natural', 'not-a-knot', 'clamped', or tuple
    
    return spline_x, spline_y, spline_curvature


def compute_cte_old(car, path_points, closest_idx):
    # Car's current position
    car_position = np.array([car.x, car.y])
    
    # Set `point1` and `point2` based on `closest_idx` boundaries efficiently
    if closest_idx == 0:
        point1, point2 = path_points[0, :2], path_points[1, :2]
    elif closest_idx == len(path_points) - 1:
        point1, point2 = path_points[-2, :2], path_points[-1, :2]
    else:
        point1, point2 = path_points[closest_idx - 1, :2], path_points[closest_idx + 1, :2]
    
    # Calculate segment vector and car-to-point1 vector
    segment_vector = point2 - point1
    point_to_car = car_position - point1
    
    # Pre-compute dot products for projection
    segment_length_squared = np.dot(segment_vector, segment_vector)
    
    # Handle edge case where segment length is ze
    if segment_length_squared == 0:
        return np.linalg.norm(point_to_car)
    
    # Compute projection factor (clamped between 0 and 1)
    projection_factor = max(0, min(1, np.dot(point_to_car, segment_vector) / segment_length_squared))
    
    # Closest point on segment and CTE calculation
    closest_point_on_segment = point1 + projection_factor * segment_vector
    car_to_closest = car_position - closest_point_on_segment
    
    # Calculate signed CTE: positive = left of path, negative = right of path
    # Normal vector (90 degrees counterclockwise from segment direction)
    normal_vector = np.array([-segment_vector[1], segment_vector[0]])
    normal_vector = normal_vector / np.linalg.norm(normal_vector) if np.linalg.norm(normal_vector) > 0 else normal_vector
    
    # Sign is determined by dot product with normal (left side = positive)
    cte_magnitude = np.linalg.norm(car_to_closest)
    sign = np.sign(np.dot(car_to_closest, normal_vector))
    cte = sign * cte_magnitude
    
    return cte


def compute_cte(car, path_points, closest_idx):
    """
    Compute Cross-Track Error (CTE) with signed direction.
    
    Args:
        car: Object with x, y attributes (car position)
        path_points: Nx2 array of path waypoints
        closest_idx: Index of closest path point
    
    Returns:
        float: Signed CTE (positive = left of path, negative = right)
    """
    car_position = np.array([car.x, car.y])
    
    # Get segment endpoints with boundary handling
    n_points = len(path_points)
    idx1 = (closest_idx - 1) % n_points  # Previous point (wraps around)
    idx2 = (closest_idx + 1) % n_points  # Next point (wraps around)
    
    point1 = path_points[idx1, :2]
    point2 = path_points[idx2, :2]
    
    # Segment vector and car-relative position
    segment_vector = point2 - point1
    car_relative = car_position - point1
    
    # Compute segment length squared (avoid redundant sqrt)
    segment_length_squared = np.dot(segment_vector, segment_vector)
    
    # Handle zero-length segment (degenerate case)
    if segment_length_squared < 1e-10:
        # Fallback: distance to closest point
        closest_point = path_points[closest_idx, :2]
        return np.linalg.norm(car_position - closest_point)
    
    # Project car onto segment (clamped to [0, 1])
    projection_factor = np.clip(
        np.dot(car_relative, segment_vector) / segment_length_squared,
        0, 1
    )
    
    # Closest point on segment
    closest_point = point1 + projection_factor * segment_vector
    car_to_closest = car_position - closest_point
    
    # Compute signed CTE using 2D cross product (more efficient than normalization)
    # cross_product = segment_vector × car_to_closest (2D z-component)
    # Positive = left side, negative = right side
    cross_product = segment_vector[0] * car_to_closest[1] - segment_vector[1] * car_to_closest[0]
    
    # CTE magnitude
    cte_magnitude = np.linalg.norm(car_to_closest)
    
    # Determine sign from cross product
    sign = np.sign(cross_product) if abs(cross_product) > 1e-10 else 0
    
    return sign * cte_magnitude


def check_collision(car, obstacles, path_points, car_radius=0.75, obstacle_shape='circle', 
                   ellipse_width=2.0, ellipse_height=6.0, ego_ellipse_width=3.0, ego_ellipse_height=1.5):
    """Check for collisions between car and obstacles.
    
    Args:
        car: Car object with x, y attributes
        obstacles: Array of obstacles [x, y, radius, velocity, path_index, ...]
        path_points: Array of raceline points for computing obstacle headings
        car_radius: Radius of the car in meters
        obstacle_shape: 'circle' or 'ellipse'
        ellipse_width: Width of ellipse (lateral) in meters
        ellipse_height: Height of ellipse (longitudinal) in meters
    
    Returns:
        tuple: (collision_detected: bool, colliding_indices: np.ndarray)
    """
    if len(obstacles) == 0:
        return False, np.array([], dtype=int)
    
    car_pos = np.array([car.x, car.y])
    
    if obstacle_shape == 'ellipse':
        # Ellipse-vs-ellipse collision check
        # Car ellipse: same dimensions as obstacles, oriented along car.yaw
        # Obstacle ellipse: oriented along path heading at obstacle position
        # We sample boundary points of the car ellipse and check if any fall inside the obstacle ellipse
        obs_positions = obstacles[:, :2]  # Extract x, y positions
        obs_path_indices = obstacles[:, 4].astype(int) if obstacles.shape[1] > 4 else np.zeros(len(obstacles), dtype=int)
        
        # Compute headings for all obstacles (vectorized)
        next_indices = (obs_path_indices + 1) % len(path_points)
        dx = path_points[next_indices, 0] - path_points[obs_path_indices, 0]
        dy = path_points[next_indices, 1] - path_points[obs_path_indices, 1]
        obs_headings = np.arctan2(dy, dx)
        
        # Car ellipse semi-axes: a_car = longitudinal, b_car = lateral
        a_car = ego_ellipse_width / 2.0
        b_car = ego_ellipse_height / 2.0
        car_yaw = car.yaw
        
        # Obstacle ellipse semi-axes
        a_obs = ellipse_height / 2.0
        b_obs = ellipse_width / 2.0
        
        # Sample points on the car ellipse boundary in car-local frame
        n_samples = 16
        angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)  # (S,)
        car_ellipse_local_x = a_car * np.cos(angles)  # (S,)
        car_ellipse_local_y = b_car * np.sin(angles)  # (S,)
        
        # Rotate car ellipse points to global frame
        cos_car = np.cos(car_yaw)
        sin_car = np.sin(car_yaw)
        car_pts_global_x = car_pos[0] + car_ellipse_local_x * cos_car - car_ellipse_local_y * sin_car  # (S,)
        car_pts_global_y = car_pos[1] + car_ellipse_local_x * sin_car + car_ellipse_local_y * cos_car  # (S,)
        
        # For each obstacle, transform car boundary points to obstacle-local frame
        # and check if any point falls inside the obstacle ellipse
        # Also check if obstacle boundary points fall inside car ellipse (for symmetry)
        n_obs = len(obstacles)
        cos_obs = np.cos(obs_headings)  # (N,)
        sin_obs = np.sin(obs_headings)  # (N,)
        
        # car_pts relative to each obstacle: (N, S)
        rel_x = car_pts_global_x[np.newaxis, :] - obs_positions[:, 0:1]  # (N, S)
        rel_y = car_pts_global_y[np.newaxis, :] - obs_positions[:, 1:2]  # (N, S)
        
        # Transform to obstacle-local frame
        local_x = rel_x * cos_obs[:, np.newaxis] + rel_y * sin_obs[:, np.newaxis]   # (N, S)
        local_y = -rel_x * sin_obs[:, np.newaxis] + rel_y * cos_obs[:, np.newaxis]  # (N, S)
        
        # Point-in-ellipse check for obstacle ellipse
        ellipse_test = (local_x / a_obs)**2 + (local_y / b_obs)**2  # (N, S)
        car_in_obs = np.any(ellipse_test <= 1.0, axis=1)  # (N,)
        
        # Sample obstacle ellipse boundary points and check against car ellipse
        obs_ellipse_local_x = a_obs * np.cos(angles)  # (S,)
        obs_ellipse_local_y = b_obs * np.sin(angles)  # (S,)
        
        # Rotate obstacle ellipse points to global frame for each obstacle: (N, S)
        obs_pts_global_x = obs_positions[:, 0:1] + obs_ellipse_local_x[np.newaxis, :] * cos_obs[:, np.newaxis] - obs_ellipse_local_y[np.newaxis, :] * sin_obs[:, np.newaxis]
        obs_pts_global_y = obs_positions[:, 1:2] + obs_ellipse_local_x[np.newaxis, :] * sin_obs[:, np.newaxis] + obs_ellipse_local_y[np.newaxis, :] * cos_obs[:, np.newaxis]
        
        # Transform obstacle boundary points to car-local frame
        obs_rel_x = obs_pts_global_x - car_pos[0]  # (N, S)
        obs_rel_y = obs_pts_global_y - car_pos[1]  # (N, S)
        obs_in_car_local_x = obs_rel_x * cos_car + obs_rel_y * sin_car    # (N, S)
        obs_in_car_local_y = -obs_rel_x * sin_car + obs_rel_y * cos_car   # (N, S)
        
        # Point-in-ellipse check for car ellipse
        car_ellipse_test = (obs_in_car_local_x / a_car)**2 + (obs_in_car_local_y / b_car)**2  # (N, S)
        obs_in_car = np.any(car_ellipse_test <= 1.0, axis=1)  # (N,)
        
        # Collision if any car boundary point is inside obstacle OR any obstacle boundary point is inside car
        collisions = car_in_obs | obs_in_car
        colliding_indices = np.where(collisions)[0]
    else:
        # Circle collision (original method)
        obs_positions = obstacles[:, :2]
        obs_radii = obstacles[:, 2]
        distances = np.linalg.norm(obs_positions - car_pos, axis=1)
        collision_thresholds = car_radius + obs_radii
        collisions = distances < collision_thresholds
        colliding_indices = np.where(collisions)[0]
    
    collision_detected = len(colliding_indices) > 0
    return collision_detected, colliding_indices


def check_boundary_violation(car, cte, closest_idx, inner_boundaries_distances, outer_boundaries_distances,
                            car_radius=0.75, trajectory_splines_inner_dist=None, trajectory_splines_outer_dist=None,
                            spline_path_length=None, prev_car_dist_to_path=None, prev_in_bounds=None,
                            prev_ignored_jump=False, jump_threshold=1.0, frame=None, dt=None):
    """Check if car is out of track boundaries.
    
    Args:
        car: Car object with x, y attributes
        cte: Cross-track error (signed distance from path)
        closest_idx: Index of closest point on raceline
        inner_boundaries_distances: Array of distances from raceline to inner boundary
        outer_boundaries_distances: Array of distances from raceline to outer boundary
        car_radius: Radius of the car in meters
        trajectory_splines_inner_dist: Optional spline for inner boundary distance
        trajectory_splines_outer_dist: Optional spline for outer boundary distance
        spline_path_length: Optional path length for spline parameterization
        prev_car_dist_to_path: Previous car distance to path (for jump detection)
        prev_in_bounds: Previous in-bounds state (for jump detection)
        prev_ignored_jump: Whether previous frame had an ignored jump
        jump_threshold: Threshold for detecting sudden jumps in meters
        frame: Current frame number (for logging)
        dt: Time step (for logging)
    
    Returns:
        tuple: (out_of_bounds: bool, actual_in_bounds: bool, should_ignore: bool, 
                car_dist_to_path: float, inner_dist_at_path: float, outer_dist_at_path: float)
    """
    # Get car's distance to the path (absolute distance)
    car_dist_to_path = abs(cte)
    
    # Get boundary distances at closest path point using splines if available
    if trajectory_splines_inner_dist is not None and trajectory_splines_outer_dist is not None and spline_path_length is not None:
        spline_param = (closest_idx / spline_path_length) % 1.0
        inner_dist_at_path = abs(trajectory_splines_inner_dist(spline_param))
        outer_dist_at_path = abs(trajectory_splines_outer_dist(spline_param))
    else:
        inner_dist_at_path = abs(inner_boundaries_distances[closest_idx])
        outer_dist_at_path = abs(outer_boundaries_distances[closest_idx])
    
    # Simple check: car is in bounds if its outer edge (accounting for car_radius) doesn't cross boundaries
    # CTE sign: negative = car on outer side, positive = car on inner side, zero = on path
    if cte > 0:
        # Car is on inner side - check if car's outer edge crosses inner boundary
        actual_in_bounds = (car_dist_to_path - car_radius) <= inner_dist_at_path
    else:
        # Car is on outer side - check if car's outer edge crosses outer boundary
        actual_in_bounds = (car_dist_to_path - car_radius) <= outer_dist_at_path
    
    # Additional checks: detect sudden jumps and ignore them unless they persist
    should_ignore = False
    currently_in_bounds = actual_in_bounds  # Start with actual state
    
    if prev_car_dist_to_path is not None and prev_in_bounds is not None:
        # Check for sudden jump in distance
        dist_change = abs(car_dist_to_path - prev_car_dist_to_path)
        
        # Check for sudden state change (from in-bounds to out-of-bounds)
        state_change = (prev_in_bounds and not actual_in_bounds)
        
        # If there's a sudden jump in distance AND state change, it might be a glitch
        if state_change and dist_change > jump_threshold:
            # If previous frame had an ignored jump and this frame is also out of bounds, count it
            if prev_ignored_jump and not actual_in_bounds:
                # Previous jump was ignored, but it persists - count it now
                should_ignore = False
                if frame is not None and frame % 10 == 0:
                    print(f"PERSISTENT JUMP CONFIRMED at t={frame*dt:.3f}s: "
                          f"dist_change={dist_change:.3f}m, prev_dist={prev_car_dist_to_path:.3f}m, "
                          f"curr_dist={car_dist_to_path:.3f}m")
            else:
                # This is a sudden jump - ignore it for now
                should_ignore = True
                currently_in_bounds = True  # Temporarily ignore this frame for counting
                if frame is not None and frame % 10 == 0:  # Print every 10 frames to avoid spam
                    print(f"IGNORED SUDDEN JUMP at t={frame*dt:.3f}s: "
                          f"dist_change={dist_change:.3f}m, prev_dist={prev_car_dist_to_path:.3f}m, "
                          f"curr_dist={car_dist_to_path:.3f}m")
    
    out_of_bounds = not currently_in_bounds and not should_ignore
    
    return out_of_bounds, actual_in_bounds, should_ignore, car_dist_to_path, inner_dist_at_path, outer_dist_at_path


def update_overtake_tracking(obstacles, s_obs_values, overtake_states, prev_s_obs, successful_overtakes):
    """Update overtake tracking state machine for each obstacle.
    
    Overtake sequence: obstacle >= 20m ahead -> passes ego (s < 0) -> >= 10m behind (s <= -10) for tight racing
    
    Args:
        obstacles: Array of obstacles
        s_obs_values: Array of s_obs (longitudinal position) for each obstacle
        overtake_states: Array of current state for each obstacle (0=not started, 1=tracking, 2=passed 0, 3=completed)
        prev_s_obs: Array of previous s_obs values for each obstacle
        successful_overtakes: Current count of successful overtakes
    
    Returns:
        tuple: (updated_overtake_states, updated_prev_s_obs, new_successful_overtakes)
    """
    if len(obstacles) == 0:
        return overtake_states, prev_s_obs, successful_overtakes
    
    updated_overtake_states = overtake_states.copy()
    updated_prev_s_obs = prev_s_obs.copy()
    new_successful_overtakes = successful_overtakes
    
    for obs_idx in range(len(obstacles)):
        if obs_idx >= len(updated_overtake_states):
            continue
        
        s_obs = s_obs_values[obs_idx]
        
        # Validate s_obs (skip if invalid)
        if not np.isfinite(s_obs):
            continue
        
        state = updated_overtake_states[obs_idx]
        prev_s = prev_s_obs[obs_idx]
        
        # State machine for overtake tracking
        if state == 0:  # Not started - wait for obstacle to be at least 100m ahead
            if s_obs >= 20.0:
                updated_overtake_states[obs_idx] = 1  # Start tracking
        
        elif state == 1:  # Tracking - obstacle was >= 20 ahead, waiting for it to pass (s < 0)
            if s_obs < 0.0:
                # Check if this is a wrap: if prev_s was large positive (>100m) and now negative, it's wrapping
                if np.isfinite(prev_s) and prev_s > 100.0:
                    # Obstacle was far ahead and now appears behind - this is wrapping, not a pass
                    # Reset to state 0 to start fresh
                    updated_overtake_states[obs_idx] = 0
                else:
                    # Obstacle passed ego car, move to state 2
                    updated_overtake_states[obs_idx] = 2
            elif s_obs < 20.0:
                # Obstacle got closer but still ahead, keep tracking
                pass
            # If s_obs >= 100 still, keep in state 1
            # Reset if obstacle goes too far ahead (might be a different obstacle or glitch)
            if s_obs > 50.0:
                # Reset if obstacle is very far ahead (might be measurement error or different obstacle)
                updated_overtake_states[obs_idx] = 0
        
        elif state == 2:  # Passed 0 - waiting for obstacle to be at least 10m behind (s <= -10) for tight racing
            # Use -10m threshold for tight racing to count overtake sooner and prevent comeback
            if s_obs <= -10.0:
                # Full sequence completed: >= 20 -> < 0 -> <= -10
                updated_overtake_states[obs_idx] = 3  # Completed
                new_successful_overtakes += 1
            elif s_obs >= 0.0:
                # Obstacle came back ahead before completing
                # For tight racing, if it was already behind (prev_s_obs < 0), count it as overtaken
                # This handles the case where obstacle passes, comes back briefly, then gets overtaken again
                if prev_s_obs[obs_idx] < -5.0:  # Was at least 5m behind before coming back
                    # Count as overtaken since it was significantly behind
                    updated_overtake_states[obs_idx] = 3
                    new_successful_overtakes += 1
                else:
                    # Reset to state 0 if it came back before being significantly behind
                    updated_overtake_states[obs_idx] = 0
            # If -10 < s_obs < 0, keep in state 2 (waiting for obstacle to be further behind)
        
        elif state == 3:  # Completed - reset if obstacle comes back ahead to >= 20m for another overtake
            if s_obs >= 20.0:
                # Check if this is a wrap: if prev_s was negative and now large positive, it's wrapping
                if np.isfinite(prev_s) and prev_s < -10.0:
                    # Obstacle was behind and now appears far ahead - this is wrapping, ignore
                    pass
                else:
                    # Obstacle legitimately came back ahead, reset to tracking for another overtake
                    updated_overtake_states[obs_idx] = 1
            # If obstacle is between -10 and 20, keep in completed state
        
        # Update previous s_obs for next iteration
        updated_prev_s_obs[obs_idx] = s_obs
    
    return updated_overtake_states, updated_prev_s_obs, new_successful_overtakes


def compute_s_obs_for_obstacles(obstacles, car, path_points, car_closest_idx, cumulative_distances=None, total_path_length=None):
    """Compute s_obs (longitudinal position in Frenet frame) for each obstacle.
    Uses pre-computed cumulative distances for fast arc length computation.
    Optimized for computational efficiency.
    
    Args:
        obstacles: Array of obstacles [x, y, radius, velocity, path_index, ...]
        car: Car object with x, y, yaw
        path_points: Full raceline path points [N, 3+] where columns are [x, y, ...]
        car_closest_idx: Index of closest point on raceline to ego car
        cumulative_distances: Pre-computed cumulative distances along path (optional, computed if None)
        total_path_length: Total path length (optional, computed if None)
    
    Returns:
        Array of s_obs values for each obstacle (distance in meters, positive = ahead, negative = behind)
    """
    if len(obstacles) == 0:
        return np.array([])
    
    # Extract path points (x, y coordinates)
    path_xy = path_points[:, :2]  # [N, 2]
    n_path = len(path_xy)
    
    # Pre-compute cumulative distances if not provided (cache globally)
    global path_cumulative_distances_cache, path_total_length_cache, cached_path_for_distances
    if cumulative_distances is None:
        # Check if we have cached distances for this path
        if ('path_cumulative_distances_cache' not in globals() or 
            'cached_path_for_distances' not in globals() or 
            not np.array_equal(cached_path_for_distances, path_points)):
            # Compute cumulative distances once (optimized)
            path_cumulative_distances_cache = np.zeros(n_path + 1)
            # Vectorized computation of segment lengths for most segments
            if n_path > 1:
                # Compute distances between consecutive points
                dx = path_xy[1:, 0] - path_xy[:-1, 0]
                dy = path_xy[1:, 1] - path_xy[:-1, 1]
                segment_lengths = np.sqrt(dx*dx + dy*dy)
                path_cumulative_distances_cache[1:n_path] = np.cumsum(segment_lengths)
            
            # Close the loop (distance from last point to first point)
            if n_path > 0:
                dx = path_xy[0, 0] - path_xy[-1, 0]
                dy = path_xy[0, 1] - path_xy[-1, 1]
                path_cumulative_distances_cache[n_path] = path_cumulative_distances_cache[n_path-1] + np.sqrt(dx*dx + dy*dy)
            path_total_length_cache = path_cumulative_distances_cache[n_path]
            cached_path_for_distances = path_points.copy()
        cumulative_distances = path_cumulative_distances_cache
        total_path_length = path_total_length_cache
    
    # Car's distance along path
    car_distance = cumulative_distances[car_closest_idx]
    
    # Extract obstacle positions and path indices (vectorized)
    obs_positions = obstacles[:, :2]  # [N_obs, 2]
    obs_path_indices = obstacles[:, 4].astype(int) if obstacles.shape[1] > 4 else np.zeros(len(obstacles), dtype=int)
    
    # Find closest point on path for each obstacle (optimized with search window)
    search_window = 100  # Search within ±100 points for speed
    obs_closest_indices = np.zeros(len(obstacles), dtype=int)
    
    # Vectorized search for obstacles with valid path indices
    has_path_idx = (obs_path_indices >= 0) & (obs_path_indices < n_path)
    
    for i in range(len(obstacles)):
        obs_x, obs_y = obs_positions[i]
        
        if has_path_idx[i]:
            obs_path_idx = obs_path_indices[i]
            # Create search window with wrapping (optimized)
            search_start = (obs_path_idx - search_window) % n_path
            search_end = (obs_path_idx + search_window + 1) % n_path
            
            if search_start < search_end:
                # No wrap: simple range
                search_indices = np.arange(search_start, search_end, dtype=int)
            else:
                # Wraps around: combine two ranges (more efficient than concatenate)
                search_indices = np.empty(search_window * 2 + 1, dtype=int)
                n1 = n_path - search_start
                search_indices[:n1] = np.arange(search_start, n_path, dtype=int)
                search_indices[n1:] = np.arange(0, search_end, dtype=int)
            
            # Compute distances only for search window (vectorized)
            search_points = path_xy[search_indices]
            distances_to_path = np.sqrt((search_points[:, 0] - obs_x)**2 + (search_points[:, 1] - obs_y)**2)
            local_min_idx = np.argmin(distances_to_path)
            obs_closest_indices[i] = search_indices[local_min_idx]
        else:
            # Full search (slower, but only if no path index available)
            distances_to_path = np.sqrt((path_xy[:, 0] - obs_x)**2 + (path_xy[:, 1] - obs_y)**2)
            obs_closest_indices[i] = np.argmin(distances_to_path)
    
    # Compute s_obs for each obstacle using pre-computed distances (vectorized where possible)
    obs_distances = cumulative_distances[obs_closest_indices]
    
    # Vectorized computation of forward and backward distances
    # Forward distance: going forward along path from car to obstacle
    # Backward distance: going backward along path from car to obstacle
    idx_diff = obs_closest_indices - car_closest_idx
    
    # Cases:
    # 1. obs_idx > car_idx: forward = obs_dist - car_dist, backward = (total - obs_dist) + car_dist
    # 2. obs_idx < car_idx: forward = (total - car_dist) + obs_dist, backward = car_dist - obs_dist
    # 3. obs_idx == car_idx: need to check actual position along path
    
    # Vectorized computation
    forward_dist = np.where(
        idx_diff > 0,
        obs_distances - car_distance,  # Case 1: simple forward
        np.where(
            idx_diff < 0,
            (total_path_length - car_distance) + obs_distances,  # Case 2: forward wraps
            np.zeros_like(obs_distances)  # Case 3: same index, will handle separately
        )
    )
    
    backward_dist = np.where(
        idx_diff > 0,
        (total_path_length - obs_distances) + car_distance,  # Case 1: backward wraps
        np.where(
            idx_diff < 0,
            car_distance - obs_distances,  # Case 2: simple backward
            np.zeros_like(obs_distances)  # Case 3: same index, will handle separately
        )
    )
    
    # Handle same-index case: use projection onto path tangent to determine direction
    same_idx_mask = (idx_diff == 0)
    if np.any(same_idx_mask):
        # Compute heading at car position
        next_idx = (car_closest_idx + 1) % n_path
        dx_ref = path_xy[next_idx, 0] - path_xy[car_closest_idx, 0]
        dy_ref = path_xy[next_idx, 1] - path_xy[car_closest_idx, 1]
        ref_yaw = np.arctan2(dy_ref, dx_ref)
        cos_yaw = np.cos(ref_yaw)
        sin_yaw = np.sin(ref_yaw)
        
        # Project obstacle positions onto path tangent (vectorized)
        same_idx_indices = np.where(same_idx_mask)[0]
        ref_point = path_xy[car_closest_idx]
        obs_rel = obs_positions[same_idx_indices] - ref_point
        proj_dists = obs_rel[:, 0] * cos_yaw + obs_rel[:, 1] * sin_yaw
        
        # Use small threshold to determine if ahead or behind
        for idx, proj_dist in zip(same_idx_indices, proj_dists):
            if abs(proj_dist) < 0.1:  # Very close, use 0
                forward_dist[idx] = 0.0
                backward_dist[idx] = 0.0
            elif proj_dist > 0:
                # Ahead: use small forward distance
                forward_dist[idx] = abs(proj_dist)
                backward_dist[idx] = total_path_length - abs(proj_dist)
            else:
                # Behind: use small backward distance
                forward_dist[idx] = total_path_length - abs(proj_dist)
                backward_dist[idx] = abs(proj_dist)
    
    # Choose the shorter path, but use sign to indicate direction
    # Positive = ahead (forward), Negative = behind (backward)
    s_obs_values = np.where(
        forward_dist <= backward_dist,
        forward_dist,
        -backward_dist
    )
    
    # Validate s_obs (handle NaN/inf)
    s_obs_values = np.where(np.isfinite(s_obs_values), s_obs_values, 0.0)
    
    return s_obs_values