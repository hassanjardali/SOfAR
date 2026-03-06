import numpy as np


# PID Controller for velocity
class PIDController:
    def __init__(self, params_pid ):
        self.kp = params_pid['kp']  # Proportional gain
        self.ki = params_pid['ki']  # Integral gain
        self.kd = params_pid['kd']  # Derivative gain
        self.max_integral = params_pid['max_integral']  # Maximum value for the integral term to prevent windup
        self.min_output = params_pid['min_output']  # Minimum output value (for negative acceleration)
        self.max_output = params_pid['max_output']  # Maximum output value (for positive acceleration)
        self.deadzone = params_pid['deadzone']  # Error deadzone for zero output
        self.integral = 0  # Integral component of the PID
        self.prev_error = 0  # Previous error for calculating derivative
        self.prev_output = 0  # Store previous output for smoothing

    def compute(self, target, current, dt, feedforward=0.0):
        """
        Compute the PID control action.
        
        Parameters:
        - target (float): Desired velocity (m/s).
        - current (float): Current velocity (m/s).
        - dt (float): Time step (seconds).
        - feedforward (float): Feedforward term for acceleration (default=0.0).
        
        Returns:
        - output (float): Acceleration to be applied.
        """
        # Calculate error
        error = target - current
        
        # Check if error is within the deadzone
        if abs(error) < self.deadzone:
            return 0.0

        # Calculate integral with anti-windup
        self.integral += error * dt
        self.integral = max(min(self.integral, self.max_integral), -self.max_integral)  # Anti-windup

        # Calculate derivative
        derivative = (error - self.prev_error) / dt

        # PID output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative + feedforward

        # Smooth saturation: instead of hard clipping, we gradually reduce the output
        if output > self.max_output:
            output = self.max_output
        elif output < self.min_output:
            output = self.min_output 

        # Update previous error and output
        self.prev_error = error
        self.prev_output = output

        return output


# Pure Pursuit Controller for steering
class PurePursuitController:
    def __init__(self, base_lookahead_distance=5.0, velocity_factor=0.5, max_lookahead_distance=15.0, min_lookahead_distance=5.0):
        self.base_lookahead_distance = base_lookahead_distance
        self.velocity_factor = velocity_factor
        self.N = 0
        self.max_lookahead_distance = max_lookahead_distance
        self.min_lookahead_distance = min_lookahead_distance

    def get_lateral_control(self, car, path):
        """
        Compute the steering angle based on pure pursuit algorithm.
        :param car: Current car state (CarKinematicModel)
        :param path: Subset of path points as (x, y) pairs.
        """
        # Adjust lookahead distance based on the current velocity
        self.lookahead_distance =  np.clip(self.base_lookahead_distance + car.velocity * self.velocity_factor, self.min_lookahead_distance, self.max_lookahead_distance)
        
        min_distance = float('inf')
        target_point = None
        
        # Find the closest lookahead point on the provided path segment
        for point in path:
            distance = np.sqrt((point[0] - car.x)**2 + (point[1] - car.y)**2)
            if abs(distance - self.lookahead_distance) < min_distance:
                min_distance = abs(distance - self.lookahead_distance)
                target_point = point

        # If no valid target point is found, return zero steering angle
        if target_point is None:
            return 0.0
        
        # Calculate the steering angle based on the target point
        alpha = np.arctan2(target_point[1] - car.y, target_point[0] - car.x) - car.yaw
        steering_angle = np.arctan2(2 * car.wheel_base * np.sin(alpha), self.lookahead_distance)
        return steering_angle, None
