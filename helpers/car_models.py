import numpy as np

# Car Kinematic Model
class CarKinematicModel:
    def __init__(self, params):
        self.params = params

        # States
        self.x = params['initial_state']['initial_x']
        self.y = params['initial_state']['initial_y']
        self.yaw = params['initial_state']['initial_yaw']
        self.velocity = params['initial_state']['initial_velocity']

        # Other Params
        self.wheel_base = params['wheel_base']
    
    def update_state(self, throttle, steering_angle, dt):
        """
        Update the state of the car using a simple kinematic bicycle model.
        :param throttle: Acceleration (m/s^2)
        :param steering_angle: Steering angle (radians)
        :param dt: Time step (s)
        """
        # Update velocity (simplified with no drag or resistance)
        self.velocity += throttle * dt

        # Update position and heading
        self.x += self.velocity * np.cos(self.yaw) * dt
        self.y += self.velocity * np.sin(self.yaw) * dt
        self.yaw += (self.velocity / self.wheel_base) * np.tan(steering_angle) * dt
        
        # Wrap yaw angle between -pi and pi
        self.yaw = (self.yaw + np.pi) % (2 * np.pi) - np.pi


# Single Track Dynamic Non-linear Model
class CarDynamicModel:
    def __init__(self, params):
        self.params = params

        # States
        self.x = params['initial_state']['initial_x']
        self.y = params['initial_state']['initial_y']
        self.psi = params['initial_state']['initial_psi']
        self.vx = params['initial_state']['initial_vx']
        self.vy = params['initial_state']['initial_vy']
        self.psidot = params['initial_state']['initial_psidot']
        self.delta = params['initial_state']['initial_delta']

        self.velocity = np.sqrt(self.vx**2 + self.vy**2)
        self.yaw = self.psi

        # Slip angles
        self.slip_angle_front = params['initial_state']['initial_slip_angle_front']
        self.slip_angle_rear = params['initial_state']['initial_slip_angle_rear']

        # Forces
        self.F_x_front = params['initial_state']['initial_F_x_front']
        self.F_x_rear = params['initial_state']['initial_F_x_rear']
        self.F_y_front = params['initial_state']['initial_F_y_front']
        self.F_y_rear = params['initial_state']['initial_F_y_rear']

        # Params (add h_cg for future load transfer)
        self.mass = params['mass']
        self.Iz = params['Iz']
        self.lf = params['lf']
        self.lr = params['lr']
        self.g = params['g']
        self.mu = params['mu']
        self.brake_bias = params['brake_bias']
        self.max_steering_angle = params['max_steering_angle']
        self.max_steering_rate = params['max_steering_rate']
        self.Bf, self.Cf, self.Ef = params['Bf'], params['Cf'], params['Ef']
        self.Br, self.Cr, self.Er = params['Br'], params['Cr'], params['Er']
        self.wheel_base = params['wheel_base']

        # Static loads (extend to dynamic later)
        self.Fz_front = self.mass * self.g * self.lr / (self.lf + self.lr)
        self.Fz_rear = self.mass * self.g * self.lf / (self.lf + self.lr)
        self.D_front = self.mu * self.Fz_front
        self.D_rear = self.mu * self.Fz_rear
    
    def pacejka_front(self, alpha):
        return self.D_front * np.sin(self.Cf * np.arctan(self.Bf * alpha - self.Ef * (self.Bf * alpha - np.arctan(self.Bf * alpha))))
    
    def pacejka_rear(self, alpha):
        return self.D_rear * np.sin(self.Cr * np.arctan(self.Br * alpha - self.Er * (self.Br * alpha - np.arctan(self.Br * alpha))))
    
    def longitudinal_forces(self, long_accel_cmd):
        if long_accel_cmd >= 0:
            Fx_front = 0.0
            Fx_rear = self.mass * long_accel_cmd
        else:
            total_brake = self.mass * long_accel_cmd
            Fx_front = total_brake * self.brake_bias
            Fx_rear = total_brake * (1.0 - self.brake_bias)
        # Clip to friction limits (simple approximation)
        Fx_front = np.clip(Fx_front, -self.mu * self.Fz_front, self.mu * self.Fz_front)
        Fx_rear = np.clip(Fx_rear, -self.mu * self.Fz_rear, self.mu * self.Fz_rear)
        return Fx_front, Fx_rear

    # Derivative function: computes dstate/dt given current state vector and inputs
    def deriv(self, state, long_accel_cmd):  # Add long_accel_cmd as param
        x, y, psi, vx, vy, psidot = state
        
        # Recompute steering projections (delta is assumed constant over dt)
        cos_d = np.cos(self.delta)
        sin_d = np.sin(self.delta)
        
        # Recompute slips and forces
        denom = max(abs(vx), 1e-6)
        slip_angle_front = np.clip(self.delta - np.arctan((vy + self.lf * psidot) / denom), -self.max_steering_angle/2, self.max_steering_angle/2)
        slip_angle_rear = np.clip(-np.arctan((vy - self.lr * psidot) / denom), -self.max_steering_angle/2, self.max_steering_angle/2)
        Fy_front = self.pacejka_front(slip_angle_front)
        Fy_rear = self.pacejka_rear(slip_angle_rear)
        Fx_front, Fx_rear = self.longitudinal_forces(long_accel_cmd)
        
        # Dynamics
        vx_dot = psidot * vy + (1 / self.mass) * (Fx_front * cos_d - Fy_front * sin_d + Fx_rear)
        vy_dot = -psidot * vx + (1 / self.mass) * (Fy_front * cos_d + Fx_front * sin_d + Fy_rear)
        psidot_dot = (1 / self.Iz) * (self.lf * (Fy_front * cos_d + Fx_front * sin_d) - self.lr * Fy_rear)
        
        X_dot = vx * np.cos(psi) - vy * np.sin(psi)
        Y_dot = vx * np.sin(psi) + vy * np.cos(psi)
        psi_dot = psidot
        
        return np.array([X_dot, Y_dot, psi_dot, vx_dot, vy_dot, psidot_dot])
        
    def update_state(self, long_accel_cmd, steering_angle_cmd, dt):
        # Steering update (unchanged)
        self.delta = np.clip(steering_angle_cmd, self.delta - self.max_steering_rate * dt, self.delta + self.max_steering_rate * dt)
        self.delta = np.clip(self.delta, -self.max_steering_angle, self.max_steering_angle)
        
        # Current state
        state = np.array([self.x, self.y, self.psi, self.vx, self.vy, self.psidot])
        
        # RK4 with recomputed derivs (pass long_accel_cmd)
        k1 = self.deriv(state, long_accel_cmd)
        k2 = self.deriv(state + (dt / 2) * k1, long_accel_cmd)
        k3 = self.deriv(state + (dt / 2) * k2, long_accel_cmd)
        k4 = self.deriv(state + dt * k3, long_accel_cmd)
        
        # Update state
        delta_state = (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        new_state = state + delta_state
        
        # Unpack
        self.x, self.y, self.psi, self.vx, self.vy, self.psidot = new_state
        
        # Update derived vars (and slips/forces for logging, based on final state)
        self.velocity = np.sqrt(self.vx**2 + self.vy**2)
        self.yaw = self.psi
        denom = max(abs(self.vx), 1e-6)
        self.slip_angle_front = self.delta - np.arctan((self.vy + self.lf * self.psidot) / denom)
        self.slip_angle_rear = -np.arctan((self.vy - self.lr * self.psidot) / denom)
        self.F_y_front = self.pacejka_front(self.slip_angle_front)
        self.F_y_rear = self.pacejka_rear(self.slip_angle_rear)
        self.F_x_front, self.F_x_rear = self.longitudinal_forces(long_accel_cmd)



