class Controller:
    """Base class for controllers."""
    def update(self, objects, dt):
        """Compute control commands for each object; to be implemented by subclasses."""
        raise NotImplementedError

class PIDController(Controller):
    def __init__(self, kp, ki, kd, setpoint=0.0, dt=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.dt = dt
        self.previous_error = 0
        self.integral = 0
        
    def update(self, current_value, setpoint):
        error = setpoint - current_value
        self.integral += error * self.dt
        derivative = (error - self.previous_error) / self.dt
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative