class Controller:
    """Base class for controllers."""
    def update(self, body, dt):
        """Compute and return a control command for the given body at this timestep."""
        raise NotImplementedError

class PIDController(Controller):
    def __init__(self, kp, ki, kd, setpoint=0.0, dt=0.01, measurement_fn=None):
        """
        A simple PID controller that by default regulates altitude (z position).
        Args:
            kp, ki, kd: PID gains
            setpoint: desired target value
            dt: timestep for internal integral/derivative
            measurement_fn: optional fn(body) -> float, to extract the current value
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.dt = dt
        self.previous_error = 0.0
        self.integral = 0.0
        # default measurement is altitude (z position)
        self.measurement_fn = (measurement_fn if measurement_fn is not None
                               else (lambda body: body.position.v[2]))

    def update(self, body, dt):
        """Compute PID command given the body's state and return a scalar control command."""
        # use provided dt for integration and derivative
        error = self.setpoint - self.measurement_fn(body)
        # integral term
        self.integral += error * dt
        # derivative term
        derivative = ((error - self.previous_error) / dt) if dt > 0.0 else 0.0
        self.previous_error = error
        # PID output
        return self.kp * error + self.ki * self.integral + self.kd * derivative