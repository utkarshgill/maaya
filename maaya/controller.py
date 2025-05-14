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
        
    def update(self, current_value, setpoint, dt=None):
        """Return PID command.

        Parameters
        ----------
        current_value : float
            The measured process variable.
        setpoint : float
            Desired target value.
        dt : float | None, optional
            Timestep in **seconds** since last call.  If *None* the instance-
            level ``self.dt`` fallback is used.  Supplying the runtime *dt*
            makes the controller independent of the integrator step size and
            avoids derivative kick when the simulation time step changes.
        """

        # Fall back to the stored dt to preserve backwards compatibility
        dt = self.dt if dt is None else dt

        error = setpoint - current_value

        # Integral term (anti-windup could be added here)
        self.integral += error * dt

        # Derivative term – first iteration uses stored previous_error (0)
        derivative = (error - self.previous_error) / dt if dt > 0 else 0.0

        # Update memory
        self.previous_error = error

        # PID output
        return self.kp * error + self.ki * self.integral + self.kd * derivative