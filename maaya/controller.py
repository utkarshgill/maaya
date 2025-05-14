class Controller:
    """Base class for controllers."""
    def update(self, objects, dt):
        """Compute control commands for each object; to be implemented by subclasses."""
        raise NotImplementedError

class PIDController(Controller):
    """PID controller example for altitude hold."""
    def __init__(self, kp, ki, kd, setpoint=0.0):
        """
        Args:
            kp: proportional gain
            ki: integral gain
            kd: derivative gain
            setpoint: desired altitude (m)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integrator = 0.0
        self.prev_error = 0.0

    def update(self, objects, dt):
        """Update control command for each object based on altitude error."""
        for obj in objects:
            # Altitude control based on z-position
            error = self.setpoint - obj.position.v[2]
            self.integrator += error * dt
            derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
            self.prev_error = error
            # Compute thrust command
            thrust_cmd = self.kp * error + self.ki * self.integrator + self.kd * derivative
            obj.control_command = thrust_cmd 