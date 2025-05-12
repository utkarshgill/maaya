# Interactive target entry needs threading
import threading, sys

import numpy as np
# Ensure project root and src are on PYTHONPATH regardless of where script is launched
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent  # project root
_SRC = _ROOT / 'src'
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from maaya import Vector3D, Body, World, Renderer, NoiseGenerator, GravitationalForce

class QuadCopter(Body):
    def __init__(self, position=None, velocity=None, acceleration=None, mass=1.0,
                 orientation=None, angular_velocity=None, ctrl=None, board=None):
        L = 0.3  # Length of each arm from center to tip
        num_arms = 4

        # Calculate the mass of each arm assuming equal distribution
        m_arm = mass / num_arms

        # Calculate the moments of inertia
        # Inertia about the Z-axis, assuming arms act like rods rotating around their center
        I_z = num_arms * (1/12) * m_arm * (L**2)

        # Inertia about the X and Y axes
        # Considering arms are at 45 degrees, projecting to L*cos(45 degrees) for each axis
        L_projected = L * np.cos(np.pi / 4)  # cos(45 degrees) = sqrt(2)/2
        I_x = num_arms * (1/3) * m_arm * (L_projected**2)
        I_y = I_x  # Symmetry in the configuration

        inertia = np.array([
            [I_x, 0, 0],
            [0, I_y, 0],
            [0, 0, I_z]
        ])
       
        self.ctrl = ctrl
        self.board = board

        super().__init__(position, velocity, acceleration, mass, orientation, angular_velocity, inertia)
        

    # PID stability loop

    def update(self, dt):
        # Position in world frame
        x_pos, y_pos, z_pos = self.position.v
        # Current attitude expressed as Euler angles
        r, p, y = self.orientation.to_euler()

        # Pass full state to the controller (x, y, z, roll, pitch, yaw)
        T, R, P, Y = self.ctrl.update(x_pos, y_pos, z_pos, r, p, y)

        self.command([T, R, P, Y], dt)
        super().update(dt)


  

    def command(self, c, dt):
        T, R, P, Y = c

        # Apply thrust in the body +Z axis, rotated to world frame
        thrust_vector = self.orientation.rotate(Vector3D(0, 0, T))
        self.apply_torque(Vector3D(R, P, Y), dt=dt)
        self.apply_force(thrust_vector)

    def __repr__(self):
        return (f"QuadCopter(position={self.position}, velocity={self.velocity}, "
                f"acceleration={self.acceleration}, mass={self.mass}, "
                f"orientation={self.orientation})")

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0.0, dt=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.dt = dt
        self.previous_error = 0
        self.integral = 0
        
    def update(self, current_value):
        """Compute PID output.

        Parameters
        ----------
        current_value : float
            The current measurement of the process variable.
        setpoint : float | None, optional
            If provided, overrides the controller's internal set-point for
            this call. This allows outer-loop controllers to supply dynamic
            setpoints without recreating the PID object each iteration.
        """

        # Support both the old signature (current_value) and the new
        # signature (current_value, setpoint=...).
        if isinstance(current_value, tuple) or isinstance(current_value, list):
            # Backwards compatibility path when caller passes
            # (current_value, setpoint)
            # Detect call pattern PID.update(value, setpoint)
            current_value, maybe_setpoint = current_value
            setpoint = maybe_setpoint
        else:
            setpoint = None

        if setpoint is not None:
            self.setpoint = setpoint

        error = self.setpoint - current_value

        self.integral += error * self.dt
        derivative = (error - self.previous_error) / self.dt

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.previous_error = error
        return output
    
class DroneController:
    def __init__(self):
        # PID controllers for roll, pitch, yaw, and altitude
        self.x_pid = PIDController(kp=0.2, ki=0.0, kd=0.3, setpoint=2.0)
        self.y_pid = PIDController(kp=0.2, ki=0.0, kd=0.3, setpoint=2.0)
        self.altitude_pid = PIDController(kp=1.5, ki=0.2, kd=3.0, setpoint=10.0)
        # Inner‐loop attitude gains tuned for quick but damped response
        self.roll_pid  = PIDController(kp=2.0, ki=0.0, kd=0.3)
        self.pitch_pid = PIDController(kp=2.0, ki=0.0, kd=0.3)
        self.yaw_pid = PIDController(kp=1.0, ki=0.0, kd=0.1)

    def update(self, x, y, z, roll, pitch, yaw):
        # Outer‐loop PIDs
        pitch_set =  np.clip(self.x_pid.update(x),  -0.3, 0.3)   # tilt forward/back
        roll_set  =  np.clip(-self.y_pid.update(y), -0.3, 0.3)   # tilt left/right
        # Gravity compensation: hover thrust baseline ≈ m·g (here m=1 kg).
        gravity_ff = 9.8  # N
        thrust     = gravity_ff + self.altitude_pid.update(z)
        thrust     = float(np.clip(thrust, 0.0, 20.0))
        # Inner‐loop: drive attitude PIDs toward those setpoints
        roll_cmd  =  self.roll_pid.update((roll,  roll_set))
        pitch_cmd =  self.pitch_pid.update((pitch, pitch_set))
        yaw_cmd   =  self.yaw_pid.update(yaw)

        # Constrain generated torques to reasonable bounds to prevent instability
        roll_cmd  = float(np.clip(roll_cmd,  -0.5, 0.5))
        pitch_cmd = float(np.clip(pitch_cmd, -0.5, 0.5))
        yaw_cmd   = float(np.clip(yaw_cmd,   -0.3, 0.3))
        return [thrust, roll_cmd, pitch_cmd, yaw_cmd]

    def set_xy_target(self, x_target, y_target):
        """Dynamically update horizontal position set‐points."""
        self.x_pid.setpoint = x_target
        self.y_pid.setpoint = y_target

frames = 1000
world = World(noise=NoiseGenerator(intensity=0.02), gravity=GravitationalForce())
# z_ctrl = PIDController(10.0, 10.0, 5.0, setpoint=10.0, dt=0.01)
ctrl = DroneController()

quad = QuadCopter(position=Vector3D(0, 0, 10.0), mass=1.0, ctrl=ctrl)

world.add_object(quad)

# ---------------------------------------------------------------------------
# Spawn a background thread that listens for "x y" pairs typed in the terminal
# and updates the controller's horizontal targets on the fly.

def _stdin_reader(ctrl_ref):
    print("Type new x y set-points (e.g. '2.0 3.5') and press <Enter>.  Ctrl-D to stop input.")
    for line in sys.stdin:
        parts = line.strip().split()
        if len(parts) != 2:
            print("Please enter exactly two numbers separated by space.")
            continue
        try:
            x_t, y_t = map(float, parts)
        except ValueError:
            print("Invalid input. Expect numeric values.")
            continue
        ctrl_ref.set_xy_target(x_t, y_t)
        print(f"→ Updated targets: x={x_t:.2f}, y={y_t:.2f}")

# Start reader BEFORE launching the blocking Matplotlib event loop
threading.Thread(target=_stdin_reader, args=(ctrl,), daemon=True).start()

r = Renderer(world)
r.run(frames)


