import numpy as np
from .math import Vector3D, Quaternion
from .physics import EulerIntegrator

class Body:
    def __init__(self, position=None, velocity=None, acceleration=None, mass=1.0,
                 orientation=None, angular_velocity=None, inertia=None, integrator=None):
        """Rigid body with 6-DOF state.

        Parameters
        ----------
        inertia : np.ndarray, optional
            3×3 inertia matrix about the body frame origin. Defaults to identity.
        integrator : object, optional
            An integrator instance implementing ``step(body, dt)``. Defaults to
            ``EulerIntegrator``.
        """
        self.position = position if position is not None else Vector3D()
        self.velocity = velocity if velocity is not None else Vector3D()
        self.acceleration = acceleration if acceleration is not None else Vector3D()
        self.mass = mass
        self.orientation = orientation if orientation is not None else Quaternion()
        # Angular velocity is stored as a pure 3-vector (rad/s)
        self.angular_velocity = angular_velocity if angular_velocity is not None else Vector3D()

        self.inertia = inertia if inertia is not None else np.eye(3)
        # Cache the inverse once; many calls avoid repeated inv() operations
        self.inertia_inv = np.linalg.inv(self.inertia)

        # Allow different integration schemes to be swapped in
        self.integrator = integrator if integrator is not None else EulerIntegrator()

        # Per-body actuator collection; populated externally (e.g., by World)
        self.actuators = []
        # Per-body sensors and controllers (moved from World)
        self.sensors = []
        self.controllers = []

    def apply_torque(self, torque, dt=0.01):
        """Update angular velocity given a torque vector.

        Parameters
        ----------
        torque : Vector3D
            Torque expressed in the body frame (N·m).
        dt : float, optional
            Timestep size in seconds. Defaults to 0.01.
        """
        # α = I⁻¹ τ
        angular_acceleration = self.inertia_inv.dot(torque.v)
        # Integrate to update angular velocity (simple Euler)
        self.angular_velocity += Vector3D(*angular_acceleration) * dt

    def update(self, dt):
        self.integrator.step(self, dt)

    def apply_force(self, force):
        # F = m * a, therefore a = F / m
        self.acceleration += Vector3D(*(force.v / self.mass))

    def __repr__(self):
        return f"Body(position={self.position}, velocity={self.velocity}, acceleration={self.acceleration}, mass={self.mass})"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def add_actuator(self, actuator):
        """Attach an Actuator instance that will be updated each step."""
        self.actuators.append(actuator)

    def add_sensor(self, sensor):
        """Attach a Sensor instance to this body."""
        self.sensors.append(sensor)

    def add_controller(self, controller):
        """Attach a Controller instance to this body."""
        self.controllers.append(controller) 