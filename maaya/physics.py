import numpy as np
from .math import Vector3D, Quaternion

# -- Forces ---------------------------------------------------------------
class GravitationalForce:
    def __init__(self, g=9.8):
        self.g = g

    def apply_to(self, obj):
        gravitational_force = Vector3D(0, 0, -self.g * obj.mass)
        obj.apply_force(gravitational_force)


# -- Integrators ----------------------------------------------------------
class EulerIntegrator:
    """Simple Euler integrator for Body state updates."""
    def __init__(self, angular_damp=0.98, linear_drag=0.995):
        self.angular_damp = angular_damp
        self.linear_drag = linear_drag

    def step(self, obj, dt):
        # Attitude integration: q̇ = ½ q ⊗ Ω
        omega_quat = Quaternion(0, *obj.angular_velocity.v)
        orientation_derivative = obj.orientation * omega_quat
        obj.orientation += orientation_derivative * (0.5 * dt)
        obj.orientation.normalize()

        # Angular damping
        obj.angular_velocity *= self.angular_damp

        # Linear integration
        obj.velocity += obj.acceleration * dt
        # Linear drag applies over the interval; damp velocity before position update
        obj.velocity *= self.linear_drag
        obj.position += obj.velocity * dt
        obj.acceleration = Vector3D()  # Reset acceleration


class RungeKuttaIntegrator:
    """Fourth-order Runge-Kutta integrator for Body state updates."""
    def step(self, obj, dt):
        raise NotImplementedError("Runge-Kutta integrator is not implemented yet.") 