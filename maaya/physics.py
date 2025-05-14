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
    def __init__(self, angular_damp: float = 0.98, linear_drag: float = 0.995):
        self.angular_damp = angular_damp
        self.linear_drag = linear_drag

    # NOTE: For now we treat external forces (→ acceleration) as *constant* over
    # the interval ``dt``.  This is true for gravity‐only motion and is a good
    # approximation for small timesteps when actuator forces vary slowly.  A
    # future improvement could re-evaluate forces at the sub-steps (k2/k3/k4).
    def step(self, obj, dt):
        # 1. -------- Orientation (keep simple Euler for now) -----------------
        omega_quat = Quaternion(0, *obj.angular_velocity.v)
        q_dot = obj.orientation * omega_quat  # q̇ = q ⊗ Ω
        obj.orientation += q_dot * (0.5 * dt)
        obj.orientation.normalize()

        # Angular damping
        obj.angular_velocity *= self.angular_damp

        # 2. -------- Translational state via RK4 -----------------------------
        # State vector y = [position, velocity]
        pos0 = obj.position.v.copy()
        vel0 = obj.velocity.v.copy()
        acc  = obj.acceleration.v.copy()       # treated constant over dt

        # Helper lambdas for derivatives
        def f_pos(v):
            return v  # derivative of position is velocity

        def f_vel():
            return acc  # derivative of velocity is acceleration (const)

        # k1
        k1_pos = f_pos(vel0)
        k1_vel = f_vel()

        # k2
        k2_pos = f_pos(vel0 + 0.5 * k1_vel * dt)
        k2_vel = k1_vel  # same since acc constant

        # k3
        k3_pos = f_pos(vel0 + 0.5 * k2_vel * dt)
        k3_vel = k1_vel

        # k4
        k4_pos = f_pos(vel0 + k3_vel * dt)
        k4_vel = k1_vel

        # Combine
        new_pos = pos0 + (dt / 6.0) * (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos)
        new_vel = vel0 + (dt / 6.0) * (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel)

        # Apply linear drag to velocity before committing
        new_vel *= self.linear_drag

        # Commit back to object
        obj.position = Vector3D(*new_pos)
        obj.velocity = Vector3D(*new_vel)

        # Clear accumulated forces for next step
        obj.acceleration = Vector3D()  # Reset acceleration 