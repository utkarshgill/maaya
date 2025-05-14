# dynamics.py
# Rigid-body dynamics: Body class, integrators, and force models

import numpy as np
from .math import Vector3D, Quaternion

# Integrators
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
        obj.velocity *= self.linear_drag
        obj.position += obj.velocity * dt
        obj.acceleration = Vector3D()  # Reset acceleration

class RungeKuttaIntegrator:
    """Fourth-order Runge-Kutta integrator for Body state updates."""
    def __init__(self, angular_damp: float = 0.98, linear_drag: float = 0.995):
        self.angular_damp = angular_damp
        self.linear_drag = linear_drag

    def step(self, obj, dt):
        # Attitude (simple Euler)
        omega_quat = Quaternion(0, *obj.angular_velocity.v)
        q_dot = obj.orientation * omega_quat
        obj.orientation += q_dot * (0.5 * dt)
        obj.orientation.normalize()
        obj.angular_velocity *= self.angular_damp

        # Translational via RK4
        pos0 = obj.position.v.copy()
        vel0 = obj.velocity.v.copy()
        acc  = obj.acceleration.v.copy()

        def f_pos(v): return v
        def f_vel(): return acc

        k1_pos = f_pos(vel0)
        k1_vel = f_vel()
        k2_pos = f_pos(vel0 + 0.5 * k1_vel * dt)
        k2_vel = k1_vel
        k3_pos = f_pos(vel0 + 0.5 * k2_vel * dt)
        k3_vel = k1_vel
        k4_pos = f_pos(vel0 + k3_vel * dt)
        k4_vel = k1_vel

        new_pos = pos0 + (dt/6.0) * (k1_pos + 2*k2_pos + 2*k3_pos + k4_pos)
        new_vel = vel0 + (dt/6.0) * (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel)
        new_vel *= self.linear_drag

        obj.position = Vector3D(*new_pos)
        obj.velocity = Vector3D(*new_vel)
        obj.acceleration = Vector3D()  # Reset acceleration

# Rigid-body
class Body:
    """Rigid body with 6-DOF state."""
    def __init__(
        self,
        position=None,
        velocity=None,
        acceleration=None,
        mass=1.0,
        orientation=None,
        angular_velocity=None,
        inertia=None,
        integrator=None
    ):
        self.position = position if position is not None else Vector3D()
        self.velocity = velocity if velocity is not None else Vector3D()
        self.acceleration = acceleration if acceleration is not None else Vector3D()
        self.mass = mass
        self.orientation = orientation if orientation is not None else Quaternion()
        self.angular_velocity = angular_velocity if angular_velocity is not None else Vector3D()

        self.inertia = inertia if inertia is not None else np.eye(3)
        self.inertia_inv = np.linalg.inv(self.inertia)

        self.integrator = integrator if integrator is not None else EulerIntegrator()

        self.actuators = []
        self.sensors = []
        self.controllers = []

    def apply_force(self, force):
        self.acceleration += Vector3D(*(force.v / self.mass))

    def apply_torque(self, torque, dt=0.01):
        omega = self.angular_velocity.v
        coriolis = np.cross(omega, self.inertia.dot(omega))
        angular_accel = self.inertia_inv.dot(torque.v - coriolis)
        self.angular_velocity += Vector3D(*angular_accel) * dt

    def update(self, dt):
        self.integrator.step(self, dt)

    def add_actuator(self, actuator):
        self.actuators.append(actuator)

    def add_sensor(self, sensor):
        self.sensors.append(sensor)

    def add_controller(self, controller):
        self.controllers.append(controller)

    def __repr__(self):
        return f"Body(position={self.position}, velocity={self.velocity}, acceleration={self.acceleration}, mass={self.mass})"

# Force models
class GravitationalForce:
    """Uniform gravity."""
    def __init__(self, g=9.8):
        self.g = g

    def apply_to(self, body):
        body.apply_force(Vector3D(0, 0, -self.g * body.mass))

class GroundCollision:
    """Simple ground collision with restitution."""
    def __init__(self, ground_level=0.0, restitution=0.5):
        self.ground_level = ground_level
        self.restitution = restitution

    def apply_to(self, body):
        z = body.position.v[2]
        if z < self.ground_level:
            body.position.v[2] = self.ground_level
            vz = body.velocity.v[2]
            if vz < 0:
                body.velocity.v[2] = -vz * self.restitution 