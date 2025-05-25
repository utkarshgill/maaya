import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH for module imports
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
import numpy as np
from firmware.hal import HAL
from sim import Vector3D, World, RungeKuttaIntegrator, GravitationalForce, GroundCollision, IMUSensor, Motor, Renderer
from sim.engine import Quadcopter

class Board(HAL):
    """Firmware-side simulated HAL driving core physics."""
    def __init__(self, dt: float = 0.01, controller=None, config=None):
        super().__init__(config)
        self.dt = dt
        # Build physics world and quad
        self.quad = Quadcopter(position=Vector3D(0, 0, 0.1))
        self.quad.integrator = RungeKuttaIntegrator()
        self.quad.add_sensor(IMUSensor(accel_noise_std=0.0, gyro_noise_std=0.0))
        # Inform renderer which URDF to load for the quad
        self.quad.urdf_filename = "quadrotor.urdf"
        # X-configuration motors positions and spins
        L = self.quad.arm_length
        diag = L / np.sqrt(2)
        motor_positions = [
            Vector3D(diag, diag, 0),
            Vector3D(-diag, diag, 0),
            Vector3D(-diag, -diag, 0),
            Vector3D(diag, -diag, 0),
        ]
        spins = [1, -1, 1, -1]
        for idx, (r, s) in enumerate(zip(motor_positions, spins)):
            self.quad.add_actuator(Motor(idx, r_body=r, spin=s,
                                          thrust_noise_std=0.0, torque_noise_std=0.0))
        # Create world with gravity and ground collision
        self.world = World(
            forces=[GravitationalForce(), GroundCollision(ground_level=0.0, restitution=0.5)],
            dt=self.dt
        )
        self.world.add_body(self.quad)
        # Initial update to populate state
        self.world.update()
        # Set up visualization
        try:
            self.renderer = Renderer(self.world, config='X', gui=True)
        except Exception as e:
            print(f"Renderer init error: {e}")

    def read(self):
        """Return the primary quad Body and the current simulation time."""
        # Return the Body instance and current sim time
        return self.quad, self.world.time

    def write(self, commands):
        """Apply motor commands, handle keyboard input, advance physics, and render."""
        # Update motor thrusts
        self.quad.motor_thrusts = commands
        # Advance simulation
        self.world.update()
        # Capture keyboard events via HIL if available
        if hasattr(self, 'hil') and hasattr(self.hil, 'handle_pybullet'):
            try:
                self.hil.handle_pybullet(self.renderer)
            except Exception as e:
                print(f"HIL handle_pybullet error: {e}")
        # Draw the scene if renderer is available
        if hasattr(self, 'renderer'):
            try:
                self.renderer.draw()
            except Exception as e:
                print(f"Renderer draw error: {e}")
        return commands

if __name__ == '__main__':
    # Run firmware main loop in simulation mode
    os.environ['TARGET'] = 'sim'
    from firmware.main import main
    main() 