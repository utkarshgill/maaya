"""
Interface definitions for sensors, actuators, and controllers.
"""

class Sensor:
    """Abstract base for sensor implementations."""
    def read(self):
        """Read data from the sensor."""
        raise NotImplementedError

class Actuator:
    """Abstract base for actuator implementations."""
    def write(self, command):
        """Send a command to the actuator."""
        raise NotImplementedError

class Controller:
    """Abstract base for control algorithms."""
    def update(self, data, dt: float):
        """Compute control command given sensor data and timestep."""
        raise NotImplementedError 