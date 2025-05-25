"""
Hardware Abstraction Layer (HAL)
Provides a unified interface for sensors, actuators, and controllers.
"""

class HAL:
    """
    Hardware Abstraction Layer (HAL) base class.
    Provides abstract read/write interfaces to communicate with a controller.
    """
    def __init__(self, config=None):
        """Initialize HAL with optional configuration."""
        self.config = config

    def read(self):
        """Read raw sensor data (e.g., observations). Must be implemented by subclass."""
        raise NotImplementedError("HAL.read() must be implemented by subclass")

    def write(self, commands):
        """Write actuator commands. Must be implemented by subclass."""
        raise NotImplementedError("HAL.write() must be implemented by subclass") 