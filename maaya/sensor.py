# maaya/sensor.py
import numpy as np

class Sensor:
    """Base class for sensors."""
    def read(self, objects, dt):
        """Read sensors for each object; to be implemented by subclasses."""
        raise NotImplementedError

class IMUSensor(Sensor):
    """IMU sensor model: returns noisy acceleration and angular velocity."""
    def __init__(self, accel_noise_std=0.0, gyro_noise_std=0.0):
        """
        Args:
            accel_noise_std: standard deviation of accelerometer noise (m/s^2)
            gyro_noise_std: standard deviation of gyroscope noise (rad/s)
        """
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std

    def read(self, objects, dt):
        """Attach noisy sensor data to each object."""
        for obj in objects:
            true_accel = obj.acceleration.v
            true_gyro = obj.angular_velocity.v
            noisy_accel = true_accel + np.random.randn(3) * self.accel_noise_std
            noisy_gyro = true_gyro + np.random.randn(3) * self.gyro_noise_std
            obj.sensor_data = {'accel': noisy_accel, 'gyro': noisy_gyro} 