# maaya/sensor.py
import numpy as np

class Sensor:
    """Base class for sensors."""
    def read(self, objects, dt):
        """Read sensors for each object; to be implemented by subclasses."""
        raise NotImplementedError

class IMUSensor(Sensor):
    """IMU sensor model: returns noisy acceleration and angular velocity."""
    def __init__(self, accel_noise_std=0.0, gyro_noise_std=0.0,
                 accel_bias_rw_std=0.0, gyro_bias_rw_std=0.0):
        """IMU white noise + bias random-walk model.

        Parameters
        ----------
        accel_noise_std : float, optional
            1σ white-noise std-dev for accelerometer (m/s²).
        gyro_noise_std : float, optional
            1σ white-noise std-dev for gyroscope (rad/s).
        accel_bias_rw_std : float, optional
            Bias random-walk std-dev (m/s² √Hz).  At each timestep we add
            ``randn()*accel_bias_rw_std*sqrt(dt)`` to the bias.
        gyro_bias_rw_std : float, optional
            Bias random-walk std-dev (rad/s √Hz).
        """

        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std

        self.accel_bias_rw_std = accel_bias_rw_std
        self.gyro_bias_rw_std = gyro_bias_rw_std

        # Initialise zero biases; they evolve over time via random-walk.
        self._accel_bias = np.zeros(3)
        self._gyro_bias = np.zeros(3)

    def read(self, objects, dt):
        """Attach noisy sensor data to each object."""
        for obj in objects:
            true_accel = obj.acceleration.v
            true_gyro = obj.angular_velocity.v

            # Bias random-walk (Gauss-Markov with σ√dt increment)
            if self.accel_bias_rw_std > 0.0:
                self._accel_bias += np.random.randn(3) * self.accel_bias_rw_std * np.sqrt(dt)
            if self.gyro_bias_rw_std > 0.0:
                self._gyro_bias += np.random.randn(3) * self.gyro_bias_rw_std * np.sqrt(dt)

            noisy_accel = true_accel + self._accel_bias + np.random.randn(3) * self.accel_noise_std
            noisy_gyro = true_gyro + self._gyro_bias + np.random.randn(3) * self.gyro_noise_std

            obj.sensor_data = {
                'accel': noisy_accel,
                'gyro': noisy_gyro,
                'accel_bias': self._accel_bias.copy(),
                'gyro_bias': self._gyro_bias.copy(),
            } 