"""
Simulation target using PS5 DualSense controller.
"""

from . import register_target
import numpy as np
try:
    import hid  # provided by `pip install hidapi`
except ImportError:
    hid = None

from maaya import Controller
from ..core import StabilityController

# PS5 DualSense constants
VENDOR_ID = 0x054C
PRODUCT_ID = 0x0CE6
REPORT_ID = 0x01


@register_target("sim_dualsense")
def sim_target():
    """
    Simulation target that uses PS5 DualSense for input.
    Returns a factory for a PS5-driven attitude controller.
    """
    return {"make_controller": make_controller}


class DualSenseController(Controller):
    """
    Reads PS5 DualSense sticks and drives StabilityController's PID loops.
    """
    def __init__(self, stability_ctrl, thrust_gain=5.0, max_tilt_rad=0.5, yaw_rate_gain=np.pi/2):
        self.stability_ctrl = stability_ctrl
        self.thrust_gain = thrust_gain
        self.max_tilt = max_tilt_rad
        self.yaw_rate_gain = yaw_rate_gain
        if hid:
            try:
                self.h = hid.device()
                self.h.open(VENDOR_ID, PRODUCT_ID)
                try:
                    self.h.set_nonblocking(True)
                except AttributeError:
                    pass # non-blocking not critical
            except (OSError, IOError) as e:
                print(f"Warning: could not open PS5 controller HID device: {e}")
                self.h = None
        else:
            self.h = None
        self.key_state = {} # For keyboard fallback

    def update(self, body, dt):
        if not self.h:
            return self._keyboard_control(body, dt)
        
        data = self.h.read(64)
        if not data or data[0] != REPORT_ID:
            return None # No new data

        lx, ly, rx, ry = data[1], data[2], data[3], data[4]

        def deadzone(val, dz=0.1):
            return val if abs(val) > dz else 0.0

        norm_lx = deadzone((lx - 127) / 127.0)
        norm_ly = deadzone((127 - ly) / 127.0) # Inverted
        norm_rx = deadzone((rx - 127) / 127.0)
        norm_ry = deadzone((127 - ry) / 127.0) # Inverted

        # Update Z setpoint from LY stick (throttle)
        z_sp = self.stability_ctrl.z_pid.setpoint + norm_ly * self.thrust_gain * dt
        self.stability_ctrl.z_pid.setpoint = float(np.clip(z_sp, 0.0, 20.0))

        # Update roll/pitch setpoints from RX/RY sticks
        target_roll = np.clip(norm_rx * self.max_tilt, -self.max_tilt, self.max_tilt)
        target_pitch = np.clip(norm_ry * self.max_tilt, -self.max_tilt, self.max_tilt)
        
        # Update yaw setpoint from LX stick (rate control)
        current_yaw_sp = self.stability_ctrl.yaw_pid.setpoint
        target_yaw = current_yaw_sp - norm_lx * self.yaw_rate_gain * dt # Inverted for intuitive control
        target_yaw = (target_yaw + np.pi) % (2 * np.pi) - np.pi # Wrap to [-pi, pi]
        
        self.stability_ctrl.set_attitude_target(target_roll, target_pitch, target_yaw)
        # Compute altitude and attitude outputs manually
        thrust_cmd = float(np.clip(9.8 + self.stability_ctrl.z_pid.update(body.position.v[2], dt), 0.0, 15.0))
        roll_cmd   = float(np.clip(self.stability_ctrl.roll_pid.update(body.orientation.to_euler()[0], dt), -self.max_tilt, self.max_tilt))
        pitch_cmd  = float(np.clip(self.stability_ctrl.pitch_pid.update(body.orientation.to_euler()[1], dt), -self.max_tilt, self.max_tilt))
        yaw_cmd    = float(np.clip(self.stability_ctrl.yaw_pid.update(body.orientation.to_euler()[2], dt), -0.3, 0.3))
        return [thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd]

    def set_xyz_target(self, x, y, z):
        """Pass through to wrapped StabilityController."""
        self.stability_ctrl.set_xyz_target(x, y, z)

    def _keyboard_control(self, body, dt):
        # Altitude (W/S)
        if self.key_state.get('w', False):
            z_sp = self.stability_ctrl.z_pid.setpoint + self.thrust_gain * dt
            self.stability_ctrl.z_pid.setpoint = float(np.clip(z_sp, 0.0, 20.0))
        if self.key_state.get('s', False):
            z_sp = self.stability_ctrl.z_pid.setpoint - self.thrust_gain * dt
            self.stability_ctrl.z_pid.setpoint = float(np.clip(z_sp, 0.0, 20.0))

        # Roll (Left/Right Arrows)
        roll_target = 0.0
        if self.key_state.get('left', False):
            roll_target = -self.max_tilt
        elif self.key_state.get('right', False):
            roll_target = self.max_tilt

        # Pitch (Up/Down Arrows)
        pitch_target = 0.0
        if self.key_state.get('up', False):
            pitch_target = self.max_tilt # Assuming up arrow = pitch forward (positive tilt)
        elif self.key_state.get('down', False):
            pitch_target = -self.max_tilt # Assuming down arrow = pitch backward (negative tilt)

        # Yaw (A/D)
        current_yaw_sp = self.stability_ctrl.yaw_pid.setpoint
        yaw_target = current_yaw_sp
        if self.key_state.get('a', False):
            yaw_target = current_yaw_sp + self.yaw_rate_gain * dt
        if self.key_state.get('d', False):
            yaw_target = current_yaw_sp - self.yaw_rate_gain * dt
        yaw_target = (yaw_target + np.pi) % (2 * np.pi) - np.pi

        self.stability_ctrl.set_attitude_target(roll_target, pitch_target, yaw_target)
        # Compute altitude and attitude outputs manually (keyboard fallback)
        thrust_cmd = float(np.clip(9.8 + self.stability_ctrl.z_pid.update(body.position.v[2], dt), 0.0, 15.0))
        roll_cmd   = float(np.clip(self.stability_ctrl.roll_pid.update(body.orientation.to_euler()[0], dt), -self.max_tilt, self.max_tilt))
        pitch_cmd  = float(np.clip(self.stability_ctrl.pitch_pid.update(body.orientation.to_euler()[1], dt), -self.max_tilt, self.max_tilt))
        yaw_cmd    = float(np.clip(self.stability_ctrl.yaw_pid.update(body.orientation.to_euler()[2], dt), -0.3, 0.3))
        return [thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd]

def make_controller():
    """Instantiate and return the PS5-driven attitude controller."""
    stability_controller = StabilityController()
    return DualSenseController(stability_controller) 