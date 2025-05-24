from .hal import HAL
from common.math import Vector3D, Quaternion
from sim import Body
from .control import StabilityController
from common.scheduler import Scheduler
from .hil import Keyboard, DualSense

class Board(HAL):
    """Firmware-side abstraction that runs control algorithms at its own rate.

    Board.update(obs) → actions
    obs is expected to be a flat numpy array matching the spec produced by
    sim.engine.World.get_state() (time, pos3, vel3, quat4, angvel3).
    """
    def __init__(self, dt: float = 0.01, controller: StabilityController = None):
        """Initialize Board as a HAL with its own scheduler for control."""
        super().__init__(config=None)
        # Simulation time for scheduling, and placeholders
        self.dt = dt
        self._sim_time = 0.0
        self._latest_obs = None
        self._body = None
        self._latest_action = None
        # Controller
        self.controller = controller if controller is not None else StabilityController()
        # Human-in-the-loop interface: keyboard or DualSense
        self.keyboard = Keyboard()
        self.dualsense = DualSense()
        # Choose HIL based on DualSense availability
        self.hil = self.dualsense if getattr(self.dualsense, 'h', None) else self.keyboard
        # Scheduler using simulation time
        self._sched = Scheduler(time_fn=lambda: self._sim_time)
        # Register read and control tasks at dt
        self._sched.add_task(self._read_task, period=self.dt)
        self._sched.add_task(self._control_task, period=self.dt)
        # Pick/drop state
        self._pick_handled = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, obs):
        """Feed the latest observation, run scheduled read→control, and return action."""
        # Update stored observation and simulation time
        self._latest_obs = obs
        # obs[0] is simulation time
        try:
            self._sim_time = float(obs[0])
        except Exception:
            self._sim_time += self.dt
        # Step scheduler for any due tasks
        self._sched.step()
        # If no action computed yet (at t=0), run control once
        if self._latest_action is None and self._body is not None:
            self._latest_action = self.controller.update(self._body, self.dt)
        return self._latest_action

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _read_task(self):
        """Scheduler task: parse latest obs into Body for control."""
        if self._latest_obs is None:
            return
        obs = self._latest_obs
        # Expected layout: [t, px,py,pz, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz]
        px, py, pz = obs[1:4]
        vx, vy, vz = obs[4:7]
        qw, qx, qy, qz = obs[7:11]
        wx, wy, wz = obs[11:14]
        self._body = Body(
            position=Vector3D(px, py, pz),
            velocity=Vector3D(vx, vy, vz),
            orientation=Quaternion(qw, qx, qy, qz),
            angular_velocity=Vector3D(wx, wy, wz),
            mass=1.0
        )

    def _control_task(self):
        """Scheduler task: update setpoints via HIL, compute control, detect pick/drop."""
        if self._body is None:
            return
        # 1) Update stability controller setpoints based on HIL input
        if self.hil:
            self.hil.update(self.controller, self.dt)
        # 2) Compute base motor commands
        base_cmds = self.controller.update(self._body, self.dt)
        # 3) Detect pick/drop button press
        pick_flag = 0.0
        # Keyboard X or space
        kb_down = False
        if isinstance(self.hil, Keyboard):
            kb_down = bool(self.hil.key_state.get('x')) or bool(self.hil.key_state.get(' '))
        # DualSense cross button
        if hasattr(self.hil, 'cross_pressed'):
            kb_down = kb_down or bool(self.hil.cross_pressed)
        if kb_down and not self._pick_handled:
            pick_flag = 1.0
            self._pick_handled = True
        elif not kb_down:
            self._pick_handled = False
        # 4) Store action vector = [thrust, roll, pitch, yaw, pick_flag]
        self._latest_action = list(base_cmds) + [pick_flag]

    def write(self, commands):
        """Convert controller outputs to motor action vector."""
        # In simplest case, commands == action vector
        return commands 