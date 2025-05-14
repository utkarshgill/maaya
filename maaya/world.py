"""Simulation world orchestrating sense → control → actuate → integrate."""
from .scheduler import Scheduler


class World:
    """Simulation world orchestrating *sense → control → actuate → integrate* using a
    tiny integer-divisor scheduler.  The base-rate loop runs at ``dt`` seconds
    and slower tasks are configured with integer divisors (à la George Hotz).
    """

    # Default sub-rate divisors (1 ⇒ run every base-rate step)
    DEFAULT_DIVISORS = {
        'sense': 1,
        'control': 1,
        'actuate': 1,
        'integrate': 1,
    }

    def __init__(self, gravity=None, dt: float = 0.01, **divisors):
        """Create a simulation *World*.

        Parameters
        ----------
        gravity : object, optional
            A force model implementing ``apply_to(body)``; e.g.,
            :class:`maaya.physics.GravitationalForce`.
        dt : float, optional
            Base-rate timestep in seconds.  All task divisors are relative to
            this value.  Defaults to ``0.01``.
        **divisors : int, optional
            Override task rates with keyword args ``sense``, ``control``,
            ``actuate``, ``integrate``.  Each must be an **integer ≥ 1**.
        """

        self.bodies = []
        self.time: float = 0.0
        self.dt: float = dt
        self.gravity = gravity

        # Merge user-supplied divisors with defaults
        self.divisors = {**World.DEFAULT_DIVISORS, **divisors}

        # Internal task scheduler
        self.scheduler = Scheduler()
        self._register_tasks()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_body(self, body):
        """Add a :class:`maaya.body.Body` instance to be simulated."""
        self.bodies.append(body)

    def update(self, dt: float | None = None):
        """Advance the simulation by one *base-rate* timestep.

        If *dt* is provided and differs from the stored ``self.dt``, the new
        value will be used **from this step onward**.  Task divisors remain
        unchanged – i.e., their absolute rate will shift with the new base-rate.
        """
        if dt is not None:
            self.dt = dt
        # Run all tasks scheduled for this tick
        self.scheduler.step()

    # Alias for backward compatibility
    step = update

    def simulate(self, frames: int):
        """Run *frames* base-rate steps."""
        for _ in range(frames):
            self.update()

    # ------------------------------------------------------------------
    # Task registration helpers
    # ------------------------------------------------------------------
    def _register_tasks(self):
        # Capture dt via closure so lambdas stay zero-arg for Scheduler
        self.scheduler.add(lambda: self._sense(self.dt), every=self.divisors['sense'])
        self.scheduler.add(lambda: self._control(self.dt), every=self.divisors['control'])
        self.scheduler.add(lambda: self._actuate(self.dt), every=self.divisors['actuate'])
        self.scheduler.add(lambda: self._integrate(self.dt), every=self.divisors['integrate'])

    # ------------------------------------------------------------------
    # Task implementations
    # ------------------------------------------------------------------
    def _sense(self, dt: float):
        for body in self.bodies:
            for sensor in getattr(body, 'sensors', []):
                sensor.read([body], dt)

    def _control(self, dt: float):
        for body in self.bodies:
            for controller in getattr(body, 'controllers', []):
                controller.update([body], dt)

    def _actuate(self, dt: float):
        for body in self.bodies:
            # Environment forces (e.g., gravity) first
            if self.gravity is not None:
                self.gravity.apply_to(body)
            # On-board actuators
            for actuator in getattr(body, 'actuators', []):
                actuator.apply_to(body, dt)

    def _integrate(self, dt: float):
        for body in self.bodies:
            body.update(dt)
        self.time += dt

