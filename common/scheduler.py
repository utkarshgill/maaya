"""
Multi-rate Scheduler for coordinating tasks at independent frequencies.
"""
import time

def now() -> float:
    """Return the current monotonic time in seconds."""
    return time.monotonic()

def sleep(duration: float):
    """Sleep for the specified duration in seconds."""
    time.sleep(duration)

class Scheduler:
    """
    Simple multi-rate scheduler: register tasks with individual periods.
    """
    def __init__(self, time_fn=None):
        """
        Initialize scheduler. If time_fn is None, use real-time scheduling (time.monotonic);
        otherwise use provided time_fn (e.g., simulation time) for step-by-step control.
        """
        if time_fn is None:
            self.time_fn = time.monotonic
            self.real_time = True
        else:
            self.time_fn = time_fn
            self.real_time = False
        self.tasks = []  # list of {'func', 'period', 'next'}

    def add_task(self, func, period: float):
        """
        Register a callable to be called every 'period' seconds.
        For real-time mode, the first execution is after 'period'.
        For simulation mode, the first execution is at time 0.
        """
        if self.real_time:
            now = self.time_fn()
            next_time = now + period
        else:
            # in simulation mode, schedule first task at time 0
            next_time = 0.0
        self.tasks.append({'func': func, 'period': period, 'next': next_time})

    def run(self):
        """
        Run the scheduler loop indefinitely.
        """
        while True:
            now = self.time_fn()
            for task in self.tasks:
                if now >= task['next']:
                    try:
                        task['func']()
                    except Exception as e:
                        print(f"Scheduler task error: {e}")
                    # schedule next execution
                    task['next'] += task['period']
            # determine sleep time until next task
            if self.real_time:
                next_times = [t['next'] for t in self.tasks]
                next_run = min(next_times)
                sleep_time = next_run - self.time_fn()
                if sleep_time > 0:
                    sleep(sleep_time)

    def step(self):
        """
        Execute any tasks due at the current time (real or simulation). Does not sleep.
        Useful for stepping simulation in a gym-like API.
        """
        now = self.time_fn()
        for task in self.tasks:
            if now >= task['next']:
                try:
                    task['func']()
                except Exception as e:
                    print(f"Scheduler task error: {e}")
                task['next'] += task['period'] 