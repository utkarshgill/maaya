"""maaya/scheduler.py
Minimal integer-divisor task scheduler – inspired by George Hotz / tinygrad style.

Usage
-----
>>> sched = Scheduler()
>>> sched.add(lambda: print("fast"), every=1)   # run every step
>>> sched.add(lambda: print("slow"), every=10)  # run every 10 steps
>>> for _ in range(20):
...     sched.step()

The scheduler keeps a single global tick counter (`step_idx`) and executes
any task whose divisor exactly divides the current tick.  That's it – no
threading, no priorities, just predictable determinism.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

class Scheduler:
    """Ultra-lightweight deterministic task scheduler.

    Each task is a `(every_n_steps, callable)` pair.  On every `step()` we
    increment an internal counter and invoke tasks whose divisor matches the
    counter.  This mirrors the pattern often used in embedded flight code:
    fast base-rate loop with slower sub-rate tasks triggered by integer division.
    """

    def __init__(self):
        self._tasks: List[Tuple[int, Callable[[], None]]] = []
        self._step_idx: int = 0

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def add(self, fn: Callable[[], None], *, every: int = 1) -> None:
        """Register *fn* to run every *`every`* scheduler steps.

        Parameters
        ----------
        fn : Callable[[], None]
            Zero-arg function to execute.
        every : int, optional (default = 1)
            Frequency divisor.  `every=1` ⇒ run every tick; `every=10` ⇒ run
            once every 10 ticks.
        """
        if every < 1:
            raise ValueError("`every` must be >= 1")
        self._tasks.append((every, fn))

    def step(self) -> None:
        """Advance the scheduler by one base-rate tick and run due tasks."""
        for every, fn in self._tasks:
            if self._step_idx % every == 0:
                fn()
        self._step_idx += 1

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def current_step(self) -> int:
        """Return the current tick count (0-indexed)."""
        return self._step_idx

    def reset(self) -> None:
        """Reset the scheduler to tick 0 (clears nothing else)."""
        self._step_idx = 0 