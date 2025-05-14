"""
Simple publish/subscribe message bus for inter-module communication.
"""
from collections import defaultdict


class Bus:
    """A minimal topic-based message bus."""

    def __init__(self):
        # Map topic -> list of subscriber callables
        self._subscribers = defaultdict(list)

    def subscribe(self, topic: str, fn):
        """Subscribe a callable to a topic."""
        self._subscribers[topic].append(fn)

    def publish(self, topic: str, data):
        """Publish data to a topic, invoking all subscriber callbacks."""
        for fn in self._subscribers.get(topic, []):
            try:
                fn(data)
            except Exception:
                # swallow subscriber exceptions to isolate
                pass 