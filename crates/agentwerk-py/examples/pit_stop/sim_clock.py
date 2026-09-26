"""A shared clock that advances physical work only after agents decide."""

import time
from threading import Condition, RLock, Thread


class SimulationClock:
    def __init__(self, publish=lambda *_, **__: None, realtime=True):
        self.lock = RLock()
        self.condition = Condition(self.lock)
        self.seconds = 0.0
        self.deciding = set()
        self.waiters = {}
        self.closed = False
        self.running = False
        self.publish = publish
        self.realtime = realtime
        self.thread = None

    def start(self):
        if self.thread is None:
            self.thread = Thread(target=self.advance, daemon=True)
            self.thread.start()

    def decide(self, actor):
        with self.condition:
            self.deciding.add(actor)
            self.set_running(False)
            self.condition.notify_all()

    def idle(self, actor):
        with self.condition:
            self.deciding.discard(actor)
            self.condition.notify_all()

    def set_running(self, running):
        if self.running != running:
            self.running = running
            self.publish(
                "pit_clock",
                running=running,
                seconds=self.seconds,
                until=min(self.waiters.values())
                if running and self.waiters
                else self.seconds,
            )

    def wait(self, seconds, actor):
        with self.condition:
            if self.closed:
                raise ValueError("The simulation clock is stopped")
            self.waiters[actor] = self.seconds + seconds
            self.deciding.discard(actor)
            self.condition.notify_all()
            self.condition.wait_for(lambda: actor not in self.waiters or self.closed)
            if self.closed:
                raise ValueError("The simulation clock is stopped")

    def advance(self):
        with self.condition:
            while not self.closed:
                if self.deciding or not self.waiters:
                    self.set_running(False)
                    self.condition.wait()
                    continue
                self.set_running(True)
                step = min(0.02, max(0, min(self.waiters.values()) - self.seconds))
                if self.realtime:
                    began = time.monotonic()
                    self.condition.wait(timeout=step)
                    if self.deciding or self.closed:
                        continue
                    step = min(step, time.monotonic() - began)
                self.seconds += step
                due = [
                    actor
                    for actor, end in self.waiters.items()
                    if end <= self.seconds + 1e-9
                ]
                for actor in due:
                    del self.waiters[actor]
                    # Reserve the next decision before waking the tool's thread.
                    self.deciding.add(actor)
                if due:
                    self.set_running(False)
                    self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.set_running(False)
            self.waiters.clear()
            self.deciding.clear()
            self.condition.notify_all()
