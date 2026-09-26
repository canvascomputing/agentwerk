"""Model latency pauses shared work without serializing concurrent durations."""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from sim_clock import SimulationClock


def test_decisions_pause_all_work_and_concurrent_waits_share_time():
    clock = SimulationClock(realtime=False)
    clock.decide("first")
    clock.decide("second")
    clock.decide("thinking")
    clock.start()

    def work(actor, seconds):
        clock.wait(seconds, actor)
        result = clock.seconds
        clock.idle(actor)
        return result

    try:
        with ThreadPoolExecutor(2) as pool:
            a = pool.submit(work, "first", 2)
            b = pool.submit(work, "second", 3)
            time.sleep(0.03)
            assert clock.seconds == 0
            clock.idle("thinking")
            assert a.result(timeout=2) == pytest.approx(2)
            assert b.result(timeout=2) == pytest.approx(3)
    finally:
        clock.close()


def test_cancellation_unblocks_a_tool_waiting_on_another_agents_decision():
    clock = SimulationClock(realtime=False)
    clock.decide("thinking")
    clock.start()
    with ThreadPoolExecutor(1) as pool:
        result = pool.submit(clock.wait, 5, "working")
        clock.close()
        with pytest.raises(ValueError, match="stopped"):
            result.result(timeout=2)
    assert not clock.waiters
    assert not clock.deciding
