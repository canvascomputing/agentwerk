"""Authored motion remains repeatable and keeps crossing crews apart."""

import math

from movement import LAYOUT, Movement, position
from simulation import CREW, DURATIONS, PitStop


def test_seed_repeats_routes_and_timing_without_synchronizing_workers():
    a = Movement(42, CREW, DURATIONS)
    b = Movement(42, CREW, DURATIONS)
    assert a.choices == b.choices
    assert a.arrival == b.arrival
    assert a.choices != Movement(43, CREW, DURATIONS).choices
    assert len({choice["pace"] for choice in a.choices.values()}) == 19
    assert 3.5 <= a.arrival["duration"] <= 5
    assert abs(a.arrival["offset"]) <= 0.45
    for member in CREW:
        choice = a.choices[member.id]
        assert 0.8 <= choice["pace"] <= 1.3
        assert abs(LAYOUT["crew"][member.id]["home"][1]) > LAYOUT["corridor"]
        for action, timing in choice["actions"].items():
            assert 0.65 <= timing["work"] / DURATIONS[action] <= 1.55
            assert 0.1 <= timing["reaction"] <= 1.4


def test_crossing_reservation_detects_collision_and_accepts_delayed_route():
    motion = Movement(1, CREW, DURATIONS)
    horizontal = [{"kind": "move", "duration": 2, "points": [[-1, 0], [1, 0]]}]
    vertical = [{"kind": "move", "duration": 2, "points": [[0, -1], [0, 1]]}]
    motion.reservations = [("first", 0, horizontal)]
    assert motion.conflicts(vertical, 0)
    delayed = [{"kind": "wait", "duration": 2, "points": [[0, -1]]}, *vertical]
    assert not motion.conflicts(delayed, 0)
    assert position(delayed, 0) == [0, -1]
    assert position(delayed, 4) == [0, 1]


def test_planned_routes_preserve_endpoints_and_reserve_shared_travel():
    motion = Movement(42, CREW, DURATIONS)
    for member in CREW:
        if member.role not in ("gunner", "wing", "jack", "steadier"):
            continue
        place = LAYOUT["crew"][member.id]
        phases = motion.plan(member, member.actions[0], place["home"], 0)
        assert position(phases, 0) == place["home"]
        assert phases[-1]["points"][-1] == (
            place["home"] if member.actions[0] == "collect" else place["work"]
        )
    for i, (_, _, a) in enumerate(motion.reservations):
        for _, _, b in motion.reservations[i + 1 :]:
            duration = min(sum(p["duration"] for p in a), sum(p["duration"] for p in b))
            for step in range(math.ceil(duration / 0.1)):
                assert (
                    math.dist(position(a, step * 0.1), position(b, step * 0.1)) >= 0.48
                )


def test_tool_possession_and_physical_clearance_are_required():
    pit = PitStop(lambda *_: None, sleep=lambda _: None, seed=42)
    pit.arrive()
    for actor, action in pit.eligible():
        if action != "collect":
            pit.perform(actor, action)
    assert not pit.ready("gunner-front-left", "loosen")
    assert not pit.ready("wing-left", "adjust")
    pit.perform("gunner-front-left", "collect")
    assert (
        pit.snapshot()["crew"]["gunner-front-left"]["equipment"]
        == "tool-gunner-front-left"
    )
    assert pit.ready("gunner-front-left", "loosen")
    while pit.eligible():
        for actor, action in pit.eligible():
            if action == "release":
                worker = pit.state["crew"]["wheel-off-front-left"]
                worker["position"] = [0, 0]
                assert not pit.ready("chief", "release")
                worker["position"] = LAYOUT["crew"]["wheel-off-front-left"]["home"]
                assert pit.ready("chief", "release")
                return
            pit.perform(actor, action)
    raise AssertionError("The crew never became ready for release")


def test_occupied_pickup_slot_holds_the_car_instead_of_overlapping_workers():
    import pytest

    pit = PitStop(lambda *_: None, sleep=lambda _: None, seed=42)
    pit.state["crew"]["wheel-off-front-left"]["position"] = LAYOUT["crew"][
        "gunner-front-left"
    ]["station"]
    with pytest.raises(ValueError, match="obstructed"):
        pit.perform("gunner-front-left", "collect")
    assert pit.snapshot()["held"]
    assert not pit.eligible()
