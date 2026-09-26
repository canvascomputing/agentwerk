"""Routes preserve endpoints, avoid obstacles, and honor chosen pace."""

from itertools import pairwise

import pytest

from movement import Movement, position
from simulation import setup


def test_cross_side_route_rounds_corners_and_avoids_arrival_corridor():
    layout, *_ = setup(42)
    motion = Movement(layout)
    points = motion.path([-2, -5.8], [2, 5.8], approaching=True)
    assert points[0] == [-2, -5.8]
    assert points[-1] == [2, 5.8]
    assert all(x >= 4.5 or abs(z) >= 1.7 for x, z in points)
    assert any(a[0] != b[0] and a[1] != b[1] for a, b in pairwise(points))
    for x, z in points:
        assert all(
            abs(x - cx) >= 1.95 or abs(z - cz) >= 0.68
            for cx, cz in layout["stations"].values()
        )


def test_running_is_faster_with_same_endpoints_and_loaded_travel_is_slower():
    layout, *_ = setup(42)
    motion = Movement(layout)
    args = ([0, -5.8], [2, -2.8])
    walk = motion.plan(*args, "walk", 0, 0)
    run = motion.plan(*args, "run", 0, 0)
    loaded = motion.plan(*args, "run", 0, 0, loaded=True)
    assert run[-1]["duration"] * 2 == pytest.approx(walk[-1]["duration"])
    assert loaded[-1]["duration"] > run[-1]["duration"]
    assert position(run, 0) == args[0]
    assert position(run, 100) == args[1]


def test_reservations_include_final_occupied_position():
    layout, *_ = setup(42)
    motion = Movement(layout)
    a = [{"kind": "move", "duration": 2, "points": [[-1, 0], [1, 0]]}]
    b = [{"kind": "move", "duration": 2, "points": [[0, -1], [0, 1]]}]
    assert motion.conflicts(a, 0, reservations=[("b", 0, b)])
    delayed = [{"kind": "yield", "duration": 2, "points": [[-1, 0]]}, *a]
    assert not motion.conflicts(delayed, 0, reservations=[("b", 0, b)])


def test_python_matches_browser_distance_profile():
    import json
    from pathlib import Path

    fixture = json.loads(
        Path(__file__).with_name("fixtures").joinpath("walking.json").read_text()
    )
    for sample in fixture["samples"]:
        assert position(fixture["phases"], sample["time"]) == sample["position"]


def test_arrival_turn_uses_shortest_rotation_and_retains_its_heading():
    import math

    from movement import end_heading, trajectory, turn

    phase = turn([0, -3], math.radians(179), math.radians(-179))
    assert abs(phase["headings"][1] - phase["headings"][0]) == pytest.approx(
        math.radians(2)
    )
    sample = trajectory([phase], with_heading=True)
    assert sample(phase["duration"] / 2)[1] == pytest.approx(math.pi)
    assert sample(100)[1] == end_heading(phase)


def test_jack_sweeps_are_reserved_even_when_the_operator_does_not_move():
    import math

    from movement import turn

    layout, *_ = setup(42)
    motion = Movement(layout)
    turning = turn([0, -3], 0, math.pi)
    turning["reach"] = 1.15
    assert motion.conflicts([turning], 0, occupied=[[1.15, -3]])
    assert not motion.conflicts([turning], 0, occupied=[[-1.15, -3]])


def test_jacks_back_out_before_turning_away_from_the_car():
    import math

    from movement import end_heading, reservation

    layout, *_ = setup(42)
    motion = Movement(layout)
    for sign, end in [(1, "front"), (-1, "rear")]:
        start = layout["destinations"][f"work:jack:{end}"]
        destination = layout["destinations"][f"storage:jack-{end}"]
        heading = -sign * math.pi / 2
        phases = motion.plan(
            start,
            destination,
            "walk",
            heading,
            0,
            loaded=True,
            reach=1.15,
            facing=math.pi,
        )
        assert phases[0]["points"][-1][0] == pytest.approx(start[0] + sign * 0.75)
        assert end_heading(phases[0]) == heading
        sample = reservation(phases)
        duration = sum(p["duration"] for p in phases)
        for i in range(int(duration / 0.02) + 1):
            for k, point in enumerate(sample(i * 0.02)):
                assert motion.clear_point(point, equipment=k > 0)


def test_empty_handed_arrival_reserves_the_stored_jack_without_swinging_it():
    import math

    from movement import reservation, turn

    layout, *_ = setup(42)
    motion = Movement(layout)
    phase = turn([-3.2, -3], math.pi / 2, math.pi)
    phase["end_reach"] = 1.15
    assert not motion.conflicts([phase], 0, occupied=[[-2.2, -3.2]])
    assert motion.conflicts([phase], 0, occupied=[[-3.2, -4.15]])
    assert reservation([phase])(0) == reservation([phase])(100)
