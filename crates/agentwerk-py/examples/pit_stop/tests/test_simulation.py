"""Physical tool decisions cannot bypass equipment or service prerequisites."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from simulation import PitStop, setup


@pytest.fixture
def pit():
    pit = PitStop(seed=42, realtime=False)
    pit.clock.start()
    yield pit
    pit.clock.close()


def execute(pit, tool, actor, *args, **kwargs):
    try:
        return getattr(pit, tool)(actor, *args, **kwargs)
    finally:
        pit.clock.idle(actor)


def place(pit, actor, destination):
    state = pit.snapshot()
    state["crew"][actor].update(
        position=pit.layout["destinations"][destination], location=destination
    )
    pit.emit("pit_initialized", initial=state)


def stop(pit):
    pit.milestone("car_stopped")


def equip(pit, actor, item):
    slot = pit.snapshot()["items"][item]["owner"].removeprefix("slot:")
    place(pit, actor, f"storage:{slot}")
    execute(pit, "operate", actor, "pickup", item=item)
    return slot


def test_seed_controls_assignments_inventory_targets_and_arrival():
    assert setup(42) == setup(42)
    a, b = setup(42), setup(43)
    assert a[1] != b[1]
    assert a[2] != b[2]
    assert a[3:] != b[3:]
    assert set(a[1]) == set(b[1])
    for role in ("gunner", "wheel-on", "wheel-off"):
        assert (
            len(
                {
                    target
                    for actor, target in a[1].items()
                    if actor.startswith(role + "-")
                }
            )
            == 4
        )


def test_wrong_tool_can_be_picked_up_but_rejected_use_preserves_state(
    pit, assert_rebuilt
):
    actor = "gunner-1"
    slot = equip(pit, actor, "wing-key-5")
    stop(pit)
    target = pit.assignments[actor]
    place(pit, actor, f"work:gunner:{target}")
    before = pit.snapshot()
    with pytest.raises(ValueError, match="Wrong equipment"):
        pit.operate(actor, "use", work="loosen", target=target)
    assert pit.snapshot() == before
    place(pit, actor, f"storage:{slot}")
    execute(pit, "operate", actor, "drop", item="wing-key-5", target=slot)
    equip(pit, actor, "wheel-gun-1")
    assert pit.snapshot()["crew"][actor]["equipment"] == "wheel-gun-1"
    assert_rebuilt(pit)


def test_operate_requires_physical_proximity_and_never_move_the_worker(pit):
    before = pit.snapshot()
    with pytest.raises(ValueError, match="Move to storage"):
        pit.operate("gunner-1", "pickup", item="wheel-gun-1")
    assert pit.snapshot() == before


def test_competing_pickups_transfer_an_item_to_only_one_worker(pit):
    slot = pit.snapshot()["items"]["wheel-gun-1"]["owner"][5:]
    for actor in ("gunner-1", "gunner-2"):
        place(pit, actor, f"storage:{slot}")

    def pickup(actor):
        try:
            execute(pit, "operate", actor, "pickup", item="wheel-gun-1")
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(pickup, ("gunner-1", "gunner-2")))
    assert sum(results) == 1
    assert (
        sum(w["equipment"] == "wheel-gun-1" for w in pit.snapshot()["crew"].values())
        == 1
    )


def test_storage_rejects_occupied_and_incompatible_slots(pit):
    equip(pit, "gunner-1", "wheel-gun-1")
    state = pit.snapshot()
    occupied = state["items"]["wing-key-5"]["owner"][5:]
    with pytest.raises(ValueError, match="occupied"):
        pit.operate("gunner-1", "drop", item="wheel-gun-1", target=occupied)
    with pytest.raises(ValueError, match="tire platforms"):
        pit.operate("gunner-1", "drop", item="wheel-gun-1", target="tire-1")
    assert pit.snapshot() == state


def test_task_target_and_lift_prerequisites_are_enforced(pit):
    actor = "gunner-1"
    equip(pit, actor, "wheel-gun-1")
    target = pit.assignments[actor]
    place(pit, actor, f"work:gunner:{target}")
    with pytest.raises(ValueError, match="car must be stopped"):
        pit.operate(actor, "use", work="loosen", target=target)
    stop(pit)
    other = next(c for c in pit.snapshot()["wheels"] if c != target)
    with pytest.raises(ValueError, match="outside"):
        pit.operate(actor, "use", work="loosen", target=other)
    before = pit.snapshot()
    with pytest.raises(ValueError, match="Both jacks"):
        pit.operate(actor, "use", work="loosen", target=target)
    assert pit.snapshot() == before


def test_jack_is_owned_engaged_lowered_and_stored_explicitly(pit, assert_rebuilt):
    actor = "jack-1"
    target = pit.assignments[actor]
    item = f"jack-{target}"
    equip(pit, actor, item)
    stop(pit)
    execute(pit, "move", actor, f"work:jack:{target}", "run")
    execute(pit, "operate", actor, "use", work="lift", target=target)
    assert pit.snapshot()["items"][item]["owner"] == f"mount:{target}"
    before = pit.snapshot()
    with pytest.raises(ValueError, match="withdraw"):
        pit.move(actor, f"parking:{actor}", "walk")
    with pytest.raises(ValueError, match="lowered jack"):
        pit.operate(actor, "pickup", item=item)
    with pytest.raises(ValueError, match="Complete wheel"):
        pit.operate(actor, "use", work="lower", target=target)
    assert pit.snapshot() == before
    state = pit.snapshot()
    state["wheels"] = dict.fromkeys(state["wheels"], "secured")
    state["wings"] = state["wing_angles"].copy()
    state["steadiers"] = dict.fromkeys(state["steadiers"], "clear")
    pit.emit("pit_initialized", initial=state)
    execute(pit, "operate", actor, "use", work="lower", target=target)
    assert pit.snapshot()["jacks"][target] == "down"
    assert not pit.clear()
    execute(pit, "operate", actor, "pickup", item=item)
    execute(pit, "move", actor, f"storage:{item}", "walk")
    execute(pit, "operate", actor, "drop", item=item, target=item)
    execute(pit, "move", actor, f"parking:{actor}", "walk")
    assert pit.snapshot()["crew"][actor]["location"] == f"parking:{actor}"
    assert pit.snapshot()["items"][item]["owner"] == f"slot:{item}"
    assert_rebuilt(pit)


def test_clearance_observation_includes_jacks_and_chief(pit):
    state = pit.snapshot()
    state["wheels"] = dict.fromkeys(state["wheels"], "secured")
    state["wings"] = state["wing_angles"].copy()
    state["steadiers"] = dict.fromkeys(state["steadiers"], "clear")
    state["items"]["jack-front"]["owner"] = "mount:front"
    pit.emit("pit_initialized", initial=state)
    assert not pit.clear()
    state["items"]["jack-front"]["owner"] = "slot:jack-front"
    state["crew"]["chief"]["clear"] = False
    pit.emit("pit_initialized", initial=state)
    assert not pit.clear()
    execute(pit, "move", "chief", "chief-clear", "walk")
    assert pit.clear()


def test_holding_positions_leave_first_wave_approaches_open(pit):
    occupied = [
        point
        for name, point in pit.layout["destinations"].items()
        if name.startswith("holding:")
    ]
    for actor, member in pit.crew.items():
        if member.role not in ("chief", "jack", "steadier"):
            continue
        start = pit.layout["destinations"][
            "chief-home" if actor == "chief" else f"parking:{actor}"
        ]
        target = pit.layout["destinations"][
            "pit-board"
            if actor == "chief"
            else f"work:{member.role}:{pit.assignments[actor]}"
        ]
        assert pit.movement.path(start, target, occupied)


def test_cancelling_jack_pickup_releases_reservations_without_transfer():
    from threading import Event

    began = Event()
    pit = PitStop(seed=42, realtime=False)
    actor = "jack-1"
    item = f"jack-{pit.assignments[actor]}"
    place(pit, actor, f"storage:{item}")
    pit.werk.on_event(
        lambda _, event: (
            began.set() if event.get_name() == "crew_task_started" else None
        )
    )
    pit.clock.decide("test barrier")
    pit.clock.start()
    with ThreadPoolExecutor(1) as pool:
        pickup = pool.submit(pit.operate, actor, "pickup", item=item)
        assert began.wait(timeout=2)
        pit.hold("Cancelled during pickup")
        with pytest.raises(ValueError, match="stopped"):
            pickup.result(timeout=2)
    assert not pit.claims
    assert not pit.active
    assert pit.snapshot()["items"][item]["owner"] == f"slot:{item}"


def test_chief_faces_the_driver_after_arrival_and_after_stepping_aside(pit):
    import math

    for destination in ("pit-board", "chief-clear"):
        execute(pit, "move", "chief", destination, "walk")
        worker = pit.snapshot()["crew"]["chief"]
        x, z = worker["position"]
        assert math.sin(worker["heading"]) == pytest.approx(-x / math.hypot(x, z))
        assert math.cos(worker["heading"]) == pytest.approx(-z / math.hypot(x, z))


def test_parking_groups_face_the_car_and_leave_jacks_clear_of_people_and_benches(pit):
    import math
    from itertools import combinations

    homes = [w["position"] for w in pit.snapshot()["crew"].values()]
    assert sum(z < 0 for x, z in homes) == 10
    assert sum(z > 0 for x, z in homes) == 9
    assert all(math.dist(a, b) >= 0.58 for a, b in combinations(homes, 2))
    for worker in pit.snapshot()["crew"].values():
        x, z = worker["position"]
        assert math.sin(worker["heading"]) == pytest.approx(-x / math.hypot(x, z))
    for end in ("front", "rear"):
        x, _, z = pit.layout["slots"][f"jack-{end}"]
        assert all(
            abs(x - cx) > 1.95 or abs(z - cz) > 0.68
            for cx, cz in pit.layout["stations"].values()
        )
        grip = [x, z + 0.72]
        assert all(math.dist(grip, home) > 0.32 for home in homes)
