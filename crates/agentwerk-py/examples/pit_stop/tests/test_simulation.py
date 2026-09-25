"""Each crew action earns its place in the release gate."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, local

import pytest

from simulation import CORNERS, CREW, PitStop


def stop():
    events = []
    pit = PitStop(lambda name, data: events.append((name, data)), sleep=lambda _: None)
    return pit, events


def finish_available(pit, omit=None):
    while True:
        actions = [action for action in pit.eligible() if action != omit]
        if not actions:
            return
        for actor, action in actions:
            pit.perform(actor, action)


def test_arrival_blocks_service_until_car_stops():
    pit, events = stop()
    assert len(pit.eligible()) == 11
    assert all(action in ("collect", "position") for _, action in pit.eligible())
    with pytest.raises(ValueError):
        pit.perform("jack-front", "lift")
    pit.arrive()
    assert [name for name, _ in events] == [
        "pit_initialized",
        "car_arriving",
        "car_stopped",
    ]
    assert len(pit.eligible()) == 15


def test_early_tool_collection_stays_clear_and_cannot_start_mechanical_work():
    from movement import LAYOUT

    pit, events = stop()
    for actor, action in pit.eligible():
        pit.perform(actor, action)
    assert pit.snapshot()["car"] == "approaching"
    assert not pit.eligible()
    for name, data in events:
        if name != "action_started":
            continue
        if data["actor"] == "chief":
            assert data["action"] == "position"
            assert all(
                point[0] >= 5.3 for phase in data["phases"] for point in phase["points"]
            )
            continue
        side = 1 if data["actor"].endswith("right") else -1
        for phase in data["phases"]:
            assert all(
                side * point[1] > LAYOUT["corridor"] + 0.45 for point in phase["points"]
            )
    for member in CREW:
        for action in member.actions:
            if action != "collect":
                with pytest.raises(ValueError):
                    pit.perform(member.id, action)
    pit.arrive()
    assert len(pit.eligible()) == 4


@pytest.mark.parametrize(
    "omitted", [(member.id, action) for member in CREW for action in member.actions]
)
def test_omitting_any_required_action_prevents_departure(omitted):
    pit, _ = stop()
    pit.arrive()
    finish_available(pit, omit=omitted)
    assert pit.snapshot()["car"] == "stopped"
    with pytest.raises(ValueError, match="chief"):
        pit.depart()


def test_every_role_changes_the_car_and_chief_releases_it_once():
    pit, events = stop()
    pit.arrive()
    finish_available(pit)
    assert pit.snapshot()["car"] == "released"
    completed = [data for name, data in events if name == "action_completed"]
    assert len(completed) == 52
    assert {data["actor"] for data in completed} == {member.id for member in CREW}
    for corner in CORNERS:
        stages = [
            data["state"]["wheels"][corner]
            for data in completed
            if data["actor"].endswith(corner)
            and data["action"] in ("loosen", "remove", "fit", "tighten")
        ]
        assert stages == ["loose", "empty", "fitted", "secured"]
    assert all(angle == 12 for angle in pit.snapshot()["wings"].values())
    assert all(worker["clear"] for worker in pit.snapshot()["crew"].values())
    pit.depart()
    assert pit.snapshot()["car"] == "departed"
    with pytest.raises(ValueError):
        pit.depart()


def test_duplicate_or_wrong_role_action_leaves_state_unchanged():
    pit, _ = stop()
    pit.arrive()
    pit.perform("jack-front", "lift")
    before = pit.snapshot()
    for actor, action in (
        ("jack-front", "lift"),
        ("chief", "lift"),
        ("missing", "lift"),
    ):
        with pytest.raises(ValueError):
            pit.perform(actor, action)
        assert pit.snapshot() == before


def test_wheel_teams_can_loosen_concurrently():
    pit, events = stop()
    pit.arrive()
    for actor, action in pit.eligible():
        pit.perform(actor, action)
    barrier = Barrier(4)
    threads = local()

    def sleep(_):
        if not getattr(threads, "started", False):
            threads.started = True
            barrier.wait(timeout=5)

    pit.sleep = sleep
    with ThreadPoolExecutor(4) as pool:
        list(
            pool.map(lambda corner: pit.perform(f"gunner-{corner}", "loosen"), CORNERS)
        )
    names = [
        name
        for name, data in events
        if data.get("action") == "loosen"
        and name in ("action_started", "action_completed")
    ]
    assert names[:4] == ["action_started"] * 4
    assert names[4:] == ["action_completed"] * 4


def test_hold_during_an_action_prevents_its_completion():
    pit, _ = stop()
    pit.arrive()
    pit.sleep = lambda _: pit.hold("Equipment failure")
    with pytest.raises(ValueError, match="held"):
        pit.perform("jack-front", "lift")
    assert pit.snapshot()["jacks"]["front"] == "down"
    assert not pit.eligible()


def test_fitting_and_tightening_do_not_wait_for_cleanup():
    pit, _ = stop()
    for actor, action in pit.eligible():
        pit.perform(actor, action)
    pit.arrive()
    for actor, action in pit.eligible():
        pit.perform(actor, action)
    corner = "front-left"
    pit.perform(f"gunner-{corner}", "loosen")
    pit.perform(f"wheel-off-{corner}", "remove")
    assert pit.ready(f"wheel-on-{corner}", "fit")
    assert pit.ready(f"wheel-off-{corner}", "stow")
    assert (
        pit.snapshot()["items"][f"old-{corner}"]["owner"] == f"crew:wheel-off-{corner}"
    )
    pit.perform(f"wheel-on-{corner}", "fit")
    assert pit.ready(f"gunner-{corner}", "tighten")
    assert pit.ready(f"wheel-on-{corner}", "withdraw")
    assert pit.snapshot()["items"][f"fresh-{corner}"]["owner"] == f"hub:{corner}"


def test_tool_use_requires_actual_item_ownership():
    pit, _ = stop()
    pit.arrive()
    for actor, action in pit.eligible():
        pit.perform(actor, action)
    item = "tool-gunner-front-left"
    state = pit.snapshot()
    state["items"][item]["owner"] = f"slot:{item}"
    pit.emit("pit_initialized", initial=state, seed=pit.seed)
    before = pit.snapshot()
    with pytest.raises(ValueError):
        pit.perform("gunner-front-left", "loosen")
    assert pit.snapshot() == before


def test_chief_steps_aside_before_the_sign_can_release_the_car():
    pit, events = stop()
    assert pit.snapshot()["crew"]["chief"]["position"] == [9, -5.8]
    pit.arrive()
    finish_available(pit, omit=("chief", "release"))
    assert pit.snapshot()["crew"]["chief"]["position"] == [5.3, 0]
    assert pit.ready("chief", "release")

    def check_departure_is_blocked(_):
        assert pit.snapshot()["car"] == "stopped"
        with pytest.raises(ValueError, match="chief"):
            pit.depart()

    pit.sleep = check_departure_is_blocked
    pit.perform("chief", "release")
    release = next(
        data
        for name, data in events
        if name == "action_started" and data["action"] == "release"
    )
    assert release["duration"] < 3
    travel = [phase for phase in release["phases"] if len(phase["points"]) > 1]
    assert len(travel) == 1
    assert travel[0]["kind"] == "withdraw"
    assert travel[0]["points"][0] == [5.3, 0]
    assert travel[0]["points"][-1] == [9, -5.8]
    assert pit.snapshot()["car"] == "released"
    assert pit.snapshot()["crew"]["chief"]["clear"]


def test_event_history_reconstructs_active_routes_and_final_mechanical_state(
    assert_rebuilt,
):
    pit, _ = stop()

    def check_projection(_):
        assert_rebuilt(pit)

    pit.sleep = check_projection
    pit.arrive()
    finish_available(pit)
    pit.depart()
    assert_rebuilt(pit)
    assert not pit.active
