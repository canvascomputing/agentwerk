import json
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from feed import Feed, application, read_recording


def test_recording_round_trip_and_reconnection_cursor(tmp_path):
    path = tmp_path / "stop.jsonl"
    feed = Feed(path)
    feed.push("run_metadata", {"source": "agentwerk"})
    feed.push("car_stopped", {"state": {"car": "stopped"}})
    assert read_recording(path) == feed.frames
    assert feed.after(0) == feed.frames[1:]
    assert feed.after(1) == []


def test_recording_rejects_out_of_order_events(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(frame)
            for frame in [
                {"n": 0, "t": 1, "name": "run_metadata"},
                {"n": 1, "t": 0, "name": "car_stopped"},
            ]
        )
    )
    with pytest.raises(ValueError, match="ordered"):
        read_recording(path)


async def test_browser_reconnect_receives_only_unseen_events(tmp_path):
    (tmp_path / "index.html").write_text("Pit Stop")
    feed = Feed()
    feed.push("run_metadata", {})
    feed.push("car_stopped", {})
    async with (
        TestClient(TestServer(application(feed, None, tmp_path))) as client,
        client.get("/events", headers={"Last-Event-ID": "0"}) as response,
    ):
        assert await response.content.readline() == b"id: 1\n"
        frame = json.loads((await response.content.readline()).decode()[6:])
        assert frame["name"] == "car_stopped"


def test_showcase_is_a_complete_real_agent_run():
    frames = read_recording(Path(__file__).parents[1] / "recordings/showcase.jsonl")
    assert frames[0]["data"]["source"] == "agentwerk"
    assert frames[-1]["name"] == "car_departed"
    assert sum(frame["name"] == "action_completed" for frame in frames) == 52
    assert sum(frame["name"] == "request_finished" for frame in frames) >= 27
    stopped = next(frame["t"] for frame in frames if frame["name"] == "car_stopped")
    actions = [frame for frame in frames if frame["name"] == "action_started"]
    early = [frame for frame in actions if frame["t"] < stopped]
    assert early
    assert all(frame["data"]["action"] in ("collect", "position") for frame in early)
    arriving = next(frame["t"] for frame in frames if frame["name"] == "car_arriving")
    moving_before_arrival = {
        frame["data"]["actor"]
        for frame in frames
        if frame["name"] == "action_phase"
        and frame["t"] < arriving
        and frame["data"]["action"] == "collect"
        and frame["data"]["kind"] == "move"
    }
    assert len(moving_before_arrival) >= 2
    assert sum(frame["name"] == "run_finished" for frame in frames) == 1
    assert sum(frame["name"] == "car_departing" for frame in frames) == 1


def test_recorded_transfers_conserve_every_tool_and_tire():
    frames = read_recording(Path(__file__).parents[1] / "recordings/showcase.jsonl")
    owners = {
        name: item["owner"]
        for name, item in frames[0]["data"]["state"]["items"].items()
    }
    assert len(owners) == 14
    for frame in frames[1:]:
        data = frame["data"]
        if frame["name"] == "action_transfer":
            assert owners[data["item"]] == data["from"]
            owners[data["item"]] = data["to"]
        if "state" in data:
            assert owners == {
                name: item["owner"] for name, item in data["state"]["items"].items()
            }
    for name, owner in owners.items():
        if name.startswith("fresh-"):
            assert owner == f"hub:{name.removeprefix('fresh-')}"
        elif name.startswith("old-"):
            assert owner == f"slot:used-{name.removeprefix('old-')}"
        else:
            assert owner == f"slot:{name}"


def test_recording_shows_cleanup_overlapping_other_corners_service():
    frames = read_recording(Path(__file__).parents[1] / "recordings/showcase.jsonl")
    active = {}
    overlap = False
    for frame in frames:
        data = frame["data"]
        if frame["name"] == "action_started":
            active[data["actor"]] = data["action"]
        if frame["name"] == "action_completed":
            active.pop(data["actor"], None)
        overlap |= "stow" in active.values() and any(
            action in ("fit", "tighten", "remove") for action in active.values()
        )
    assert overlap


def test_recorded_paths_keep_moving_and_stationary_workers_apart():
    import math
    from itertools import combinations

    from movement import position

    frames = read_recording(Path(__file__).parents[1] / "recordings/showcase.jsonl")
    state = frames[0]["data"]["state"]
    actions, phases = {}, {}
    index = 0
    for step in range(math.ceil(frames[-1]["t"] / 0.1)):
        now = step * 0.1
        while index < len(frames) and frames[index]["t"] <= now:
            frame = frames[index]
            index += 1
            data = frame["data"]
            state = data.get("state", state)
            if frame["name"] == "action_started":
                actions[data["actor"]] = frame
            if frame["name"] == "action_phase":
                phases[data["actor"]] = frame
            if frame["name"] == "action_completed":
                actions.pop(data["actor"], None)
                phases.pop(data["actor"], None)
        points = {}
        for actor, worker in state["crew"].items():
            points[actor] = worker["position"]
            if actor in actions and actor in phases:
                phase = phases[actor]
                route = actions[actor]["data"]["phases"][phase["data"]["phase"]]
                points[actor] = position([route], now - phase["t"])
        for a, b in combinations(points, 2):
            assert math.dist(points[a], points[b]) >= 0.46, (now, a, b)
