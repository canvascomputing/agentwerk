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
    assert frames[0]["data"]["version"] == 6
    names = [f["name"] for f in frames]
    order = [
        "car_approaching",
        "car_stopped",
        "car_lifted",
        "pit_service_completed",
        "car_unbraced",
        "car_lowered",
        "pit_crew_clear",
        "pit_released",
        "car_departed",
    ]
    positions = [names.index(name) for name in order]
    assert positions == sorted(positions)
    reviews = [
        f["data"]["result"]
        for f in frames
        if f["name"] == "task_finished" and f["data"].get("step") == "review"
    ]
    assert reviews == [{"decision": "go"}]
    titles = [f["data"]["title"] for f in frames if f["name"] == "pit_title"]
    assert titles[-1] == "Car departed"
    stopped = frames[names.index("car_stopped")]
    assert any(
        f["name"] == "crew_task_started" and f["t"] < stopped["t"] for f in frames
    )
    assert names.count("car_departing") == 1
    assert names.count("request_finished") > names.count("task_finished")


def test_recorded_transfers_conserve_every_tool_and_tire():
    frames = read_recording(Path(__file__).parents[1] / "recordings/showcase.jsonl")
    owners = {
        name: item["owner"]
        for name, item in frames[0]["data"]["state"]["items"].items()
    }
    assert set(owners) == set(frames[0]["data"]["state"]["items"])
    for frame in frames[1:]:
        data = frame["data"]
        if frame["name"] == "crew_transfer":
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
            assert owner.startswith("slot:tire-")
        elif name.startswith("jack-"):
            assert owner == f"slot:{name}"
        else:
            assert owner.startswith("slot:bench-")


def test_recorded_paths_keep_moving_and_stationary_workers_apart():
    import math
    from itertools import combinations

    from movement import position

    frames = read_recording(Path(__file__).parents[1] / "recordings/showcase.jsonl")
    state = frames[0]["data"]["state"]
    tasks, phases = {}, {}
    index = 0
    for step in range(math.ceil(frames[-1]["t"] / 0.1)):
        now = step * 0.1
        while index < len(frames) and frames[index]["t"] <= now:
            frame = frames[index]
            index += 1
            data = frame["data"]
            state = data.get("state", state)
            if frame["name"] == "crew_task_started":
                tasks[data["actor"]] = frame
            if frame["name"] == "crew_task_phase":
                phases[data["actor"]] = frame
            if frame["name"] == "crew_task_completed":
                tasks.pop(data["actor"], None)
                phases.pop(data["actor"], None)
        points = {}
        for actor, worker in state["crew"].items():
            points[actor] = worker["position"]
            if actor in tasks and actor in phases:
                phase = phases[actor]
                route = tasks[actor]["data"]["phases"][phase["data"]["phase"]]
                points[actor] = position([route], now - phase["t"])
        for a, b in combinations(points, 2):
            assert math.dist(points[a], points[b]) >= 0.46, (now, a, b)
