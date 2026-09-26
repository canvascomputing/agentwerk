"""Exercise multi-turn agents through the real Werk and a local provider."""

import asyncio
import json
import math

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from feed import Feed
from orchestration import STEPS, TRIGGERS, assignment, build_crew, run_stop
from simulation import CREW, MILESTONES, PitStop


def choose(task, observation):
    actor, step, target = task["actor"], task["step"], task["target"]
    if step == "review":
        if "blocked" in task["reports"].values():
            decision = "hold"
        elif not observation["self"]["clear"]:
            return "move", {"destination": "chief-home", "pace": "walk"}
        else:
            decision = "go"
        return "finish", {
            "decision": decision,
        }
    worker = observation["self"]
    role = worker["role"]

    def move(destination):
        return "move", {
            "destination": destination,
            "pace": "walk" if step in ("prepare", "cleanup") else "run",
        }

    def finish():
        return "finish", {"status": "completed"}

    if step == "prepare":
        if role in ("gunner", "wing", "wheel-on", "jack") and not worker["equipment"]:
            number = int(actor.rsplit("-", 1)[1])
            item = (
                f"wheel-gun-{number}"
                if role == "gunner"
                else f"wing-key-{number + 4}"
                if role == "wing"
                else f"jack-{target}"
                if role == "jack"
                else f"fresh-{target}"
            )
            owner = observation["items"][item]["owner"]
            destination = "storage:" + owner.removeprefix("slot:")
            if worker["location"] != destination:
                return move(destination)
            return "operate", {"task": "pickup", "item": item}
        prefix = (
            "holding"
            if role in ("gunner", "wing", "wheel-on", "wheel-off")
            else "work"
            if role == "chief"
            else "stage"
        )
        destination = "pit-board" if role == "chief" else f"{prefix}:{role}:{target}"
        return finish() if worker["location"] == destination else move(destination)
    if step == "cleanup":
        if (
            role == "jack"
            and observation["items"][f"jack-{target}"]["owner"] == f"mount:{target}"
        ):
            return "operate", {"task": "pickup", "item": f"jack-{target}"}
        if worker["equipment"]:
            item = worker["equipment"]
            kind = observation["items"][item]["kind"]
            occupied = {i["owner"] for i in observation["items"].values()}
            slots = [
                d.removeprefix("storage:")
                for d in task["destinations"]
                if d.startswith("storage:")
                and (
                    d == f"storage:jack-{target}"
                    if kind == "jack"
                    else d.startswith("storage:tire-")
                    if kind in ("old", "fresh")
                    else d.startswith("storage:bench-")
                )
                and "slot:" + d.removeprefix("storage:") not in occupied
                and d not in observation["busy_destinations"]
            ]
            if not slots:
                return move(
                    f"stage:{role}:{target}"
                    if worker["location"] == f"parking:{actor}"
                    else f"parking:{actor}"
                )
            if (
                worker["location"].startswith("storage:")
                and worker["location"].removeprefix("storage:") in slots
            ):
                slot = worker["location"].removeprefix("storage:")
            else:
                preferred = observation["items"][item]["storage"]
                slot = preferred if preferred in slots else slots[0]
            destination = f"storage:{slot}"
            if worker["location"] != destination:
                return move(destination)
            return "operate", {"task": "drop", "item": item, "target": slot}
        return (
            finish()
            if worker["location"] == f"parking:{actor}"
            else move(f"parking:{actor}")
        )
    if step not in worker["done"]:
        destination = f"work:{role}:{target}"
        if worker["location"] != destination:
            return move(destination)
        args = {"task": "use", "work": step, "target": target}
        if step == "adjust":
            args["value"] = observation["wing_angles"][target]
        return "operate", args
    destination = (
        f"work:{role}:{target}"
        if step in ("lift", "brace", "lower")
        else f"stage:{role}:{target}"
    )
    return finish() if worker["location"] == destination else move(destination)


def scripted_provider(false_report=False, chief_go=False, blocked=False):
    calls = []

    async def respond(request):
        body = await request.json()
        tool_names = {t["function"]["name"] for t in body["tools"]}
        content = next(m["content"] for m in body["messages"] if m["role"] == "user")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content)
        task = json.JSONDecoder().raw_decode(content.lstrip())[0]
        task.update(json.JSONDecoder().raw_decode(content.split("Context:\n", 1)[1])[0])
        assert tool_names == {"move", "operate", "finish"}
        if task["step"] == "review":

            def section(title):
                text = content.split(title + ":\n", 1)[1].lstrip()
                return (
                    json.JSONDecoder().raw_decode(text)[0]
                    if text.startswith("[")
                    else []
                )

            reports = section("Crew reports")
            finished = [t for t in task["tasks"] if t["status"] == "finished"]
            task["reports"] = {t["id"]: t["result"]["status"] for t in finished}
            assert len(reports) >= len(finished)
            assert all(status in ("completed", "blocked") for status in reports)
            calls.append(("review", task["actor"], task))
        observation = task.get("observation")
        latest_result, rejections = None, 0
        for message in body["messages"]:
            if message["role"] == "tool":
                try:
                    result = json.loads(message["content"])
                except json.JSONDecodeError:
                    continue
                if isinstance(result, dict) and "observation" in result:
                    observation = result["observation"]
                    latest_result = result
                    rejections += not result["ok"]
        if latest_result and not latest_result["ok"]:
            calls.append(("rejected", task["actor"], latest_result["message"]))
        name, args = choose(task, observation)
        if latest_result and "Wrong equipment" in latest_result.get("message", ""):
            name, args = "finish", {"status": "blocked"}
        if (
            latest_result
            and not latest_result["ok"]
            and name == "move"
            and any(
                word in latest_result["message"].lower()
                for word in ("route", "occupied")
            )
        ):
            alternatives = {
                destination: point
                for destination, point in task["destinations"].items()
                if destination not in observation["busy_destinations"]
                and destination
                not in (observation["self"]["location"], args["destination"])
            }
            # Paused time never clears a busy route, so try each alternative in turn.
            nearest = sorted(
                alternatives,
                key=lambda destination: math.dist(
                    observation["self"]["position"], alternatives[destination]
                ),
            )
            args["destination"] = nearest[rejections % len(nearest)]
            args["pace"] = "walk"
        if (
            (false_report or blocked)
            and task["actor"] == "gunner-1"
            and task["step"] == "prepare"
        ):
            name, args = "finish", {"status": "blocked" if blocked else "completed"}
        if chief_go and task["step"] == "review":
            name, args = (
                "finish",
                {"decision": "go"},
            )
        calls.append((name, task["actor"], args))
        if len(calls) > 1200:
            raise RuntimeError(str([(name, actor) for name, actor, _ in calls[-15:]]))
        chunk = {
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": f"call-{len(calls)}",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(args),
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        return web.Response(
            text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
            content_type="text/event-stream",
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", respond)
    return app, calls


def configure_provider(monkeypatch, server):
    monkeypatch.setenv("LITELLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", str(server.make_url("/")).rstrip("/"))
    monkeypatch.setenv("MODEL", "test")


@pytest.mark.parametrize("seed", [42, 43, 7])
async def test_conditions_and_chief_coordinate_the_full_stop(
    monkeypatch, seed, assert_rebuilt
):
    app, calls = scripted_provider()
    async with TestServer(app) as server:
        configure_provider(monkeypatch, server)
        pit = PitStop(seed=seed, realtime=False)
        werk = build_crew(pit)
        events = []
        werk.on_event(lambda _, event: events.append(event))
        pit.approach()
        arrival = asyncio.create_task(asyncio.to_thread(pit.arrive))
        try:
            werk.start()
            await asyncio.wait_for(werk.finish(), timeout=90)
            await arrival
            assert pit.snapshot()["car"] == "released", calls[-20:]
            assert pit.clear()
            assert_rebuilt(pit)
            names = [e.get_name() for e in events]
            for name in MILESTONES - {"pit_held", "car_departing", "car_departed"}:
                assert names.count(name) == 1, name
            assert (
                names.index("car_approaching")
                < names.index("car_stopped")
                < names.index("pit_service_completed")
                < names.index("pit_released")
            )
            assert not any(
                name.startswith("pit_ready:") or name == "pit_report" for name in names
            )
            assert (
                len(werk.find_tasks("task.label = chief AND task.input ~ review")) == 1
            )
            tasks = werk.find_tasks("task.label != chief")
            assert len(tasks) == sum(
                len(STEPS[m.role]) for m in CREW if m.role != "chief"
            )
            assert len(
                {(assignment(t)["actor"], assignment(t)["step"]) for t in tasks}
            ) == len(tasks)
            # GO cancels the run, so a parked crew member may not have reported yet.
            assert all(
                t.get_result() == {"status": "completed"}
                or (t.get_result() is None and assignment(t)["step"] == "cleanup")
                for t in tasks
            )
            for t in tasks:
                step, target = assignment(t)["step"], assignment(t)["target"]
                if step not in ("remove", "fit", "tighten"):
                    continue
                handoff_at = next(
                    i
                    for i, e in enumerate(events)
                    if e.get_name() == TRIGGERS[step]
                    and e.get_data()["corner"] == target
                )
                started_at = next(
                    i
                    for i, e in enumerate(events)
                    if e.get_name() == "task_started" and e.get_task_id() == t.get_id()
                )
                assert handoff_at < started_at
            assert any(c[0] == "move" and c[2]["pace"] == "walk" for c in calls)
            assert any(c[0] == "move" and c[2]["pace"] == "run" for c in calls)
        finally:
            pit.clock.close()
            werk.cancel()
            await asyncio.gather(arrival, return_exceptions=True)


@pytest.mark.parametrize(
    "false_report,blocked,chief_go",
    [(True, False, False), (False, True, False), (True, False, True)],
)
async def test_chief_owns_hold_and_go_without_a_host_verdict(
    monkeypatch, false_report, blocked, chief_go
):
    app, _ = scripted_provider(
        false_report=false_report, blocked=blocked, chief_go=chief_go
    )
    async with TestServer(app) as server:
        configure_provider(monkeypatch, server)
        pit = PitStop(seed=42, realtime=False)
        werk = build_crew(pit)
        pit.approach()
        arrival = asyncio.create_task(asyncio.to_thread(pit.arrive))
        try:
            werk.start()
            await asyncio.wait_for(werk.finish(), timeout=30)
            state = pit.snapshot()
            assert bool(state["held"]) == (not chief_go)
            if chief_go:
                assert state["car"] == "released"
                assert not pit.clear()
            else:
                assert not werk.find_events("event.name = pit_released")
        finally:
            pit.clock.close()
            werk.cancel()
            await asyncio.gather(arrival, return_exceptions=True)


async def test_run_stop_records_simulation_time_and_model(monkeypatch):
    app, _ = scripted_provider()
    async with TestServer(app) as server:
        configure_provider(monkeypatch, server)
        monkeypatch.setattr(
            "orchestration.PitStop", lambda **kw: PitStop(**kw, realtime=False)
        )
        feed = Feed()
        assert await asyncio.wait_for(run_stop(feed, seed=42), timeout=90)
        assert feed.frames[0]["data"]["model"] == "test"
        assert feed.frames[0]["data"]["version"] == 6
        assert any(
            f["name"] == "pit_clock" and not f["data"]["running"] for f in feed.frames
        )
        assert all(a["t"] <= b["t"] for a, b in zip(feed.frames, feed.frames[1:]))
        assert any(f["name"] == "car_departed" for f in feed.frames)
        lifted = next(f for f in feed.frames if f["name"] == "car_lifted")
        assert "state" in lifted["data"]
        titles = [f["data"]["title"] for f in feed.frames if f["name"] == "pit_title"]
        assert titles[0].startswith("Car arrives in")
        assert "Car lifted" in titles and titles[-1] == "Car departed"


async def test_cancelled_run_holds_the_car_and_releases_the_clock(monkeypatch):
    app, _ = scripted_provider()
    async with TestServer(app) as server:
        configure_provider(monkeypatch, server)
        feed = Feed()
        run = asyncio.create_task(run_stop(feed, seed=42))

        async def wait_for_start():
            while not any(f["name"] == "task_started" for f in feed.frames):
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_for_start(), timeout=5)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
        assert feed.clock.closed
        assert sum(f["name"] == "pit_held" for f in feed.frames) == 1
        assert not any(f["name"] == "pit_released" for f in feed.frames)


def test_conditions_and_results_start_the_right_tasks_once(monkeypatch):
    from agentwerk import Event, Task

    monkeypatch.setenv("LITELLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MODEL", "test")
    pit = PitStop(seed=42, realtime=False)
    werk = build_crew(pit)
    before = pit.snapshot()
    corner = pit.assignments["gunner-1"]
    remover = next(
        m.id for m in CREW if m.role == "wheel-off" and pit.assignments[m.id] == corner
    )
    removal = f"task.label = {remover} AND task.input ~ remove"
    cleanup = "task.label = gunner-1 AND task.input ~ cleanup"

    def loosened(actor):
        data = {"actor": actor, "corner": pit.assignments[actor]}
        werk.emit_event(Event("wheel_loosened").data(data))

    def report(actor, step, status):
        body = json.dumps({"actor": actor, "step": step}, separators=(",", ":"))
        task_id = werk.add_task(Task(body, label=actor))
        werk.set_task_finished(task_id, {"status": status})
        return task_id

    try:
        loosened("gunner-2")
        assert not werk.find_tasks(removal)
        loosened("gunner-1")
        loosened("gunner-1")
        assert len(werk.find_tasks(removal)) == 1
        finished = report("gunner-1", "loosen", "completed")
        werk.emit_event(Event("task_created").task_id(finished))
        assert "gunner-1" not in pit.clock.deciding
        assert not werk.find_tasks(cleanup)
        werk.emit_event(Event("car_lowered"))
        werk.emit_event(Event("car_lowered"))
        assert len(werk.find_tasks(cleanup)) == 1
        assert len(werk.find_tasks("task.input ~ cleanup")) == len(CREW) - 1
        assert not werk.find_tasks("task.label = chief AND task.input ~ review")
        report("gunner-2", "loosen", "blocked")
        assert len(werk.find_tasks("task.label = chief AND task.input ~ review")) == 1
        assert pit.snapshot() == before
        werk.emit_event(Event("car_stopped"))
        werk.emit_event(Event("car_stopped"))
        assert len(werk.find_tasks("task.label = jack-1 AND task.input ~ lift")) == 1
    finally:
        pit.clock.close()
        werk.cancel()
