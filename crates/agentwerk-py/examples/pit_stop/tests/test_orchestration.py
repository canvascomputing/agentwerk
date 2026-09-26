"""Exercise multi-turn agents through the real Werk and a local provider."""

import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from feed import Feed
from orchestration import build_crew, run_stop
from simulation import CREW, MILESTONES, PitStop


def choose(task, observation):
    actor, step, target = task["actor"], task["step"], task["target"]
    if step == "review":
        worker = observation["self"] if observation else task["state"]["crew"]["chief"]
        if not worker["clear"] and not task["outstanding"]:
            return "move", {"destination": "stage:chief:chief", "pace": "walk"}
        return "finish", {
            "decision": "hold"
            if task["outstanding"] or any(not r["valid"] for r in task["reports"])
            else "go",
            "reviewed_tasks": task["reviewed_tasks"],
            "reason": "Reviewed all recorded outcomes and physical clearance",
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
        destination = f"{prefix}:{role}:{target}"
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


def scripted_provider(false_report=False, chief_go=False, chief_parks=False):
    calls = []

    async def respond(request):
        body = await request.json()
        assert {t["function"]["name"] for t in body["tools"]} == {
            "move",
            "operate",
            "finish",
        }
        content = next(m["content"] for m in body["messages"] if m["role"] == "user")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content)
        task = json.JSONDecoder().raw_decode(content.lstrip())[0]
        if task["step"] == "review":
            statuses = json.JSONDecoder().raw_decode(
                content.split("Reported statuses:\n", 1)[1]
            )[0]
            task["reports"] = json.JSONDecoder().raw_decode(
                content.split("Verified reports and task IDs:\n", 1)[1]
            )[0]
            task["reviewed_tasks"] = json.JSONDecoder().raw_decode(
                content.split("Task IDs to copy into reviewed_tasks:\n", 1)[1]
            )[0]
            assert task["reviewed_tasks"] == [r["task_id"] for r in task["reports"]]
            assert statuses and all(s in ("completed", "blocked") for s in statuses)
            assert task["reports"] and all(r["task_id"] for r in task["reports"])
            calls.append(("review", task["actor"], task["reports"]))
        observation = task.get("observation")
        latest_result = None
        for message in body["messages"]:
            if message["role"] == "tool":
                result = json.loads(message["content"])
                if "observation" in result:
                    observation = result["observation"]
                    latest_result = result
        if latest_result and not latest_result["ok"]:
            calls.append(("rejected", task["actor"], latest_result["message"]))
        name, args = choose(task, observation)
        if chief_parks and task["step"] == "review":
            worker = (
                observation["self"] if observation else task["state"]["crew"]["chief"]
            )
            if worker["location"] != "parking:chief":
                name, args = "move", {"destination": "parking:chief", "pace": "walk"}
        if false_report and task["actor"] == "gunner-1" and task["step"] == "prepare":
            name, args = "finish", {"status": "completed"}
        if chief_go and task["step"] == "review" and name == "finish":
            args["decision"] = "go"
        calls.append((name, task["actor"], args))
        if len(calls) > 1200:
            raise RuntimeError(str(calls[-15:]))
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


@pytest.mark.parametrize(
    "false_report, seed, chief_go",
    [
        (False, 42, True),
        (False, 43, True),
        (False, 7, True),
        (True, 42, True),
        (True, 42, False),
    ],
)
async def test_reports_require_physical_work_and_chief_checks_all_results(
    monkeypatch, false_report, seed, chief_go, assert_rebuilt
):
    app, calls = scripted_provider(
        false_report, chief_go=chief_go, chief_parks=not false_report
    )
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
            await asyncio.gather(arrival, return_exceptions=True)
            state = pit.snapshot()
            assert state["car"] in (
                ("approaching", "stopped") if false_report else ("released",)
            ), (state["held"], calls[-20:])
            assert bool(state["held"]) == false_report
            assert_rebuilt(pit)
            reviewed = next(c[2] for c in calls if c[0] == "review")
            assert any(not report["valid"] for report in reviewed) == false_report
            assert {r["task_id"] for r in reviewed} <= {
                event.get_data()["task_id"]
                for event in events
                if event.get_name() == "pit_report"
            }
            if not false_report:
                assert state["crew"]["chief"]["location"] == "parking:chief"
                names = [e.get_name() for e in events]
                for name in MILESTONES - {"pit_held", "car_departing", "car_departed"}:
                    assert names.count(name) == 1, name
                assert (
                    names.index("car_approaching")
                    < names.index("car_stopped")
                    < names.index("pit_service_completed")
                    < names.index("pit_released")
                )
                approach = next(
                    e.get_data() for e in events if e.get_name() == "car_approaching"
                )
                stopped = next(
                    e.get_data() for e in events if e.get_name() == "car_stopped"
                )
                assert stopped["seconds"] - approach["seconds"] == pytest.approx(
                    approach["arrives_in_seconds"]
                )
                assert sum(c[0] == "move" for c in calls) > len(CREW)
                assert sum(c[0] == "operate" for c in calls) > len(CREW)
                verified = set()
                last_clear = None
                lowerings = []
                for event in events:
                    name, data = event.get_name(), event.get_data()
                    if name == "pit_report" and data["valid"]:
                        verified.add((data["actor"], data["step"]))
                        if data["step"] == "clear":
                            last_clear = data["seconds"]
                    if name != "crew_task_started":
                        continue
                    if data["task"] == "lower":
                        lowerings.append(data["seconds"])
                        assert all(
                            (m.id, "clear") in verified
                            for m in CREW
                            if m.role == "steadier"
                        )
                    destination = data.get("destination", "")
                    if not destination.startswith("work:"):
                        continue
                    role = pit.crew[data["actor"]].role
                    target = pit.assignments[data["actor"]]
                    if role in ("gunner", "wing", "wheel-off", "wheel-on"):
                        assert all(
                            (m.id, "lift" if m.role == "jack" else "brace") in verified
                            for m in CREW
                            if m.role in ("jack", "steadier")
                        )
                    predecessor = {
                        "wheel-off": ("gunner", "loosen"),
                        "wheel-on": ("wheel-off", "remove"),
                    }.get(role)
                    if predecessor:
                        assert any(
                            m.role == predecessor[0]
                            and pit.assignments[m.id] == target
                            and (m.id, predecessor[1]) in verified
                            for m in CREW
                        )
                assert lowerings and min(lowerings) - last_clear < 0.01
                assert any(c[0] == "move" and c[2]["pace"] == "walk" for c in calls)
                assert any(c[0] == "move" and c[2]["pace"] == "run" for c in calls)
            else:
                assert sum(e.get_name() == "pit_held" for e in events) == 1
                assert not any(e.get_name() == "pit_released" for e in events)
        finally:
            pit.clock.close()
            werk.cancel()


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
