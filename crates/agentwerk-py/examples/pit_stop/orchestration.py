"""Assign outcomes through Werk and let agents choose their physical tasks."""

import asyncio
import json
from pathlib import Path
from uuid import uuid4

from agentwerk import Agent, Condition, Event, Model, Policy, Schema, Task, Werk, tool

from simulation import CREW, PitStop

PROMPTS = Path(__file__).with_name("prompts")
LIFECYCLE_EVENTS = {
    "task_started",
    "task_finished",
    "task_failed",
    "request_started",
    "request_finished",
    "request_failed",
    "run_finished",
    "policy_violated",
}
REPORT = Schema(
    {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["completed", "blocked"]},
        },
        "required": ["status"],
        "additionalProperties": False,
    }
)
VERDICT = Schema(
    {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["go", "hold"]},
            "reviewed_tasks": {
                "type": "array",
                "items": {"type": "string"},
            },
            "reason": {"type": "string", "maxLength": 240},
        },
        "required": ["decision", "reviewed_tasks", "reason"],
        "additionalProperties": False,
    }
)
STEPS = {
    "gunner": ("prepare", "loosen", "tighten", "cleanup"),
    "wheel-off": ("prepare", "remove", "cleanup"),
    "wheel-on": ("prepare", "fit", "cleanup"),
    "wing": ("prepare", "adjust", "cleanup"),
    "jack": ("prepare", "lift", "lower", "cleanup"),
    "steadier": ("prepare", "brace", "clear", "cleanup"),
    "chief": ("prepare",),
}


def finish_destination(pit, actor, step):
    role, target = pit.crew[actor].role, pit.assignments[actor]
    if step == "cleanup":
        return f"parking:{actor}"
    if step in ("lift", "brace", "lower") or role == "chief":
        return f"work:{role}:{target}"
    prefix = (
        "holding"
        if step == "prepare" and role in ("gunner", "wheel-off", "wheel-on", "wing")
        else "stage"
    )
    return f"{prefix}:{role}:{target}"


def objective(pit, actor, step):
    role, target = pit.crew[actor].role, pit.assignments[actor]
    destination = finish_destination(pit, actor, step)
    if step == "prepare":
        kit = {
            "gunner": "a wheel gun",
            "wing": "a wing key",
            "wheel-on": f"the fresh tire for {target}",
            "jack": f"the {target} jack",
        }.get(role, "empty hands")
        board = (
            " Hold the STOP board there throughout service."
            if role == "chief"
            else " Service comes in a separate task."
        )
        return f"Prepare at {destination} with {kit}.{board}"
    if step == "cleanup":
        withdraw = "Pick up your lowered jack first. " if role == "jack" else ""
        return f"{withdraw}Store your equipment and return to {destination} with empty hands."
    work = {
        "loosen": f"Loosen the wheel fasteners at {target}",
        "remove": f"Remove the old tire at {target}",
        "fit": f"Fit the fresh tire at {target}",
        "tighten": f"Tighten the wheel fasteners at {target}",
        "lift": f"Raise the {target} end of the car with its jack",
        "lower": f"Lower the {target} end of the car with its mounted jack",
        "brace": f"Brace the car on its {target} side",
        "clear": f"Let go of the car on its {target} side",
    }
    instruction = (
        f"Set the {target} front-wing flap to {pit.snapshot()['wing_angles'][target]} degrees"
        if step == "adjust"
        else work[step]
    )
    return f"{instruction}. Finish at {destination}. Leave equipment return and parking for cleanup."


def timing(pit, actor, step):
    role, target = pit.crew[actor].role, pit.assignments[actor]
    worker = pit.snapshot()["crew"][actor]
    destination = (
        finish_destination(pit, actor, step)
        if step in ("prepare", "cleanup")
        else f"work:{role}:{target}"
    )
    # Estimates omit temporary traffic; the move tool reserves the actual route.
    phases = pit.movement.plan(
        worker["position"],
        pit.layout["destinations"][destination],
        "walk",
        worker["heading"],
        0,
        [],
        [],
        pit.snapshot()["car"] == "approaching",
        bool(worker["equipment"]),
    )
    seconds = sum(p["duration"] for p in phases)
    waiting = {
        "loosen": "wheel-off worker",
        "remove": "wheel-on worker",
        "fit": "gunner",
        "tighten": "steadiers",
        "adjust": "steadiers",
        "brace": "wheel and wing crew",
        "lift": "wheel and wing crew",
        "clear": "jack operators",
        "lower": "jack cleanup",
    }
    return {
        "waiting_on_you": waiting.get(
            step, "nobody yet" if step == "prepare" else "Chief's release review"
        ),
        "prerequisite": "Earlier required work is complete."
        if step != "prepare"
        else "The car is approaching.",
        "travel": {
            "destination": destination,
            "walk_seconds": round(seconds, 1),
            "run_seconds": round(
                sum(p["duration"] / (2 if p["kind"] == "move" else 1) for p in phases),
                1,
            ),
        },
    }


def report_valid(pit, actor, step, result):
    state = pit.snapshot()
    worker = state["crew"][actor]
    if (
        result.get("status") != "completed"
        or worker["task"]
        or not pit.near(actor, finish_destination(pit, actor, step))
    ):
        return False
    role, target = worker["role"], worker["station"]
    if step == "prepare":
        item = state["items"].get(worker["equipment"], {})
        if role in ("gunner", "wing"):
            return item.get("kind") == role
        if role == "wheel-on":
            return item.get("kind") == "fresh" and item.get("corner") == target
        if role == "jack":
            return item.get("kind") == "jack" and item.get("end") == target
        return worker["equipment"] is None
    if step in ("cleanup", "clear", "lower") and worker["equipment"]:
        return False
    if step == "cleanup" and role == "jack":
        return state["items"][f"jack-{target}"]["owner"] == f"slot:jack-{target}"
    return step == "cleanup" or step in worker["done"]


def crew_tools(pit, member):
    def invoke(fn, **kwargs):
        try:
            with pit.lock:
                current = pit.werk.find_task(
                    f"task.label = {member.id} AND task.status = in_progress"
                )
                if current is None:
                    raise ValueError("There is no assigned task")
                assignment = json.JSONDecoder().raw_decode(current.get_task())[0]
                if (
                    kwargs.get("task") == "use"
                    and kwargs.get("work") != assignment["step"]
                ):
                    raise ValueError(
                        f"Work {kwargs.get('work')!r} is not assigned. {assignment['objective']}"
                    )
            return {"ok": True, "observation": fn(member.id, **kwargs)}
        except ValueError as error:
            return {
                "ok": False,
                "message": str(error),
                "observation": pit.observation(member.id),
            }

    @tool(
        name="move",
        description=(PROMPTS / "move.tool.md").read_text(),
        timeout=0,
        schema={
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "description": "An exact destination name from the task map",
                },
                "pace": {"type": "string", "enum": ["walk", "run"]},
            },
            "required": ["destination", "pace"],
            "additionalProperties": False,
        },
    )
    def move(destination: str, pace: str):
        return invoke(pit.move, destination=destination, pace=pace)

    @tool(
        name="operate",
        description=(PROMPTS / "operate.tool.md").read_text(),
        timeout=0,
        schema={
            "type": "object",
            "properties": {
                "task": {"type": "string", "enum": ["pickup", "drop", "use"]},
                "item": {"type": "string"},
                "target": {"type": "string"},
                "work": {
                    "type": "string",
                    "enum": [
                        "loosen",
                        "remove",
                        "fit",
                        "tighten",
                        "lift",
                        "lower",
                        "brace",
                        "clear",
                        "adjust",
                    ],
                },
                "value": {"type": "number"},
            },
            "required": ["task"],
            "additionalProperties": False,
        },
    )
    def operate(
        task: str,
        item: str | None = None,
        target: str | None = None,
        work: str | None = None,
        value: float | None = None,
    ):
        return invoke(
            pit.operate, task=task, item=item, target=target, work=work, value=value
        )

    return move, operate


def build_crew(pit):
    werk = pit.werk.set_policy(
        Policy(
            max_time=900,
            max_turns=1000,
            max_input_tokens=1_000_000,
            max_request_tokens=1024,
            max_request_retries=2,
        )
    )
    offered, completed, reports = set(), set(), []
    reviewing = False
    for member in CREW:
        role = (PROMPTS / f"{member.role}.md").read_text()
        agent = Agent.from_env().label(member.id).role(role)
        for crew_tool in crew_tools(pit, member):
            agent = agent.tool(crew_tool)
        werk.add_agent(agent)

    def add_task(actor, step, body, schema, prompt=""):
        offered.add((actor, step))
        pit.clock.decide(actor)
        payload = {
            "actor": actor,
            "step": step,
            "target": pit.assignments[actor],
            **body,
        }
        task = Task(
            json.dumps(payload, separators=(",", ":")) + "\n\n" + prompt,
            label=actor,
            schema=schema,
        )
        name = f"pit_ready:{actor}:{step}"
        ready = Condition(f"event.name = {name}").task(task)
        werk.add_condition(ready)
        werk.emit_event(Event(name))

    def review():
        nonlocal reviewing
        if reviewing:
            return
        # The chief's preparation must finish before it can claim the verdict task.
        if (
            ("chief", "prepare") not in completed
            and ("chief", "prepare") in offered
            and not any(r["actor"] == "chief" for r in reports)
        ):
            return
        reviewing = True
        missing = [
            f"{m.id}:{s}"
            for m in CREW
            for s in STEPS[m.role]
            if (m.id, s) not in completed
        ]
        add_task(
            "chief",
            "review",
            {
                "objective": "Decide whether the car can leave. Move to parking:chief or stage:chief:chief before choosing go. Choose hold if any release check fails.",
                "outstanding": missing,
                "state": pit.snapshot(),
                "destinations": pit.layout["destinations"],
            },
            VERDICT,
            """Review the crew's work and decide GO or HOLD.

Reported statuses:
{{ find_results(task.status = finished)[*].status }}

Verified reports and task IDs:
{{ find_events(event.name = pit_report)[*].data }}

Task IDs to copy into reviewed_tasks:
{{ find_events(event.name = pit_report)[*].data.task_id }}

Choose HOLD for blocked reports, invalid reports, or unfinished work.
Check physical clearance before GO.""",
        )

    def ready(actor, step):
        role, target = pit.crew[actor].role, pit.assignments[actor]
        steps = STEPS[role]
        index = steps.index(step)
        if index and (actor, steps[index - 1]) not in completed:
            return False
        if step == "prepare":
            return True
        if pit.snapshot()["car"] != "stopped":
            return False

        def done(role, step, target=None):
            members = [
                m
                for m in CREW
                if m.role == role
                and (target is None or pit.assignments[m.id] == target)
            ]
            return all((m.id, step) in completed for m in members)

        if step in ("loosen", "remove", "fit", "tighten"):
            if not done("jack", "lift") or not done("steadier", "brace"):
                return False
            predecessor = {
                "remove": ("gunner", "loosen"),
                "fit": ("wheel-off", "remove"),
                "tighten": ("wheel-on", "fit"),
            }.get(step)
            return not predecessor or done(*predecessor, target)
        if step == "adjust":
            return done("jack", "lift") and done("steadier", "brace")
        if step == "clear":
            return done("gunner", "tighten") and done("wing", "adjust")
        if step == "lower":
            return done("steadier", "clear")
        return True

    def schedule():
        if pit.snapshot()["held"] or reviewing:
            return
        if any(not r["valid"] for r in reports):
            review()
            return
        for member in CREW:
            for step in STEPS[member.role]:
                key = member.id, step
                if key not in offered and ready(*key):
                    add_task(
                        member.id,
                        step,
                        {
                            "objective": objective(pit, *key),
                            "timing": timing(pit, *key),
                            "observation": pit.observation(member.id),
                            "destinations": {
                                name: point
                                for name, point in pit.layout["destinations"].items()
                                if name.startswith("storage:")
                                or name
                                in (
                                    f"parking:{member.id}",
                                    f"work:{member.role}:{pit.assignments[member.id]}",
                                    f"stage:{member.role}:{pit.assignments[member.id]}",
                                    f"holding:{member.role}:{pit.assignments[member.id]}",
                                )
                            },
                            "reports": [
                                r
                                for r in reports
                                if r["target"] == pit.assignments[member.id]
                            ],
                        },
                        REPORT,
                    )
        if all((m.id, "prepare") in completed for m in CREW):
            pit.milestone("pit_prepared")
        if all((m.id, s) in completed for m in CREW for s in STEPS[m.role]):
            review()

    def accept_result(host, task, result):
        with pit.lock:
            actor, step = (
                task.get_label(),
                json.JSONDecoder().raw_decode(task.get_task())[0]["step"],
            )
            pit.clock.idle(actor)
            if step == "review":
                ids = {r["task_id"] for r in reports}
                all_done = all(
                    (m.id, s) in completed for m in CREW for s in STEPS[m.role]
                )
                if (
                    result["decision"] == "go"
                    and set(result["reviewed_tasks"]) == ids
                    and all_done
                    and pit.clear()
                ):
                    pit.release(result["reviewed_tasks"])
                else:
                    pit.hold(
                        result["reason"]
                        if result["decision"] == "hold"
                        else "Chief GO rejected: incomplete reports or physical clearance"
                    )
                return
            valid = report_valid(pit, actor, step, result)
            report = {
                "task_id": task.get_id(),
                "actor": actor,
                "step": step,
                "target": pit.assignments[actor],
                "valid": valid,
                "result": result,
                "observed": {
                    key: pit.snapshot()["crew"][actor][key]
                    for key in ("location", "equipment", "done")
                },
            }
            reports.append(report)
            pit.emit("pit_report", **report)
            if valid:
                completed.add((actor, step))
            schedule()

    def observe(host, event):
        name = event.get_name()
        with pit.lock:
            if name in ("car_approaching", "car_stopped"):
                schedule()
            elif name in ("task_failed", "policy_violated"):
                pit.hold(f"{event.get_label() or 'Run'}: {name.replace('_', ' ')}")
            elif name == "pit_held":
                host.cancel()

    werk.on_result(accept_result)
    werk.on_event(observe)
    return werk


async def run_stop(feed, seed=None):
    pit = PitStop(
        seed=seed, werk=Werk(str(Path(".agentwerk") / f"pit-stop-{uuid4().hex}"))
    )
    werk = build_crew(pit)
    feed.clock = pit.clock

    def observe(_, event):
        name = event.get_name()
        if name.startswith(("pit_", "car_", "crew_")) and not name.startswith(
            "pit_ready:"
        ):
            feed.push(name, {**event.get_data(), "state": pit.snapshot()})
        elif name in LIFECYCLE_EVENTS:
            feed.push(name, {"actor": event.get_label(), "task": event.get_task_id()})

    werk.on_event(observe)
    feed.push(
        "run_metadata",
        {
            "source": "agentwerk",
            "version": 6,
            "seed": pit.seed,
            "model": Model.from_env().get_name(),
            "layout": pit.layout,
            "crew": [
                {"id": m.id, "role": m.role, "station": pit.assignments[m.id]}
                for m in CREW
            ],
            "motion": {m.id: {"pace": 1} for m in CREW},
            "state": pit.snapshot(),
        },
    )
    arrival = None
    try:
        pit.approach()
        arrival = asyncio.create_task(asyncio.to_thread(pit.arrive))
        werk.start()
        await asyncio.wait_for(werk.finish(), timeout=900)
        await arrival
        if pit.snapshot()["car"] == "released":
            await asyncio.to_thread(pit.depart)
        elif not pit.snapshot()["held"]:
            pit.hold("Crew stopped before the Chief authorized release")
    except asyncio.CancelledError:
        pit.hold("Run cancelled")
        raise
    except Exception:
        if not pit.snapshot()["held"]:
            pit.hold("Run failed; inspect the local terminal for the cause")
            raise
    finally:
        pit.clock.close()
        werk.cancel()
        if arrival:
            await asyncio.gather(arrival, return_exceptions=True)
    return pit.snapshot()["car"] == "departed"
