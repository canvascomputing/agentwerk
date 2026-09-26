"""Assign outcomes through Werk and let agents choose their physical tasks."""

import asyncio
import json
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

from agentwerk import (
    Agent,
    Condition,
    Model,
    Policy,
    Schema,
    Task,
    Werk,
    tool,
)

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
        },
        "required": ["decision"],
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
TRIGGERS = {
    "prepare": "car_approaching",
    "lift": "car_stopped",
    "brace": "car_stopped",
    "loosen": "car_lifted",
    "adjust": "car_lifted",
    "remove": "wheel_loosened",
    "fit": "wheel_removed",
    "tighten": "wheel_fitted",
    "clear": "pit_service_completed",
    "lower": "car_unbraced",
    "cleanup": "car_lowered",
}


def finish_destination(pit, actor, step):
    role, target = pit.crew[actor].role, pit.assignments[actor]
    if role == "chief":
        return "pit-board"
    if step == "cleanup":
        return f"parking:{actor}"
    if step in ("lift", "brace", "lower"):
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
    travel = {"destination": destination}
    try:
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
        travel.update(
            walk_seconds=round(sum(p["duration"] for p in phases), 1),
            run_seconds=round(
                sum(p["duration"] / (2 if p["kind"] == "move" else 1) for p in phases),
                1,
            ),
        )
    except ValueError:
        # Work positions lie in the arriving car's path until it stops.
        pass
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
        "prerequisite": "Check the current observation before working."
        if step != "prepare"
        else "The car is approaching.",
        "travel": travel,
    }


def crew_tools(pit, member):
    def invoke(fn, **kwargs):
        try:
            with pit.lock:
                current = pit.werk.find_task(
                    f"task.label = {member.id} AND task.status = in_progress"
                )
                if current is None:
                    raise ValueError("There is no assigned task")
                assigned = assignment(current)
                if (
                    kwargs.get("task") == "use"
                    and kwargs.get("work") != assigned["step"]
                ):
                    raise ValueError(
                        f"Work {kwargs.get('work')!r} is not assigned. {objective(pit, member.id, assigned['step'])}"
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
    def move_tool(destination: str, pace: str):
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
    def operate_tool(
        task: str,
        item: str | None = None,
        target: str | None = None,
        work: str | None = None,
        value: float | None = None,
    ):
        return invoke(
            pit.operate, task=task, item=item, target=target, work=work, value=value
        )

    return move_tool, operate_tool


def assignment(task):
    return json.JSONDecoder().raw_decode(task.get_task())[0]


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
    for member in CREW:
        role = (PROMPTS / f"{member.role}.md").read_text()
        agent = Agent.from_env().label(member.id).role(role)
        agent.tools(crew_tools(pit, member))
        werk.add_agent(agent)

    def task_for(actor, step):
        payload = {"actor": actor, "step": step, "target": pit.assignments[actor]}
        task = json.dumps(payload, separators=(",", ":"))
        task += "\n\nContext:\n{{ context_" + actor + " }}"
        if step == "review":
            task += "\n\n" + (PROMPTS / "review.task.md").read_text()
        return Task(task, label=actor, schema=VERDICT if step == "review" else REPORT)

    crew_labels = ", ".join(m.id for m in CREW if m.role != "chief")
    werk.set_template("crew_labels", crew_labels)
    triggers = defaultdict(list)
    for member in CREW:
        actor, role = member.id, member.role
        for step in STEPS[role]:
            if step in ("remove", "fit", "tighten"):
                corner = pit.assignments[actor]
                query = f"event.name = {TRIGGERS[step]} AND event.data ~ {corner}"
            else:
                query = f"event.name = {TRIGGERS[step]}"
            triggers[query].append(task_for(actor, step))
    triggers["event.name = pit_crew_clear"].append(task_for("chief", "review"))
    for query, tasks in triggers.items():
        werk.add_condition(Condition(query).tasks(tasks))

    def apply_result(host, task, result):
        actor = task.get_label()
        if actor == "chief" and "decision" in result:
            if result["decision"] == "go":
                pit.release()
                host.cancel()
            else:
                pit.hold("The Chief held the car")
            return
        if result.get("status") == "blocked":
            host.add_task(task_for("chief", "review"))

    werk.on_result(apply_result)

    def observe(host, event):
        name, actor = event.get_name(), event.get_label()
        with pit.lock:
            if name == "task_created":
                # Creation can arrive after completion; queued reviews must not
                # interrupt an operation already waiting on the clock.
                created = host.get_task(event.get_task_id())
                if created.get_status() == "todo" and actor not in pit.clock.waiters:
                    pit.clock.decide(actor)
            elif name == "task_started":
                pit.clock.decide(actor)
                current = assignment(host.get_task(event.get_task_id()))
                step = current["step"]
                context = {
                    "observation": pit.observation(actor),
                    "destinations": {
                        key: point
                        for key, point in pit.layout["destinations"].items()
                        if key.startswith("storage:")
                        or key == f"parking:{actor}"
                        or (
                            actor == "chief"
                            and key in ("pit-board", "chief-home", "chief-clear")
                        )
                        or key.endswith(
                            f":{pit.crew[actor].role}:{pit.assignments[actor]}"
                        )
                    },
                }
                if step == "review":
                    context["observation"] = {
                        "self": context["observation"]["self"],
                        "seconds": pit.clock.seconds,
                    }
                    context["destinations"] = {
                        key: pit.layout["destinations"][key]
                        for key in ("pit-board", "chief-home", "chief-clear")
                    }
                    context.update(
                        state=pit.snapshot(),
                        assignments=pit.assignments,
                        tasks=[
                            {
                                "id": t.get_id(),
                                **assignment(t),
                                "status": t.get_status(),
                                "result": t.get_result(),
                            }
                            for t in host.find_tasks(
                                f"task.label IN ({crew_labels}) ORDER BY task.id"
                            )
                        ],
                    )
                else:
                    context.update(
                        objective=objective(pit, actor, step),
                        timing=timing(pit, actor, step),
                    )
                host.set_template(
                    f"context_{actor}", json.dumps(context, separators=(",", ":"))
                )
            elif name == "task_finished":
                pit.clock.idle(actor)
            elif name in ("task_failed", "policy_violated"):
                pit.clock.idle(actor)
                pit.hold(f"{actor or 'Run'}: {name.replace('_', ' ')}")
            elif name == "pit_held":
                host.cancel()

    werk.on_event(observe)
    return werk


async def run_stop(feed, seed=None):
    pit = PitStop(
        seed=seed, werk=Werk(str(Path(".agentwerk") / f"pit-stop-{uuid4().hex}"))
    )
    werk = build_crew(pit)
    feed.clock = pit.clock

    def show_title(_, event):
        data = event.get_data()
        match event.get_name():
            case "car_approaching":
                title = f"Car arrives in {data['arrives_in_seconds']:.0f} s"
            case "car_arriving":
                title = "Car arriving"
            case "car_stopped":
                title = "Car stopped"
            case "pit_service_started":
                title = "Service started"
            case "car_lifted":
                title = "Car lifted"
            case "pit_service_completed":
                title = "Service complete"
            case "car_unbraced":
                title = "Lowering the car"
            case "car_lowered":
                title = "Car lowered"
            case "pit_crew_clear":
                title = "Crew clear"
            case "pit_released":
                title = "GO"
            case "pit_held":
                title = f"HOLD: {data['message']}"
            case "car_departing":
                title = "Car leaving"
            case "car_departed":
                title = "Car departed"
            case _:
                return
        feed.set_title(title)

    def observe(_, event):
        name = event.get_name()
        if name.startswith(("pit_", "car_", "crew_")):
            data = event.get_data()
            data = data if isinstance(data, dict) else {"data": data}
            feed.push(name, {**data, "state": pit.snapshot()})
        elif name in LIFECYCLE_EVENTS:
            data = {"actor": event.get_label(), "task": event.get_task_id()}
            if name == "task_finished":
                task = werk.get_task(event.get_task_id())
                data.update(step=assignment(task)["step"], result=task.get_result())
            feed.push(name, data)

    werk.on_event(show_title)
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
