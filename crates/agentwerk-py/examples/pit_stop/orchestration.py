"""Compose the pit crew with agentwerk; the browser is only an observer."""

import asyncio
from pathlib import Path
from threading import RLock

from agentwerk import Agent, Event, Model, Policy, Schema, Task, Werk, tool

from movement import LAYOUT
from simulation import CREW, PitStop

PROMPTS = Path(__file__).with_name("prompts")
MECHANICAL_EVENTS = ("car_", "action_", "pit_")
LIFECYCLE_EVENTS = {
    "task_started",
    "task_finished",
    "task_failed",
    "request_started",
    "request_finished",
    "run_finished",
    "policy_violated",
}
FAILURE_EVENTS = {"task_failed", "policy_violated"}


def action_tool(pit, member):
    allowed_actions = {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": list(member.actions)}},
        "required": ["action"],
        "additionalProperties": False,
    }

    @tool(
        name="perform",
        description=(PROMPTS / "perform.tool.md").read_text(),
        schema=allowed_actions,
    )
    def perform(action: str):
        return pit.perform(member.id, action)

    return perform


def build_crew(pit):
    werk = Werk().set_policy(
        Policy(
            max_time=300,
            max_turns=160,
            max_input_tokens=200_000,
            max_request_tokens=1024,
            max_request_retries=2,
        )
    )
    for member in CREW:
        role = (PROMPTS / f"{member.role}.md").read_text()
        agent = Agent.from_env().label(member.id).role(role)
        werk.add_agent(agent.tool(action_tool(pit, member)))

    scheduled = set()
    assigned = {}
    scheduling = RLock()

    def schedule():
        # Result hooks can run on different agent threads; claim each action once.
        with scheduling:
            for actor, action in pit.eligible():
                if (actor, action) in scheduled:
                    continue

                scheduled.add((actor, action))
                result_schema = Schema(
                    {
                        "type": "object",
                        "properties": {"action": {"type": "string", "enum": [action]}},
                        "required": ["action"],
                        "additionalProperties": False,
                    }
                )
                instruction = (
                    f"You are {actor}. Perform {action}, then call finish "
                    f"with action={action} after success."
                )
                task = Task(instruction, label=actor, schema=result_schema)
                assigned[werk.add_task(task)] = (actor, action)

    def completed(_, task, result):
        actor = task.get_label()
        state = pit.snapshot()["crew"][actor]
        expected = assigned.get(task.get_id())
        if (
            not isinstance(result, dict)
            or expected != (actor, result.get("action"))
            or result["action"] not in state["done"]
        ):
            pit.hold(f"{actor} finished without completing its mechanical action")
            return
        schedule()

    werk.on_result(completed)
    return werk, schedule


async def run_stop(feed, seed=None):
    werk = None

    def publish(name, data):
        if werk is not None:
            werk.emit_event(Event(name).data(data))

    pit = PitStop(publish, seed=seed)
    werk, schedule = build_crew(pit)
    loop = asyncio.get_running_loop()
    crew_moving = asyncio.Event()
    moving = set()

    def preparation_started(actor):
        moving.add(actor)
        if len(moving) >= 2:
            crew_moving.set()

    def observe(_, event):
        name = event.get_name()
        if name.startswith(MECHANICAL_EVENTS):
            data = event.get_data()
            feed.push(name, data)
            if (
                name == "action_phase"
                and data["action"] == "collect"
                and data["kind"] == "move"
            ):
                loop.call_soon_threadsafe(preparation_started, data["actor"])
            if name == "pit_held":
                loop.call_soon_threadsafe(crew_moving.set)
        elif name in LIFECYCLE_EVENTS:
            feed.push(name, {"actor": event.get_label(), "task": event.get_task_id()})
        if name in FAILURE_EVENTS:
            pit.hold(f"{event.get_label() or 'Run'}: {name.replace('_', ' ')}")

    werk.on_event(observe)
    crew = [
        {"id": member.id, "role": member.role, "station": member.station}
        for member in CREW
    ]
    metadata = {
        "source": "agentwerk",
        "version": 2,
        "seed": pit.seed,
        "motion": pit.movement.choices,
        "layout": LAYOUT,
        "model": Model.from_env().get_name(),
        "crew": crew,
        "state": pit.snapshot(),
    }
    feed.push("run_metadata", metadata)
    try:
        schedule()
        werk.start()
        await asyncio.wait_for(crew_moving.wait(), timeout=180)
        if pit.snapshot()["held"]:
            await werk.finish()
            return False
        await asyncio.to_thread(pit.arrive)
        schedule()
        await werk.finish()
        if pit.snapshot()["car"] == "released":
            await asyncio.to_thread(pit.depart)
        else:
            pit.hold("Crew stopped before the chief authorized release")
    except Exception:
        pit.hold("Run failed; inspect the local terminal for the cause")
        werk.cancel()
        raise
    return pit.snapshot()["car"] == "departed"
