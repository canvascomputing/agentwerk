"""Compose the pit crew with agentwerk; the browser is only an observer."""

import asyncio
from pathlib import Path
from uuid import uuid4

from agentwerk import Agent, Condition, Event, Model, Policy, Task, Werk, tool

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


async def run_stop(feed, seed=None):
    session = Path(".agentwerk") / f"pit-stop-{uuid4().hex}"
    pit = PitStop(seed=seed, werk=Werk(str(session)))
    werk = build_crew(pit)
    loop = asyncio.get_running_loop()
    crew_moving = asyncio.Event()

    def observe(_, event):
        name = event.get_name()
        if name in LIFECYCLE_EVENTS:
            task_event = {"actor": event.get_label(), "task": event.get_task_id()}
            feed.push(name, task_event)
        if name in FAILURE_EVENTS:
            pit.hold(f"{event.get_label() or 'Run'}: {name.replace('_', ' ')}")
        if not name.startswith(MECHANICAL_EVENTS):
            return

        data = event.get_data()
        feed.push(name, {**data, "state": pit.snapshot()})
        if name == "pit_held":
            loop.call_soon_threadsafe(crew_moving.set)
            return
        if (
            name != "action_phase"
            or data["action"] != "collect"
            or data["kind"] != "move"
        ):
            return

        recorded_phases = werk.find_events("event.name = action_phase")
        phases = [phase.get_data() for phase in recorded_phases]
        moving_collectors = {
            phase["actor"]
            for phase in phases
            if phase["action"] == "collect" and phase["kind"] == "move"
        }
        if len(moving_collectors) >= 2:
            loop.call_soon_threadsafe(crew_moving.set)

    werk.on_event(observe)
    crew = [
        {"id": member.id, "role": member.role, "station": member.station}
        for member in CREW
    ]
    metadata = {
        "source": "agentwerk",
        "version": 3,
        "seed": pit.seed,
        "motion": pit.movement.choices,
        "layout": LAYOUT,
        "model": Model.from_env().get_name(),
        "crew": crew,
        "state": pit.snapshot(),
    }
    feed.push("run_metadata", metadata)
    try:
        werk.start()
        await asyncio.wait_for(crew_moving.wait(), timeout=180)
        if pit.snapshot()["held"]:
            await werk.finish()
            return False
        await asyncio.to_thread(pit.arrive)
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


def build_crew(pit):
    policy = Policy(
        max_time=300,
        max_turns=160,
        max_input_tokens=200_000,
        max_request_tokens=1024,
        max_request_retries=2,
    )
    werk = pit.werk.set_policy(policy)
    for member in CREW:
        role = (PROMPTS / f"{member.role}.md").read_text()
        # Host completion keeps FinishTool out of the model's tool list.
        agent = Agent.from_env().interactive().label(member.id).role(role)
        werk.add_agent(agent.tool(action_tool(pit, member)))

    for member in CREW:
        for action in member.actions:
            assignment = {
                "actor": member.id,
                "action": action,
                "instruction": f'Call perform with action="{action}" exactly once. '
                "The tool completes this task. Do not answer with a completion claim.",
            }
            task = Task(assignment, label=member.id)
            readiness = f"event.name = ready:{member.id}:{action}"
            werk.add_condition(Condition(readiness).task(task))

    def publish_ready_actions():
        for actor, action in pit.eligible():
            werk.emit_event(Event(f"ready:{actor}:{action}"))

    def advance_crew(host, event):
        name = event.get_name()
        if name == "pit_held":
            host.cancel()
            return
        if name in ("task_failed", "policy_violated", "tool_call_failed"):
            pit.hold(f"{event.get_label() or 'Run'}: {name.replace('_', ' ')}")
            return
        if name in ("run_started", "car_stopped"):
            publish_ready_actions()
            return
        if name not in ("tool_call_finished", "task_finished"):
            return
        if (
            name == "tool_call_finished"
            and event.get_data().get("tool_name") != "perform"
        ):
            return

        task = host.get_task(event.get_task_id())
        actor = task.get_label()
        action = task.get_task()["action"]
        completed_actions = pit.snapshot()["crew"][actor]["done"]
        if action not in completed_actions:
            if name == "task_finished":
                pit.hold(f"{actor} finished without completing its mechanical action")
            return
        if name == "task_finished":
            publish_ready_actions()
            return

        result = {"actor": actor, "action": action, "completed": True}
        host.set_task_finished(task.get_id(), result)

    werk.on_event(advance_crew)
    return werk


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
        task = pit.werk.find_task(
            f"task.label = {member.id} AND task.status = in_progress"
        )
        if task is None or task.get_task()["action"] != action:
            pit.hold(f"{member.id} attempted an action outside its assigned task")
            raise ValueError("Perform only the action assigned by your task")
        return pit.perform(member.id, action)

    return perform
