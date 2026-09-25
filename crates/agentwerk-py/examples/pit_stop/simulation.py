"""Mechanical state and permitted actions for one pit stop."""

import math
import secrets
import time
from copy import deepcopy
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from threading import RLock

from agentwerk import Event, Werk

from movement import LAYOUT, Movement, end_heading

CORNERS = ("rear-left", "front-left", "rear-right", "front-right")
DURATIONS = {
    "collect": 0.65,
    "lift": 1.6,
    "brace": 1.2,
    "loosen": 1.6,
    "remove": 2.2,
    "fit": 2.4,
    "tighten": 1.8,
    "adjust": 3.2,
    "clear": 1.2,
    "lower": 1.6,
    "release": 0.18,
    "position": 0.35,
    "return": 1.2,
    "stow": 1.4,
    "withdraw": 0.5,
}
WHEEL_PREREQUISITES = {
    "loosen": "old-secured",
    "remove": "loose",
    "fit": "empty",
    "tighten": "fitted",
}
WHEEL_RESULTS = {
    "loosen": "loose",
    "remove": "empty",
    "fit": "fitted",
    "tighten": "secured",
}


@dataclass(frozen=True)
class Crew:
    id: str
    role: str
    station: str
    actions: tuple


CREW = (
    tuple(
        Crew(f"{role}-{corner}", role, corner, actions)
        for corner in CORNERS
        for role, actions in (
            ("gunner", ("collect", "loosen", "tighten", "return")),
            ("wheel-off", ("remove", "stow")),
            ("wheel-on", ("collect", "fit", "withdraw")),
        )
    )
    + tuple(
        Crew(f"jack-{end}", "jack", end, ("lift", "lower")) for end in ("front", "rear")
    )
    + tuple(
        Crew(f"steadier-{side}", "steadier", side, ("brace", "clear"))
        for side in ("left", "right")
    )
    + tuple(
        Crew(f"wing-{side}", "wing", side, ("collect", "adjust", "return"))
        for side in ("left", "right")
    )
    + (Crew("chief", "chief", "chief", ("position", "release")),)
)


class PitStop:
    def __init__(self, publish=lambda *_: None, sleep=time.sleep, seed=None, werk=None):
        self.session = TemporaryDirectory(prefix="pit-stop-") if werk is None else None
        self.werk = werk if werk is not None else Werk(self.session.name)
        self.publish = publish
        self.active = {}
        self.sleep = sleep
        self.lock = RLock()
        self.crew = {member.id: member for member in CREW}
        self.seed = seed if seed is not None else secrets.randbits(32)
        self.movement = Movement(self.seed, CREW, DURATIONS)
        initial = {
            "items": self.equipment(),
            "car": "approaching",
            "held": None,
            "jacks": {end: "down" for end in ("front", "rear")},
            "steadiers": {side: "waiting" for side in ("left", "right")},
            "wings": {side: 0 for side in ("left", "right")},
            "wheels": {corner: "old-secured" for corner in CORNERS},
            "crew": {
                member.id: {
                    "action": None,
                    "done": [],
                    "clear": True,
                    "position": LAYOUT["crew"][member.id]["home"],
                    "heading": math.atan2(
                        LAYOUT["crew"][member.id]["work"][0]
                        - LAYOUT["crew"][member.id]["home"][0],
                        LAYOUT["crew"][member.id]["work"][1]
                        - LAYOUT["crew"][member.id]["home"][1],
                    ),
                    "equipment": None,
                }
                for member in CREW
            },
        }

        self._state = {}
        self.werk.on_event(self.observe)
        self.emit("pit_initialized", initial=initial, seed=self.seed)

    def observe(self, _, event):
        name, data = event.get_name(), event.get_data()
        if not name.startswith(("pit_", "car_", "action_")):
            return
        with self.lock:
            reduce_event(self._state, self.active, name, data)
            self.publish(name, {**data, "state": self.snapshot()})

    def rebuild(self):
        with self.lock:
            state, active = {}, {}
            for event in self.werk.find_events(
                lambda event: event.get_name().startswith(("pit_", "car_", "action_"))
            ):
                reduce_event(state, active, event.get_name(), event.get_data())
            return state, active

    def equipment(self):
        items = {}
        for member in CREW:
            if member.role in ("gunner", "wing"):
                name = f"tool-{member.id}"
                items[name] = {"kind": member.role, "owner": f"slot:{name}"}
        for corner in CORNERS:
            items[f"fresh-{corner}"] = {
                "kind": "fresh",
                "owner": f"slot:fresh-{corner}",
            }
            items[f"old-{corner}"] = {"kind": "old", "owner": f"hub:{corner}"}
        return items

    def snapshot(self):
        with self.lock:
            return deepcopy(self._state)

    def emit(self, name, **data):
        self.werk.emit_event(Event(name).data(data))

    def arrive(self):
        self.emit("car_arriving", **self.movement.arrival)
        self.sleep(self.movement.arrival["duration"])
        with self.lock:
            self.emit("car_stopped")

    def serviced(self):
        wheels_secured = all(
            wheel == "secured" for wheel in self._state["wheels"].values()
        )
        wings_adjusted = all(angle == 12 for angle in self._state["wings"].values())
        return wheels_secured and wings_adjusted

    def ready(self, actor, action):
        member = self.crew.get(actor)
        if member is None or action not in member.actions:
            return False
        state = self._state
        worker = state["crew"][actor]
        if state["held"] or state["car"] not in ("approaching", "stopped"):
            return False
        if worker["action"] or action in worker["done"]:
            return False
        if (
            worker["equipment"]
            and state["items"][worker["equipment"]]["owner"] != f"crew:{actor}"
        ):
            return False

        station = member.station
        if action == "collect":
            return worker["equipment"] is None
        if action == "position":
            return True
        if state["car"] != "stopped":
            return False
        if member.role in ("gunner", "wing") and worker["equipment"] != f"tool-{actor}":
            return False
        if action == "return":
            return ("tighten" if member.role == "gunner" else "adjust") in worker[
                "done"
            ]
        if action == "stow":
            return (
                "remove" in worker["done"]
                and worker["equipment"] == f"old-{station}"
                and state["items"][f"fresh-{station}"]["owner"]
                != f"slot:fresh-{station}"
            )
        if action == "withdraw":
            return "fit" in worker["done"] and worker["equipment"] is None
        if action == "lift":
            return state["jacks"][station] == "down"
        if action == "brace":
            return state["steadiers"][station] == "waiting"
        if action in WHEEL_PREREQUISITES:
            predecessor = {
                "remove": (f"gunner-{station}", "loosen"),
                "fit": (f"wheel-off-{station}", "remove"),
                "tighten": (f"wheel-on-{station}", "fit"),
            }.get(action)
            if (
                predecessor
                and predecessor[1] not in state["crew"][predecessor[0]]["done"]
            ):
                return False
            if action == "fit" and worker["equipment"] != f"fresh-{station}":
                return False
            jacks_up = all(value == "up" for value in state["jacks"].values())
            steadiers_braced = all(
                value == "braced" for value in state["steadiers"].values()
            )
            return (
                jacks_up
                and steadiers_braced
                and state["wheels"][station] == WHEEL_PREREQUISITES[action]
            )
        if action == "adjust":
            return all(value == "up" for value in state["jacks"].values())
        if action == "clear":
            return self.serviced() and state["steadiers"][station] == "braced"
        if action == "lower":
            return self.serviced() and all(
                value == "clear" for value in state["steadiers"].values()
            )
        if (
            action != "release"
            or "position" not in worker["done"]
            or not self.serviced()
        ):
            return False
        if any(value != "down" for value in state["jacks"].values()):
            return False
        if any(
            action not in state["crew"][member.id]["done"]
            for member in CREW
            if member.role != "chief"
            for action in member.actions
        ):
            return False
        return all(
            crew_member["clear"]
            and not crew_member["action"]
            and abs(crew_member["position"][1]) > LAYOUT["corridor"]
            for name, crew_member in state["crew"].items()
            if name != "chief"
        )

    def eligible(self):
        with self.lock:
            return [
                (member.id, action)
                for member in CREW
                for action in member.actions
                if self.ready(member.id, action)
            ]

    def perform(self, actor, action):
        with self.lock:
            if not self.ready(actor, action):
                raise ValueError(
                    f"{actor} cannot {action}: prerequisites are not satisfied"
                )
            worker = self._state["crew"][actor]
            try:
                phases = self.movement.plan(
                    self.crew[actor],
                    action,
                    worker["position"],
                    time.monotonic(),
                    self._state["crew"],
                    reservations=list(self.active.values()),
                )
            except ValueError as error:
                self.hold(str(error))
                raise
            self.emit(
                "action_started",
                actor=actor,
                action=action,
                duration=sum(phase["duration"] for phase in phases),
                phases=phases,
                began=time.monotonic(),
            )
        for index, phase in enumerate(phases):
            with self.lock:
                self.emit(
                    "action_phase",
                    actor=actor,
                    action=action,
                    phase=index,
                    kind=phase["kind"],
                    began=time.monotonic(),
                )
            self.sleep(phase["duration"])
            with self.lock:
                if self._state["held"]:
                    raise ValueError(
                        "The pit stop is held; no further work can complete"
                    )
                self.complete_phase(actor, action, phase)
        with self.lock:
            if action == "release" and abs(worker["position"][1]) <= LAYOUT["corridor"]:
                self.hold("The chief must clear the car's path before release")
                raise ValueError(self._state["held"])
            self.emit("action_completed", actor=actor, action=action)
            return {"actor": actor, "action": action, "completed": True}

    def complete_phase(self, actor, action, phase):
        self.emit(
            "action_phase_completed",
            actor=actor,
            action=action,
            position=phase["points"][-1],
            heading=end_heading(phase),
        )
        if transfer := phase.get("transfer"):
            item = self._state["items"][transfer["item"]]
            if item["owner"] != transfer["from"]:
                raise ValueError("Equipment changed owners before its handoff")
            self.emit("action_transfer", actor=actor, action=action, **transfer)
        if phase.get("effect"):
            self.emit(
                "action_effect",
                actor=actor,
                action=action,
                station=self.crew[actor].station,
            )

    def depart(self):
        with self.lock:
            if self._state["held"] or self._state["car"] != "released":
                raise ValueError("Departure requires the chief's release")
            if any(
                not worker["clear"]
                or worker["action"]
                or abs(worker["position"][1]) <= LAYOUT["corridor"]
                for worker in self._state["crew"].values()
            ):
                raise ValueError("Departure requires the chief's release")
            self.emit("car_departing", duration=4.0)
        self.sleep(4.0)
        with self.lock:
            self.emit("car_departed")

    def hold(self, message):
        with self.lock:
            if self._state["held"]:
                return
            self.emit("pit_held", message=message)


def reduce_event(state, active, name, data):
    """Project mechanical facts; only Werk's event log is authoritative."""
    if name == "pit_initialized":
        state.clear()
        state.update(deepcopy(data["initial"]))
        active.clear()
    elif name == "pit_held":
        state["held"] = data["message"]
        active.clear()
        for worker in state["crew"].values():
            worker["action"] = None
    elif name in ("car_stopped", "car_departing", "car_departed"):
        state["car"] = name.removeprefix("car_")
    elif name.startswith("action_") and "actor" in data:
        actor, action = data["actor"], data["action"]
        worker = state["crew"][actor]
        if name == "action_started":
            worker.update(action=action, clear=False)
            active[actor] = (actor, data["began"], data["phases"])
        elif name == "action_phase" and actor in active:
            _, _, phases = active[actor]
            # Keep the original phases so each observed index addresses the recorded plan.
            active[actor] = (
                actor,
                data["began"] - sum(p["duration"] for p in phases[: data["phase"]]),
                phases,
            )
        elif name == "action_phase_completed":
            worker.update(position=data["position"], heading=data["heading"])
        elif name == "action_transfer":
            state["items"][data["item"]]["owner"] = data["to"]
            worker["equipment"] = (
                data["item"] if data["to"] == f"crew:{actor}" else None
            )
        elif name == "action_effect":
            station = data["station"]
            if action in ("lift", "lower"):
                state["jacks"][station] = "up" if action == "lift" else "down"
            elif action in ("brace", "clear"):
                state["steadiers"][station] = "braced" if action == "brace" else "clear"
            elif action in WHEEL_RESULTS:
                state["wheels"][station] = WHEEL_RESULTS[action]
            elif action == "adjust":
                state["wings"][station] = 12
        elif name == "action_completed":
            worker["done"].append(action)
            worker.update(
                action=None, clear=abs(worker["position"][1]) > LAYOUT["corridor"]
            )
            active.pop(actor, None)
            if action == "release":
                state["car"] = "released"
