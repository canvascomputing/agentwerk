"""Physical facts, explicit crew tools, and validated pit-stop outcomes."""

import math
import random
import secrets
from copy import deepcopy
from dataclasses import dataclass
from tempfile import TemporaryDirectory

from agentwerk import Event, Werk

from movement import LAYOUT, Movement, end_heading, footprint
from sim_clock import SimulationClock

CORNERS = ("rear-left", "front-left", "rear-right", "front-right")
ROLES = {
    "gunner": 4,
    "wheel-off": 4,
    "wheel-on": 4,
    "jack": 2,
    "steadier": 2,
    "wing": 2,
    "chief": 1,
}
WORK = {
    "gunner": ("loosen", "tighten"),
    "wheel-off": ("remove",),
    "wheel-on": ("fit",),
    "jack": ("lift", "lower"),
    "steadier": ("brace", "clear"),
    "wing": ("adjust",),
    "chief": (),
}
DURATIONS = {
    "pickup": 0.6,
    "drop": 0.6,
    "lift": 1.2,
    "lower": 1.2,
    "brace": 0.6,
    "clear": 0.4,
    "loosen": 0.8,
    "tighten": 1.0,
    "remove": 1.1,
    "fit": 1.3,
    "adjust": 1.5,
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
WHEEL_EVENTS = {
    "loosen": "wheel_loosened",
    "remove": "wheel_removed",
    "fit": "wheel_fitted",
}
MILESTONES = {
    "car_approaching",
    "car_arriving",
    "car_stopped",
    "pit_service_started",
    "car_lifted",
    "pit_service_completed",
    "car_unbraced",
    "car_lowered",
    "pit_crew_clear",
    "pit_released",
    "pit_held",
    "car_departing",
    "car_departed",
}


@dataclass(frozen=True)
class Crew:
    id: str
    role: str


CREW = tuple(
    Crew("chief" if role == "chief" else f"{role}-{i + 1}", role)
    for role, count in ROLES.items()
    for i in range(count)
)


def setup(seed):
    rng = random.Random(seed)
    layout = deepcopy(LAYOUT)
    layout["crew"], layout["destinations"] = {}, {}
    assignments = {}
    for role, count in ROLES.items():
        targets = (
            list(CORNERS)
            if count == 4
            else ["front", "rear"]
            if role == "jack"
            else ["left", "right"]
            if count == 2
            else ["chief"]
        )
        rng.shuffle(targets)
        for i, target in enumerate(targets):
            actor = "chief" if role == "chief" else f"{role}-{i + 1}"
            assignments[actor] = target
            old = deepcopy(
                LAYOUT["crew"][f"{role}-{target}" if role != "chief" else "chief"]
            )
            if role == "chief":
                old["work"] = [6.2, 0]
            if role == "jack":
                old["work"] = [4.65 if target == "front" else -4.65, 0]
            layout["crew"][actor] = old
            destinations = layout["destinations"]
            destinations["chief-home" if role == "chief" else f"parking:{actor}"] = old[
                "home"
            ]
            destinations[
                "pit-board" if role == "chief" else f"work:{role}:{target}"
            ] = old["work"]
            stage = [old["work"][0], math.copysign(2.8, old["work"][1] or -1)]
            if role == "jack":
                stage = [5.2, -2.5] if target == "front" else [-3.8, -2.95]
            if role == "chief":
                stage = [5.3, -2.8]
            destinations[
                "chief-clear" if role == "chief" else f"stage:{role}:{target}"
            ] = stage
            if role in ("gunner", "wheel-off", "wheel-on", "wing"):
                distance = {
                    "gunner": 3.4,
                    "wheel-off": 4.1,
                    "wheel-on": 4.8,
                    "wing": 3.5,
                }[role]
                destinations[f"holding:{role}:{target}"] = [
                    -2.4
                    if role == "gunner" and target.startswith("rear-")
                    else old["work"][0],
                    math.copysign(distance, old["work"][1]),
                ]
    parking = {
        -1: [
            [-3.85, -3.6],
            [-2.6, -4.8],
            [-1.4, -3.6],
            [-0.6, -4.0],
            [0.2, -4.6],
            [0.2, -3.6],
            [1.0, -4.0],
            [2.6, -3.6],
            [3.4, -4.6],
            [4.0, -3.8],
        ],
        1: [
            [-3.4, 3.6],
            [-2.8, 4.4],
            [-1.4, 3.6],
            [-0.8, 4.4],
            [-0.4, 3.6],
            [0.6, 3.6],
            [2.6, 3.6],
            [3.6, 4.6],
            [4.0, 3.8],
        ],
    }
    for side, positions in parking.items():
        members = sorted(
            (
                actor
                for actor, spec in layout["crew"].items()
                if (1 if actor == "chief" or spec["home"][1] > 0 else -1) == side
            ),
            key=lambda actor: layout["crew"][actor]["home"][0],
        )
        for actor, home in zip(members, positions, strict=True):
            layout["crew"][actor]["home"] = home
            layout["destinations"][
                "chief-home" if actor == "chief" else f"parking:{actor}"
            ] = home
    layout["jack"] = {"handle": [-0.72, 0.78, 0], "grip_forward": 0.33, "reach": 1.15}
    slots = {}
    for i, point in enumerate(
        p for key, p in LAYOUT["slots"].items() if key.startswith("tool-")
    ):
        slots[f"bench-{i + 1}"] = point
    for i, corner in enumerate(CORNERS):
        slots[f"tire-{i + 1}"] = LAYOUT["slots"][f"fresh-{corner}"]
    for end, x in (("front", 4.6), ("rear", -3.2)):
        slots[f"jack-{end}"] = [x, 0, -3.95 if end == "front" else -4.45]
    layout["slots"] = slots
    for name, point in slots.items():
        layout["destinations"][f"storage:{name}"] = [
            point[0],
            point[2]
            - math.copysign(1.45 if name.startswith("jack-") else 0.8, point[2]),
        ]
    bench = [name for name in slots if name.startswith("bench")]
    rng.shuffle(bench)
    kinds = ["gunner"] * 4 + ["wing"] * 2
    items = {
        f"{'wheel-gun' if kind == 'gunner' else 'wing-key'}-{i + 1}": {
            "kind": kind,
            "storage": slot,
            "owner": f"slot:{slot}",
        }
        for i, (kind, slot) in enumerate(zip(kinds, bench))
    }
    tires = [name for name in slots if name.startswith("tire")]
    rng.shuffle(tires)
    for corner, slot in zip(CORNERS, tires):
        items[f"fresh-{corner}"] = {
            "kind": "fresh",
            "corner": corner,
            "storage": slot,
            "owner": f"slot:{slot}",
        }
        items[f"old-{corner}"] = {
            "kind": "old",
            "corner": corner,
            "storage": slot,
            "owner": f"hub:{corner}",
        }
    for end in ("front", "rear"):
        items[f"jack-{end}"] = {
            "kind": "jack",
            "end": end,
            "storage": f"jack-{end}",
            "owner": f"slot:jack-{end}",
        }
    wing_angles = {side: rng.choice((6, 9, 12)) for side in ("left", "right")}
    arrival = {
        "warning": rng.uniform(8, 12),
        "duration": rng.uniform(3.5, 5),
        "offset": rng.uniform(-0.45, 0.45),
        "braking": rng.uniform(1.8, 2.5),
    }
    return layout, assignments, items, wing_angles, arrival


class PitStop:
    def __init__(self, publish=lambda *_: None, seed=None, werk=None, realtime=True):
        self.session = TemporaryDirectory(prefix="pit-stop-") if werk is None else None
        self.werk = werk or Werk(self.session.name)
        self.seed = seed if seed is not None else secrets.randbits(32)
        self.layout, self.assignments, items, wing_angles, self.arrival = setup(
            self.seed
        )
        self.clock = SimulationClock(self.emit, realtime=realtime)
        self.lock = self.clock.lock
        self.publish = publish
        self.movement = Movement(self.layout)
        self.crew = {member.id: member for member in CREW}
        self.active = {}
        self.claims = {}
        self.milestones = set()
        self._state = {}
        self.werk.on_event(self.observe)
        self.emit(
            "pit_initialized",
            initial={
                "car": "approaching",
                "held": None,
                "items": items,
                "wing_angles": wing_angles,
                "wheels": dict.fromkeys(CORNERS, "old-secured"),
                "wings": dict.fromkeys(wing_angles, 0),
                "jacks": dict.fromkeys(("front", "rear"), "down"),
                "steadiers": dict.fromkeys(wing_angles, "waiting"),
                "crew": {
                    m.id: {
                        "role": m.role,
                        "station": self.assignments[m.id],
                        "position": self.layout["crew"][m.id]["home"],
                        "location": "chief-home"
                        if m.id == "chief"
                        else f"parking:{m.id}",
                        "heading": math.atan2(
                            -self.layout["crew"][m.id]["home"][0],
                            -self.layout["crew"][m.id]["home"][1],
                        ),
                        "equipment": None,
                        "task": None,
                        "done": [],
                        "clear": True,
                    }
                    for m in CREW
                },
            },
        )

    def observe(self, _, event):
        name, data = event.get_name(), event.get_data()
        if not name.startswith(("pit_", "car_", "crew_")):
            return
        data = data if isinstance(data, dict) else {"data": data}
        with self.lock:
            reduce_event(self._state, self.active, name, data)
            self.publish(name, {**data, "state": self.snapshot()})

    def emit(self, name, **data):
        self.werk.emit_event(Event(name).data({"seconds": self.clock.seconds, **data}))

    def milestone(self, name, **data):
        with self.lock:
            if name not in self.milestones:
                self.milestones.add(name)
                self.emit(name, **data)

    def snapshot(self):
        with self.lock:
            return deepcopy(self._state)

    def rebuild(self):
        state, active = {}, {}
        for event in self.werk.find_events(
            lambda e: e.get_name().startswith(("pit_", "car_", "crew_"))
        ):
            reduce_event(state, active, event.get_name(), event.get_data())
        return state, active

    def observation(self, actor):
        state = self.snapshot()
        return {
            "seconds": self.clock.seconds,
            "arrival_at": self.arrival["warning"] + self.arrival["duration"],
            "busy_destinations": {
                name: other
                for name, point in self.layout["destinations"].items()
                for other, worker in state["crew"].items()
                if other != actor
                and math.dist(
                    point,
                    self.active[other][2][-1]["points"][-1]
                    if other in self.active
                    else worker["position"],
                )
                < 0.56
            },
            "self": state["crew"][actor],
            "car": state["car"],
            "items": state["items"],
            "wheels": state["wheels"],
            "wings": state["wings"],
            "jacks": state["jacks"],
            "steadiers": state["steadiers"],
            "wing_angles": state["wing_angles"],
        }

    def approach(self):
        self.clock.decide("@car")
        self.milestone(
            "car_approaching",
            arrives_in_seconds=self.arrival["warning"] + self.arrival["duration"],
        )
        self.clock.start()

    def arrive(self):
        try:
            self.clock.wait(self.arrival["warning"], "@car")
            self.milestone("car_arriving", **self.arrival)
            self.clock.wait(self.arrival["duration"], "@car")
            self.milestone("car_stopped")
            self.milestone("pit_service_started")
        finally:
            self.clock.idle("@car")

    def depart(self):
        with self.lock:
            if self._state["car"] != "released":
                raise ValueError("Departure requires the Chief's GO")
            self.milestone("car_departing", duration=4)
        self.clock.wait(4, "@car")
        self.milestone("car_departed")
        self.clock.idle("@car")

    def serviced(self):
        return (
            all(v == "secured" for v in self._state["wheels"].values())
            and self._state["wings"] == self._state["wing_angles"]
        )

    def clear(self, crew=CREW):
        state = self._state
        workers = [state["crew"][m.id] for m in crew]
        return (
            self.serviced()
            and all(v == "down" for v in state["jacks"].values())
            and all(v == "clear" for v in state["steadiers"].values())
            and all(
                w["clear"] and not w["task"] and not w["equipment"] for w in workers
            )
            and all(
                item["owner"] == f"slot:{item['storage']}"
                for item in state["items"].values()
                if item["kind"] == "jack"
            )
            and all(
                not item["owner"].startswith(("crew:", "mount:"))
                for item in state["items"].values()
            )
        )

    def release(self):
        self.milestone("pit_released")

    def hold(self, message):
        self.milestone("pit_held", message=message)
        self.clock.close()

    def available(self, actor):
        if self._state["held"] or self._state["car"] not in ("approaching", "stopped"):
            raise ValueError("The pit stop is no longer accepting work")
        if self._state["crew"][actor]["task"]:
            raise ValueError(
                "You are already moving or working. Wait for that action to finish."
            )

    def move(self, actor, destination, pace):
        with self.lock:
            self.available(actor)
            if destination not in self.layout["destinations"] or pace not in (
                "walk",
                "run",
            ):
                raise ValueError("Choose a named destination and pace walk or run")
            worker = self._state["crew"][actor]
            if (
                worker["role"] == "steadier"
                and self._state["steadiers"][worker["station"]] == "braced"
            ):
                raise ValueError(
                    "Clear the car with operate before leaving the bracing position"
                )
            if worker["role"] == "jack" and self._state["items"][
                f"jack-{worker['station']}"
            ]["owner"].startswith("mount:"):
                raise ValueError("Lower and withdraw your jack before moving away")
            target = self.layout["destinations"][destination]
            occupied = [
                point
                for name, other in self._state["crew"].items()
                if name != actor and name not in self.active
                for point in footprint(
                    other["position"], other["heading"], self.reach(other)
                )
            ]
            occupied.extend(
                self.layout["slots"][item["owner"][5:]][::2]
                for item in self._state["items"].values()
                if item["kind"] == "jack"
                and item["owner"].startswith("slot:")
                and destination != f"storage:{item['storage']}"
            )
            try:
                phases = self.movement.plan(
                    worker["position"],
                    target,
                    pace,
                    worker["heading"],
                    self.clock.seconds,
                    occupied,
                    list(self.active.values()),
                    self._state["car"] == "approaching",
                    bool(worker["equipment"]),
                    facing=self.facing(destination),
                    end_reach=self.layout["jack"]["reach"]
                    if destination.startswith("storage:jack-")
                    and worker["role"] == "jack"
                    else None,
                    reach=self.layout["jack"]["reach"]
                    if self._state["items"].get(worker["equipment"], {}).get("kind")
                    == "jack"
                    else 0,
                )
            except ValueError as error:
                rejected = error
            else:
                rejected = None
                self.begin(actor, "move", phases, destination=destination, pace=pace)
        if rejected:
            # Time stays paused while an agent decides, so a busy route would never clear.
            self.clock.wait(0.5, actor)
            raise rejected
        self.execute(actor, "move", phases, destination=destination)
        return self.observation(actor)

    def reach(self, worker):
        jack = self._state["items"].get(worker["equipment"], {}).get("kind") == "jack"
        positioned = worker["role"] == "jack" and worker["location"].startswith(
            ("work:jack:", "storage:jack-")
        )
        return self.layout["jack"]["reach"] if jack or positioned else 0

    def facing(self, destination):
        point = self.layout["destinations"][destination]
        parts = destination.split(":")
        if destination in ("pit-board", "chief-home", "chief-clear"):
            target = [0, 0]
        elif parts[0] == "storage":
            target = self.layout["slots"][parts[1]][::2]
        elif parts[0] != "parking" and parts[1] == "jack":
            target = [3.5 if parts[2] == "front" else -3.5, 0]
        elif parts[0] == "parking" or parts[1] == "chief":
            target = [0, 0]
        else:
            target = LAYOUT["corners"].get(
                parts[2], [3 if parts[1] == "wing" else 0, 0]
            )
        return math.atan2(target[0] - point[0], target[1] - point[1])

    def near(self, actor, destination):
        return (
            math.dist(
                self._state["crew"][actor]["position"],
                self.layout["destinations"][destination],
            )
            < 0.12
        )

    def operate(self, actor, task, item=None, target=None, work=None, value=None):
        with self.lock:
            self.available(actor)
            worker = self._state["crew"][actor]
            transfer, effect = None, None
            resource = None
            if task in ("pickup", "drop"):
                equipment = self._state["items"].get(item)
                if equipment is None:
                    raise ValueError("Choose an item from the inventory")
                mounted = equipment["owner"].startswith("mount:")
                if task == "pickup":
                    if worker["equipment"] or not (
                        equipment["owner"].startswith("slot:") or mounted
                    ):
                        raise ValueError("Hands must be empty and the item available")
                    if mounted and (
                        worker["role"] != "jack"
                        or worker["station"] != equipment["end"]
                        or self._state["jacks"][equipment["end"]] != "down"
                    ):
                        raise ValueError(
                            "Only the assigned operator can withdraw a lowered jack"
                        )
                    slot = (
                        equipment["storage"]
                        if mounted
                        else equipment["owner"].removeprefix("slot:")
                    )
                    source, dest = equipment["owner"], f"crew:{actor}"
                else:
                    slot = target
                    if (
                        worker["equipment"] != item
                        or equipment["owner"] != f"crew:{actor}"
                    ):
                        raise ValueError("You must hold the item to drop it")
                    if slot not in self.layout["slots"]:
                        raise ValueError("Choose a storage slot as target")
                    compatible = (
                        slot == equipment["storage"]
                        if equipment["kind"] == "jack"
                        else slot.startswith("tire-")
                        if equipment["kind"] in ("fresh", "old")
                        else slot.startswith("bench-")
                    )
                    if not compatible:
                        raise ValueError(
                            "Use tire platforms for tires, benches for tools, and designated jack storage"
                        )
                    if any(
                        e["owner"] == f"slot:{slot}"
                        for e in self._state["items"].values()
                    ):
                        raise ValueError(
                            "That storage slot is occupied. Choose an empty slot for this equipment."
                        )
                    source, dest = f"crew:{actor}", f"slot:{slot}"
                destination = (
                    f"work:jack:{equipment['end']}" if mounted else f"storage:{slot}"
                )
                if not self.near(actor, destination):
                    raise ValueError(f"Move to {destination} before handling this item")
                resource = equipment["owner"] if mounted else f"slot:{slot}"
                transfer = {"item": item, "from": source, "to": dest}
                kind = "grip" if task == "pickup" else "place"
                face = (
                    [3.5 if equipment["end"] == "front" else -3.5, 0]
                    if mounted
                    else self.layout["slots"][slot][::2]
                )
                operation = task
            elif task == "use":
                self.validate_work(actor, work, target, value)
                resource = f"work:{worker['role']}:{target}"
                effect = {"work": work, "target": target, "value": value}
                operation = work
                kind = {"remove": "pull", "fit": "seat"}.get(work, "work")
                face = LAYOUT["corners"].get(
                    target, [3 if worker["role"] == "wing" else 0, 0]
                )
                if work == "lift":
                    transfer = {
                        "item": worker["equipment"],
                        "from": f"crew:{actor}",
                        "to": f"mount:{target}",
                    }
                if work == "remove":
                    transfer = {
                        "item": f"old-{target}",
                        "from": f"hub:{target}",
                        "to": f"crew:{actor}",
                    }
                if work == "fit":
                    transfer = {
                        "item": worker["equipment"],
                        "from": f"crew:{actor}",
                        "to": f"hub:{target}",
                    }
            else:
                raise ValueError("Choose operate task pickup, drop, or use")
            if resource in self.claims:
                raise ValueError("Another worker is using that equipment or position")
            self.claims[resource] = actor
            origin = worker["position"]
            heading = math.atan2(face[0] - origin[0], face[1] - origin[1])
            phases = [
                {
                    "kind": "turn",
                    "duration": 0.15,
                    "points": [origin],
                    "headings": [worker["heading"], heading],
                },
                {
                    "kind": kind,
                    "duration": DURATIONS[operation],
                    "points": [origin],
                    "heading": heading,
                    "transfer": transfer,
                    "effect": effect,
                },
            ]
            self.begin(actor, operation, phases, target=target, item=item, value=value)
        try:
            self.execute(actor, operation, phases)
        finally:
            with self.lock:
                self.claims.pop(resource, None)
        return self.observation(actor)

    def validate_work(self, actor, work, target, value):
        state, worker = self._state, self._state["crew"][actor]
        role = worker["role"]
        if work not in WORK[role] or target != worker["station"]:
            raise ValueError(
                "This work or target is outside your assigned task. Follow your task's assignment."
            )
        if state["car"] != "stopped" or not self.near(actor, f"work:{role}:{target}"):
            raise ValueError(
                "The car must be stopped and you must be at the assigned work position"
            )
        if work in worker["done"]:
            raise ValueError(
                "You have already completed this work. Go to the requested finish position and report completion."
            )
        equipment = state["items"].get(worker["equipment"], {})
        if role in ("gunner", "wing") and equipment.get("kind") != role:
            raise ValueError(
                f"Wrong equipment: this work requires a {'wheel gun' if role == 'gunner' else 'wing key'}"
            )
        if role in ("steadier", "wheel-off") and worker["equipment"]:
            raise ValueError("This work requires empty hands")
        if work == "lift" and (
            equipment.get("kind") != "jack" or equipment.get("end") != target
        ):
            raise ValueError("Hold the jack for your assigned end")
        if work == "lower" and (
            state["items"][f"jack-{target}"]["owner"] != f"mount:{target}"
            or worker["equipment"]
        ):
            raise ValueError("Lower the engaged jack with empty hands")
        if work in WHEEL_PREREQUISITES:
            if not all(v == "up" for v in state["jacks"].values()) or not all(
                v == "braced" for v in state["steadiers"].values()
            ):
                raise ValueError("Both jacks must be up and both steadiers braced")
            if state["wheels"][target] != WHEEL_PREREQUISITES[work]:
                raise ValueError(
                    f"The wheel must be {WHEEL_PREREQUISITES[work]} before {work}"
                )
            if work == "fit" and (
                equipment.get("kind") != "fresh" or equipment.get("corner") != target
            ):
                raise ValueError("Hold the fresh tire specified for this corner")
        if work == "adjust" and (
            not all(v == "up" for v in state["jacks"].values())
            or not all(v == "braced" for v in state["steadiers"].values())
            or value != state["wing_angles"][target]
        ):
            raise ValueError(
                "Both jacks must be up and both steadiers braced. Use the requested wing angle."
            )
        if work in ("clear", "lower") and not self.serviced():
            raise ValueError(
                "Complete wheel and wing service before clearing or lowering"
            )
        if work == "lower" and not all(
            v == "clear" for v in state["steadiers"].values()
        ):
            raise ValueError("Both steadiers must clear before lowering")

    def begin(self, actor, task, phases, **data):
        worker = self._state["crew"][actor]
        if (
            self.reach(worker)
            and task != "move"
            or self._state["items"].get(data.get("item"), {}).get("kind") == "jack"
        ):
            for phase in phases:
                phase["reach"] = self.layout["jack"]["reach"]
        self.emit(
            "crew_task_started",
            actor=actor,
            task=task,
            phases=phases,
            duration=sum(p["duration"] for p in phases),
            began=self.clock.seconds,
            **data,
        )

    def execute(self, actor, task, phases, destination=None):
        for index, phase in enumerate(phases):
            with self.lock:
                self.emit(
                    "crew_task_phase",
                    actor=actor,
                    task=task,
                    phase=index,
                    kind=phase["kind"],
                    began=self.clock.seconds,
                )
            self.clock.wait(phase["duration"], actor)
            with self.lock:
                if self._state["held"]:
                    raise ValueError("The pit stop is held")
                self.emit(
                    "crew_task_phase_completed",
                    actor=actor,
                    task=task,
                    position=phase["points"][-1],
                    heading=end_heading(phase),
                )
                if transfer := phase.get("transfer"):
                    if (
                        self._state["items"][transfer["item"]]["owner"]
                        != transfer["from"]
                    ):
                        raise ValueError(
                            "The item moved during handling. Report blocked."
                        )
                    self.emit("crew_transfer", actor=actor, **transfer)
                if effect := phase.get("effect"):
                    self.emit("crew_work", actor=actor, **effect)
                    if handoff := WHEEL_EVENTS.get(effect["work"]):
                        self.emit(handoff, actor=actor, corner=effect["target"])
                    jacks, steadiers = self._state["jacks"], self._state["steadiers"]
                    if all(v == "up" for v in jacks.values()) and all(
                        v == "braced" for v in steadiers.values()
                    ):
                        self.milestone("car_lifted")
                    if all(v == "clear" for v in steadiers.values()):
                        self.milestone("car_unbraced")
                    if self.serviced() and all(v == "down" for v in jacks.values()):
                        self.milestone("car_lowered")
        with self.lock:
            self.emit(
                "crew_task_completed", actor=actor, task=task, destination=destination
            )
            if self.serviced():
                self.milestone(
                    "pit_service_completed", wing_angles=self._state["wing_angles"]
                )
            # The Chief holds the board in front of the car until GO.
            if self.clear([m for m in CREW if m.role != "chief"]):
                self.milestone("pit_crew_clear")


def reduce_event(state, active, name, data):
    if name == "pit_initialized":
        state.clear()
        state.update(deepcopy(data["initial"]))
        active.clear()
    elif name == "pit_held":
        state["held"] = data["message"]
        active.clear()
    elif name in ("car_stopped", "car_departing", "car_departed", "pit_released"):
        state["car"] = (
            "released" if name == "pit_released" else name.removeprefix("car_")
        )
    elif name.startswith("crew_") and "actor" in data:
        actor = data["actor"]
        worker = state["crew"][actor]
        if name == "crew_task_started":
            worker["task"] = data["task"]
            active[actor] = (actor, data["began"], data["phases"])
        elif name == "crew_task_phase_completed":
            worker.update(position=data["position"], heading=data["heading"])
        elif name == "crew_transfer":
            state["items"][data["item"]]["owner"] = data["to"]
            worker["equipment"] = (
                data["item"] if data["to"] == f"crew:{actor}" else None
            )
        elif name == "crew_work":
            work, target = data["work"], data["target"]
            worker["done"].append(work)
            if work in WHEEL_RESULTS:
                state["wheels"][target] = WHEEL_RESULTS[work]
            elif work in ("lift", "lower"):
                state["jacks"][target] = "up" if work == "lift" else "down"
            elif work in ("brace", "clear"):
                state["steadiers"][target] = "braced" if work == "brace" else "clear"
            elif work == "adjust":
                state["wings"][target] = data["value"]
        elif name == "crew_task_completed":
            worker["task"] = None
            if data.get("destination"):
                worker["location"] = data["destination"]
            worker["clear"] = abs(worker["position"][1]) > LAYOUT["corridor"]
            active.pop(actor, None)
