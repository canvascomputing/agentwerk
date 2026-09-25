"""Seeded handling, apron routes, and local crossing reservations."""

import json
import math
import random
from itertools import pairwise
from pathlib import Path

LAYOUT = json.loads(Path(__file__).with_name("layout.json").read_text())


def route(start, target, lane):
    def apron_access(point):
        x, z = point
        for cx, cz in LAYOUT["stations"].values():
            crosses = min(z, lane) < cz < max(z, lane)
            if crosses and abs(x - cx) < 2.3:
                back = cz + math.copysign(0.67, cz)
                edge = cx + math.copysign(2.3, x - cx)
                if abs(z) < abs(cz):
                    return [list(point), [edge, z]]
                return [list(point), [x, back], [edge, back]]
        return [list(point)]

    entry, exit = apron_access(start), apron_access(target)
    corners = [*entry, [entry[-1][0], lane], [exit[-1][0], lane], *reversed(exit)]
    corners = [p for i, p in enumerate(corners) if i == 0 or p != corners[i - 1]]
    points = [corners[0]]
    for index in range(1, len(corners) - 1):
        before, corner, after = corners[index - 1 : index + 2]
        radius = min(0.55, math.dist(before, corner) / 3, math.dist(corner, after) / 3)
        entry = [
            corner[i] + (before[i] - corner[i]) * radius / math.dist(before, corner)
            for i in range(2)
        ]
        leave = [
            corner[i] + (after[i] - corner[i]) * radius / math.dist(after, corner)
            for i in range(2)
        ]
        points.append(entry)
        for step in range(1, 13):
            p = step / 12
            points.append(
                [
                    (1 - p) ** 2 * entry[i]
                    + 2 * p * (1 - p) * corner[i]
                    + p * p * leave[i]
                    for i in range(2)
                ]
            )
    if corners[-1] != points[-1]:
        points.append(corners[-1])
    curved = [points[0]]
    for a, b in pairwise(points):
        length = math.dist(a, b)
        steps = max(1, math.ceil(length / 0.12))
        bow = (
            min(0.12, length * 0.035)
            if length > 0.8
            and abs(a[1]) < 3.5
            and abs(b[1]) < 3.5
            and abs(b[0] - a[0]) > abs(b[1] - a[1])
            else 0
        )
        for step in range(1, steps + 1):
            if step == steps:
                curved.append(list(b))
                continue
            t = step / steps
            bend = bow * math.sin(math.pi * t) ** 2
            curved.append(
                [
                    a[0] + (b[0] - a[0]) * t - (b[1] - a[1]) / length * bend,
                    a[1] + (b[1] - a[1]) * t + (b[0] - a[0]) / length * bend,
                ]
            )
    for point in curved:
        point[0] = max(
            min(p[0] for p in corners), min(max(p[0] for p in corners), point[0])
        )
    return curved


def distance(points):
    return sum(math.dist(a, b) for a, b in pairwise(points))


def position(phases, seconds):
    for phase in phases:
        points = phase["points"]
        duration = phase["duration"]
        if seconds > duration:
            seconds -= duration
            continue
        progress = min(1, max(0, seconds / duration))
        if phase.get("easing") == "smooth":
            progress = progress * progress * (3 - 2 * progress)
        distance_left = distance(points) * progress
        for start, end in pairwise(points):
            length = math.dist(start, end)
            if distance_left <= length and length:
                return [
                    start[i] + (end[i] - start[i]) * distance_left / length
                    for i in range(2)
                ]
            distance_left -= length
        return points[-1]
    return phases[-1]["points"][-1]


def end_heading(phase):
    if phase.get("headings"):
        return phase["headings"][-1]
    points = phase["points"]
    if len(points) > 1:
        a, b = points[-2:]
        return math.atan2(b[0] - a[0], b[1] - a[1])
    return phase.get("heading", 0)


class Movement:
    def __init__(self, seed, crew, durations):
        rng = random.Random(seed)
        self.choices = {}
        for member in crew:
            self.choices[member.id] = {
                "pace": rng.uniform(0.8, 1.3),
                "lane_offset": rng.uniform(-0.04, 0.04),
                "actions": {
                    action: {
                        "reaction": rng.uniform(0.1, 1.4),
                        "work": durations[action] * rng.uniform(0.65, 1.55),
                        "handling": rng.uniform(1.2, 2.2),
                        "variant": rng.randrange(3),
                        "pose": rng.uniform(0.85, 1.15),
                    }
                    for action in member.actions
                },
            }
        self.arrival = {
            "duration": rng.uniform(3.5, 5),
            "offset": rng.uniform(-0.45, 0.45),
            "braking": rng.uniform(1.8, 2.5),
        }

    def plan(self, member, action, start, now, crew=None, reservations=()):
        actor = member.id
        place = LAYOUT["crew"][actor]
        choice = self.choices[actor]
        timing = choice["actions"][action]
        side = math.copysign(1, place["home"][1])
        phases = [
            {
                "kind": "wait",
                "duration": min(timing["reaction"], 0.04)
                if action == "release"
                else timing["reaction"],
                "points": [start],
                "heading": (crew or {}).get(actor, {}).get("heading", 0),
            }
        ]
        active = {entry[0] for entry in reservations}
        occupied = [
            worker["position"]
            for name, worker in (crew or {}).items()
            if name != actor and name not in active
        ]

        def pause(kind, duration=None, **data):
            phases.append(
                {
                    "kind": kind,
                    "duration": duration or timing["handling"],
                    "points": [phases[-1]["points"][-1]],
                    "pose": timing["pose"],
                    "heading": end_heading(phases[-1]),
                    **data,
                }
            )

        def face(target):
            origin = phases[-1]["points"][-1]
            heading = math.atan2(target[0] - origin[0], target[1] - origin[1])
            if (
                abs(
                    math.atan2(
                        math.sin(heading - end_heading(phases[-1])),
                        math.cos(heading - end_heading(phases[-1])),
                    )
                )
                < 0.01
            ):
                return
            pause(
                "turn",
                timing["handling"] * 0.3,
                headings=[end_heading(phases[-1]), heading],
            )

        def walk(target, kind="move", loaded=False):
            origin = phases[-1]["points"][-1]
            if origin == target:
                return
            # Yield at this journey's safe waypoint, not back at the start of the task.
            for delay in range(151):
                for offset in range(3):
                    variant = (timing["variant"] + offset) % 3
                    lane = side * (3.12, 3.25, 5.17)[variant]
                    if variant < 2:
                        lane += choice["lane_offset"]
                    if member.role == "chief":
                        lane = side * 3.25
                    points = route(origin, target, lane)
                    if any(
                        abs(x - cx) < 1.9 and abs(z - cz) < 0.62
                        for x, z in points
                        for cx, cz in LAYOUT["stations"].values()
                    ):
                        continue
                    travel = {
                        "kind": kind,
                        "duration": distance(points)
                        / (
                            2.8
                            * choice["pace"]
                            * (2 if action == "release" else 0.82 if loaded else 1)
                        ),
                        "points": points,
                        "easing": "smooth",
                    }
                    waiting = (
                        [{"kind": "yield", "duration": delay * 0.2, "points": [origin]}]
                        if delay
                        else []
                    )
                    heading = math.atan2(
                        points[1][0] - origin[0], points[1][1] - origin[1]
                    )
                    turning = {
                        "kind": "turn",
                        "duration": 0.1
                        if action == "release"
                        else timing["handling"] * 0.3,
                        "points": [origin],
                        "headings": [end_heading(phases[-1]), heading],
                    }
                    for wait in waiting:
                        wait["heading"] = end_heading(phases[-1])
                    candidate = phases + waiting + [turning, travel]
                    # A clear arrival is insufficient if another worker crosses during handling.
                    handling = {
                        "kind": "wait",
                        "duration": timing["handling"] * 4 + timing["work"],
                        "points": [target],
                    }
                    if not self.conflicts(
                        candidate + [handling], now, occupied, reservations
                    ):
                        phases.extend(waiting + [turning, travel])
                        return
            raise ValueError(f"The crew route is obstructed for {actor}: {action}")

        def transfer(kind, item, source, target, effect=False):
            pause(
                kind,
                transfer={"item": item, "from": source, "to": target},
                effect=effect,
            )

        owner = f"crew:{actor}"
        tool = f"tool-{actor}"
        corner = member.station
        if action == "collect":
            item = f"fresh-{corner}" if member.role == "wheel-on" else tool
            walk(place["station"])
            face(LAYOUT["slots"][item][::2])
            pause("reach")
            transfer("grip", item, f"slot:{item}", owner)
            pause("lift_item")
            walk(place.get("stage", place["home"]), "stage", loaded=True)
        elif action in ("return", "stow"):
            item = f"old-{corner}" if action == "stow" else tool
            slot = f"used-{corner}" if action == "stow" else tool
            walk(place["station"], "return", loaded=True)
            face(LAYOUT["slots"][slot][::2])
            transfer("place", item, owner, f"slot:{slot}")
            pause("release_item", timing["handling"] * 0.45)
            walk(place["home"], "withdraw")
        elif action == "withdraw":
            walk(place["home"], "withdraw")
        else:
            walk(place["work"], loaded=action == "fit")
            target = LAYOUT["corners"].get(
                corner, [3 if member.role == "wing" else 0, 0]
            )
            face(target)
            if action == "remove":
                pause("reach")
                transfer("pull", f"old-{corner}", f"hub:{corner}", owner, effect=True)
                pause("lift_item", timing["handling"] * 0.6)
            elif action == "fit":
                pause("align")
                transfer("seat", f"fresh-{corner}", owner, f"hub:{corner}", effect=True)
                pause("release_item", timing["handling"] * 0.45)
            else:
                pause("work", timing["work"], effect=True)
            if action in ("remove", "fit", "tighten", "adjust"):
                walk(place["aside"], "step_aside", loaded=action == "remove")
            elif action in ("clear", "lower", "release"):
                walk(place["home"], "withdraw")
        if self.conflicts(phases, now, occupied, reservations):
            raise ValueError(f"The work position is obstructed for {actor}: {action}")
        return phases

    def conflicts(self, phases, now, occupied=(), reservations=()):
        duration = sum(phase["duration"] for phase in phases)
        for step in range(math.ceil(duration / 0.08) + 1):
            elapsed = min(duration, step * 0.08)
            point = position(phases, elapsed)
            # Leave room for phase-clock jitter while other agents plan their routes.
            if any(math.dist(point, other) < 0.60 for other in occupied):
                return True
            for _, began, other in reservations:
                if math.dist(point, position(other, now + elapsed - began)) < 0.60:
                    return True
        return False
