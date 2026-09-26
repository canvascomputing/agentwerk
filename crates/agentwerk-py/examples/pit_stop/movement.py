"""Seeded handling, apron routes, and local crossing reservations."""

import heapq
import json
import math
from bisect import bisect_left
from itertools import pairwise
from pathlib import Path

LAYOUT = json.loads(Path(__file__).with_name("layout.json").read_text())


def distance(points):
    return sum(math.dist(a, b) for a, b in pairwise(points))


def trajectory(phases, with_heading=False):
    ends, tracks = [], []
    elapsed = 0
    for phase in phases:
        elapsed += phase["duration"]
        ends.append(elapsed)
        distances = [0.0]
        for a, b in pairwise(phase["points"]):
            distances.append(distances[-1] + math.dist(a, b))
        tracks.append(distances)

    def sample(seconds):
        index = min(bisect_left(ends, max(0, seconds)), len(phases) - 1)
        phase, distances = phases[index], tracks[index]
        progress = min(
            1, max(0, (seconds - (ends[index - 1] if index else 0)) / phase["duration"])
        )
        if phase.get("easing") == "smooth":
            progress = progress * progress * (3 - 2 * progress)
        points = phase["points"]
        left = distances[-1] * progress
        at = max(1, bisect_left(distances, left))
        heading = end_heading(phase)
        if phase.get("headings"):
            start, end = phase["headings"]
            delta = math.atan2(math.sin(end - start), math.cos(end - start))
            p = min(
                1,
                max(
                    0, (seconds - (ends[index - 1] if index else 0)) / phase["duration"]
                ),
            )
            heading = start + delta * p * p * (3 - 2 * p)
        if at >= len(points):
            point = points[-1]
        else:
            length = distances[at] - distances[at - 1]
            p = (left - distances[at - 1]) / length if length else 1
            point = [
                points[at - 1][j] + (points[at][j] - points[at - 1][j]) * p
                for j in range(2)
            ]
            heading = math.atan2(
                points[at][0] - points[at - 1][0], points[at][1] - points[at - 1][1]
            )
        if "heading" in phase:
            heading = phase["heading"]
        return (point, heading) if with_heading else point

    return sample


def position(phases, seconds):
    return trajectory(phases)(seconds)


def end_heading(phase):
    if phase.get("headings"):
        return phase["headings"][-1]
    if "heading" in phase:
        return phase["heading"]
    points = phase["points"]
    if len(points) > 1:
        a, b = points[-2:]
        return math.atan2(b[0] - a[0], b[1] - a[1])
    return phase.get("heading", 0)


def footprint(point, heading, reach=0):
    return [
        point,
        *(
            [
                point[0] + math.sin(heading) * offset,
                point[1] + math.cos(heading) * offset,
            ]
            for offset in (reach / 2, reach)
            if reach
        ),
    ]


def reservation(phases):
    sample = trajectory(phases, with_heading=True)
    ends, elapsed = [], 0
    for phase in phases:
        elapsed += phase["duration"]
        ends.append(elapsed)

    def occupied(seconds):
        phase = phases[min(bisect_left(ends, max(0, seconds)), len(phases) - 1)]
        points = footprint(*sample(seconds), phase.get("reach", 0))
        if phase.get("end_reach"):
            # The stored jack stays still while its empty-handed operator turns.
            points.extend(
                footprint(phase["points"][-1], end_heading(phase), phase["end_reach"])[
                    1:
                ]
            )
        return points

    return occupied


def turn(point, start, end):
    delta = math.atan2(math.sin(end - start), math.cos(end - start))
    return {
        "kind": "turn",
        "duration": max(0.12, abs(delta) / (2 * math.pi)),
        "points": [point],
        "headings": [start, start + delta],
    }


class Movement:
    """Route named journeys around fixtures, the car, and occupied positions."""

    def __init__(self, layout):
        self.layout = layout

    def clear_point(self, point, occupied=(), approaching=False, equipment=False):
        x, z = point
        if not (-10 <= x <= 10 and -6.8 <= z <= 6.8):
            return False
        in_corridor = (-22 if approaching else -3.65) < x < (
            4.5 if approaching else 3.65
        ) and abs(z) < (1.7 if approaching else 1.4)
        jack_contact = equipment and not approaching and abs(x) >= 3.05 and abs(z) < 0.5
        if in_corridor and not jack_contact:
            return False
        if any(
            abs(x - cx) < 1.95 and abs(z - cz) < 0.68
            for cx, cz in self.layout["stations"].values()
        ):
            return False
        return all(math.dist(point, other) >= 0.58 for other in occupied)

    def path(self, start, target, occupied=(), approaching=False, reach=0):
        clear = lambda point, equipment=False: self.clear_point(
            point, occupied, approaching, equipment
        )

        def segment(a, b):
            steps = max(1, math.ceil(math.dist(a, b) / 0.08))
            heading = math.atan2(b[0] - a[0], b[1] - a[1])
            return all(
                clear(p, equipment=k > 0)
                for i in range(steps + 1)
                for k, p in enumerate(
                    footprint(
                        [a[j] + (b[j] - a[j]) * i / steps for j in range(2)],
                        heading,
                        reach,
                    )
                )
            )

        if not clear(target):
            raise ValueError(
                "The destination is occupied or in the arriving car's path. Choose another destination."
            )
        if math.dist(start, target) < 0.01:
            return [list(start)]
        grid = 0.3
        first = tuple(round(v / grid) for v in start)
        point = lambda cell: [v * grid for v in cell]
        frontier = [(0, first)]
        costs, previous = {first: 0}, {}
        reached = None
        while frontier:
            _, cell = heapq.heappop(frontier)
            origin = start if cell == first else point(cell)
            if math.dist(origin, target) < 0.65 and segment(origin, target):
                reached = cell
                break
            for dx, dz in (
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ):
                neighbor = cell[0] + dx, cell[1] + dz
                dest = point(neighbor)
                cost = costs[cell] + math.dist(origin, dest)
                if cost >= costs.get(neighbor, math.inf) or not segment(origin, dest):
                    continue
                costs[neighbor], previous[neighbor] = cost, cell
                heapq.heappush(frontier, (cost + math.dist(dest, target), neighbor))
        if reached is None:
            raise ValueError(
                "No clear route. Choose another destination or report blocked."
            )
        nodes = [list(target)]
        while reached != first:
            nodes.append(point(reached))
            reached = previous[reached]
        nodes.append(list(start))
        nodes.reverse()
        corners = [nodes[0]]
        index = 0
        while index < len(nodes) - 1:
            end = len(nodes) - 1
            while end > index + 1 and not segment(nodes[index], nodes[end]):
                end -= 1
            corners.append(nodes[end])
            index = end
        # Rounded corners stay inside the same collision envelope as straight travel.
        points = [corners[0]]
        for i in range(1, len(corners) - 1):
            a, b, c = corners[i - 1 : i + 2]
            radius = min(0.45, math.dist(a, b) / 3, math.dist(b, c) / 3)
            for _ in range(5):
                entry = [
                    b[j] + (a[j] - b[j]) * radius / math.dist(a, b) for j in range(2)
                ]
                leave = [
                    b[j] + (c[j] - b[j]) * radius / math.dist(b, c) for j in range(2)
                ]
                curve = [
                    [
                        (1 - t) ** 2 * entry[j]
                        + 2 * t * (1 - t) * b[j]
                        + t * t * leave[j]
                        for j in range(2)
                    ]
                    for t in [n / 12 for n in range(13)]
                ]
                if all(segment(u, v) for u, v in pairwise([points[-1], *curve])):
                    points.extend(curve)
                    break
                radius /= 2
            else:
                points.append(b)
        points.append(corners[-1])
        return points

    def plan(
        self,
        start,
        target,
        pace,
        heading,
        now,
        occupied=(),
        reservations=(),
        approaching=False,
        loaded=False,
        facing=None,
        reach=0,
        end_reach=None,
    ):
        endpoints = [
            point
            for _, _, other in reservations
            for point in footprint(
                other[-1]["points"][-1],
                end_heading(other[-1]),
                other[-1].get("end_reach", other[-1].get("reach", 0)),
            )
        ]
        retreat = []
        if reach and not approaching and abs(start[1]) < 0.1 and 4 < abs(start[0]) < 5:
            clear = [start[0] + math.copysign(0.75, start[0]), start[1]]
            retreat = [
                {
                    "kind": "move",
                    "duration": 0.75 / 1.28,
                    "points": [start, clear],
                    "easing": "smooth",
                    "pace": "walk",
                    "heading": heading,
                    "reach": reach,
                }
            ]
            start = clear
        points = self.path(start, target, [*occupied, *endpoints], approaching, reach)
        if len(points) == 1:
            route = [turn(start, heading, facing if facing is not None else heading)]
        else:
            travel = {
                "kind": "move",
                "duration": distance(points)
                / ((3.2 if pace == "run" else 1.6) * (0.8 if loaded else 1)),
                "points": points,
                "easing": "smooth",
                "pace": pace,
            }
            departure = math.atan2(points[1][0] - start[0], points[1][1] - start[1])
            route = [turn(start, heading, departure), travel]
            if facing is not None:
                route.append(turn(target, end_heading(travel), facing))
        route = retreat + route
        for phase in route:
            phase["reach"] = reach
        if end_reach is not None:
            route[-1]["end_reach"] = end_reach
        if reach or end_reach:
            swept = reservation(route)
            duration = sum(p["duration"] for p in route)
            for i in range(math.ceil(duration / 0.04) + 1):
                if any(
                    not self.clear_point(
                        point, approaching=approaching, equipment=k > 0
                    )
                    for k, point in enumerate(swept(min(duration, i * 0.04)))
                ):
                    raise ValueError(
                        "The jack needs more turning room. Choose another destination."
                    )
        for n in range(151):
            phases = (
                [
                    {
                        "kind": "yield",
                        "duration": n * 0.2,
                        "points": [retreat[0]["points"][0] if retreat else start],
                        "heading": heading,
                        "reach": reach,
                    }
                ]
                if n
                else []
            ) + route
            if not self.conflicts(phases, now, occupied, reservations):
                return phases
        raise ValueError(
            "The route is busy. Choose another destination or report blocked."
        )

    def conflicts(self, phases, now, occupied=(), reservations=()):
        duration = sum(p["duration"] for p in phases)
        horizon = max(
            [
                duration,
                *(
                    began + sum(p["duration"] for p in other) - now
                    for _, began, other in reservations
                ),
            ]
        )
        here_at = reservation(phases)
        others = [(began, reservation(other)) for _, began, other in reservations]
        for step in range(math.ceil(horizon / 0.08) + 1):
            elapsed = min(horizon, step * 0.08)
            here = here_at(elapsed)
            if any(math.dist(p, q) < 0.56 for p in here for q in occupied):
                return True
            for began, other_at in others:
                there = other_at(now + elapsed - began)
                if any(math.dist(p, q) < 0.56 for p in here for q in there):
                    return True
        return False
