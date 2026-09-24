const clamp = (value) => Math.max(0, Math.min(1, value));

export function arrivalPose(data, progress) {
  const p = clamp(progress);
  const remaining = 1 - p;
  const braking = data.braking ?? 2;
  const offset = data.offset ?? 0;
  // Old recordings keep their original straight approach.
  if (data.offset === undefined)
    return { x: -19 * remaining ** braking, z: 0, heading: 0 };
  // Zero velocity and acceleration at the marks avoid a snap into service.
  const distance = 1 - remaining ** 3 * (1 + (3 - braking) * p);
  const rest = 1 - distance;
  const bend = (offset < 0 ? -1 : 1) * 0.38;
  const z =
    offset * rest ** 2 * (1 + 2 * distance) +
    bend * 16 * distance ** 2 * rest ** 2;
  const slope =
    -6 * offset * distance * rest +
    32 * bend * distance * rest * (1 - 2 * distance);
  return {
    x: -19 * rest,
    z,
    heading: -Math.atan2(slope, 19),
  };
}

export function routePosition(points, progress) {
  const lengths = points
    .slice(1)
    .map((point, index) =>
      Math.hypot(point[0] - points[index][0], point[1] - points[index][1]),
    );
  let remaining = lengths.reduce((total, length) => total + length, 0);
  remaining *= clamp(progress);
  for (let index = 0; index < lengths.length; index++) {
    if (remaining <= lengths[index] && lengths[index]) {
      const start = points[index];
      const end = points[index + 1];
      const fraction = remaining / lengths[index];
      return {
        position: [
          start[0] + (end[0] - start[0]) * fraction,
          start[1] + (end[1] - start[1]) * fraction,
        ],
        heading: Math.atan2(end[0] - start[0], end[1] - start[1]),
      };
    }
    remaining -= lengths[index];
  }
  return { position: points.at(-1), heading: null };
}

export function actionMotion(event, time) {
  const phaseIndex = event.phaseEvent?.data.phase;
  let elapsed = Math.max(0, time - (event.phaseEvent?.t ?? event.t));
  let work = 0;
  let previousPhase = null;
  for (const [index, phase] of event.data.phases.entries()) {
    if (
      phaseIndex === undefined
        ? elapsed <= phase.duration
        : index === phaseIndex
    ) {
      const progress = clamp(elapsed / phase.duration);
      const pose = routePosition(phase.points, progress);
      if (phase.headings) {
        const [from, to] = phase.headings;
        const delta = Math.atan2(Math.sin(to - from), Math.cos(to - from));
        pose.heading = from + delta * progress * progress * (3 - 2 * progress);
      } else if (phase.heading !== undefined) pose.heading = phase.heading;
      return {
        ...pose,
        previousPhase,
        kind: phase.kind,
        phase,
        progress,
        work: phase.kind === "work" || phase.effect ? progress : work,
        walking: phase.points.length > 1,
      };
    }
    if (phase.kind === "work" || phase.effect) work = 1;
    if (phaseIndex === undefined) elapsed -= phase.duration;
    previousPhase = phase;
  }
  return {
    position: event.data.phases.at(-1).points.at(-1),
    heading: null,
    kind: "done",
    phase: null,
    progress: 1,
    work: 1,
    walking: false,
  };
}

export function workProgress(event, time) {
  if (!event) return 0;
  if (event.data.phases) return actionMotion(event, time).work;
  return clamp(((time - event.t) / event.data.duration - 0.25) / 0.5);
}
