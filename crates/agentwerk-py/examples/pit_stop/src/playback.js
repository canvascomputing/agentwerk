/** Recorded simulation time drives live observation and repeatable replay. */
export const REPLAY_SECONDS = 12;

export function normalizeFrame(frame) {
  const names = {
    action_started: "crew_task_started",
    action_phase: "crew_task_phase",
    action_phase_completed: "crew_task_phase_completed",
    action_completed: "crew_task_completed",
    action_transfer: "crew_transfer",
    action_effect: "crew_work",
  };
  const data = { ...frame.data };
  if (data.action !== undefined) data.task = data.action;
  if (data.work !== undefined) data.task = data.work;
  return { ...frame, name: names[frame.name] ?? frame.name, data };
}

export function workerAt(sample, role, target) {
  return sample.metadata.crew.find(
    (member) =>
      member.role === role &&
      (sample.state.crew[member.id].station ?? member.station) === target,
  )?.id;
}

export const MILESTONES = new Set([
  "car_approaching",
  "car_arriving",
  "car_stopped",
  "pit_prepared",
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
]);

export function phaseTitle(sample) {
  // Recordings before titles were set through werk.on_event show the milestone.
  return sample.title ?? sample.milestone ?? "car_approaching";
}

export function phaseCompleted(sample) {
  const times = sample.milestones;
  return {
    preparation: times.car_stopped !== undefined,
    service: times.pit_service_completed !== undefined,
    clearance: times.pit_released !== undefined,
    total: times.pit_released !== undefined,
  };
}

export function phaseTimers(sample) {
  const times = sample.milestones;
  const end = times.pit_released ?? sample.time;
  const duration = (start, stop) =>
    start === undefined ? 0 : Math.max(0, (stop ?? end) - start);
  return {
    preparation: duration(times.car_approaching ?? 0, times.car_stopped),
    service: duration(times.car_stopped, times.pit_service_completed),
    clearance: duration(times.pit_service_completed, times.pit_released),
    total: duration(times.car_approaching ?? 0, times.pit_released),
  };
}

export class Playback {
  constructor(frames = [], mode = "replay") {
    this.frames = [];
    this.mode = mode;
    this.time = 0;
    this.paused = false;
    this.clockRunning = false;
    this.clockUntil = Infinity;
    this.append(frames);
  }

  append(frames) {
    for (const raw of frames) {
      const frame = normalizeFrame(raw);
      if (frame.n < this.frames.length) continue;
      if (frame.n !== this.frames.length)
        throw new Error("Event stream has a gap; reload to recover.");
      if (frame.t < (this.frames.at(-1)?.t ?? 0))
        throw new Error("Event time moved backwards.");
      this.frames.push(frame);
      if (this.mode === "live" && this.frames[0]?.data.version >= 4) {
        if (!this.paused) this.time = frame.t;
        if (frame.name === "pit_clock") {
          this.clockRunning = frame.data.running;
          this.clockUntil = frame.data.until ?? Infinity;
        }
      }
    }
  }

  get duration() {
    return (this.frames.at(-1)?.t ?? 0) + 3;
  }
  get speed() {
    return this.mode === "live" ? 1 : this.duration / REPLAY_SECONDS;
  }

  tick(seconds) {
    if (this.paused) return;
    if (
      this.mode === "live" &&
      this.frames[0]?.data.version >= 4 &&
      !this.clockRunning
    )
      return;
    this.time += seconds * this.speed;
    if (this.mode === "live" && this.frames[0]?.data.version >= 4)
      this.time = Math.min(this.time, this.clockUntil);
    if (this.mode === "replay") this.time %= this.duration;
  }

  reset() {
    this.time = 0;
  }

  sample(time = this.time) {
    let metadata = this.frames[0]?.data ?? null;
    let state = metadata?.state ?? null;
    let carEvent = null,
      heldAt = null,
      latest = null,
      milestone = null,
      title = null;
    const tasks = {},
      completed = {},
      milestones = {};
    for (const frame of this.frames) {
      if (frame.t > time) break;
      const { name, data } = frame;
      if (data.state) state = data.state;
      if (name === "run_metadata") metadata = data;
      if (
        [
          "car_arriving",
          "car_stopped",
          "car_departing",
          "car_departed",
        ].includes(name)
      )
        carEvent = frame;
      if (name === "pit_held") heldAt = frame.t;
      if (MILESTONES.has(name)) milestone = name;
      if (name === "pit_title") title = data.title;
      if (name.startsWith("pit_") || name.startsWith("car_"))
        milestones[name] ??= frame.t;
      if (name === "crew_task_started") tasks[data.actor] = frame;
      if (name === "crew_task_phase" && tasks[data.actor])
        tasks[data.actor] = { ...tasks[data.actor], phaseEvent: frame };
      if (name === "crew_task_completed") {
        completed[`${data.actor}:${data.task}`] = frame;
        delete tasks[data.actor];
        if (data.task === "release") milestones.pit_released ??= frame.t;
      }
      if (
        (metadata?.version ?? 1) < 4 &&
        data.state &&
        Object.values(state.wheels).every((value) => value === "secured") &&
        Object.values(state.wings).every((value) => value === 12)
      ) {
        milestones.pit_service_completed ??= frame.t;
      }
      if (name.startsWith("crew_") || name.startsWith("car_")) latest = frame;
    }
    return {
      state,
      metadata,
      carEvent,
      tasks,
      completed,
      milestones,
      milestone,
      title,
      latest,
      time: heldAt ?? time,
    };
  }
}
