/** The same ordered event history drives live observation and repeatable replay. */
export const REPLAY_SECONDS = 12;

export class Playback {
  constructor(frames = [], mode = "replay") {
    this.frames = [];
    this.mode = mode;
    this.time = 0;
    this.paused = false;
    this.append(frames);
  }

  append(frames) {
    for (const frame of frames) {
      if (frame.n < this.frames.length) continue;
      if (frame.n !== this.frames.length)
        throw new Error("Event stream has a gap; reload to recover.");
      if (frame.t < (this.frames.at(-1)?.t ?? 0))
        throw new Error("Event time moved backwards.");
      this.frames.push(frame);
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
    this.time += seconds * this.speed;
    if (this.mode === "replay") this.time %= this.duration;
  }

  reset() {
    this.time = 0;
  }

  sample(time = this.time) {
    let metadata = this.frames[0]?.data ?? null;
    let state = metadata?.state ?? null;
    let carEvent = null;
    let heldAt = null;
    const actions = {};
    const completed = {};
    let latest = null;
    const activeTasks = new Map();
    for (const frame of this.frames) {
      if (frame.t > time) break;
      const { name, data } = frame;
      if (name === "task_started") activeTasks.set(data.task, data.actor);
      if (["task_finished", "task_failed"].includes(name))
        activeTasks.delete(data.task);
      if (name === "run_finished") activeTasks.clear();
      if (data.state) state = data.state;
      if (name === "run_metadata") metadata = data;
      if (name.startsWith("car_")) carEvent = frame;
      if (name === "pit_held") heldAt = frame.t;
      if (name === "action_started") actions[data.actor] = frame;
      if (name === "action_phase" && actions[data.actor])
        actions[data.actor] = { ...actions[data.actor], phaseEvent: frame };
      if (name === "action_completed") {
        completed[`${data.actor}:${data.action}`] = frame;
        delete actions[data.actor];
      }
      if (name.startsWith("action_") || name.startsWith("car_")) latest = frame;
    }
    return {
      state,
      metadata,
      carEvent,
      actions,
      completed,
      activeAgents: [...new Set(activeTasks.values())].filter(Boolean),
      latest,
      time: heldAt ?? time,
    };
  }
}
