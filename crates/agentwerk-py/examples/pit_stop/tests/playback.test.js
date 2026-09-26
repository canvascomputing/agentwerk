import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { Playback, normalizeFrame, phaseTimers } from "../src/playback.js";

const frames = readFileSync(
  new URL("../recordings/showcase.jsonl", import.meta.url),
  "utf8",
)
  .trim()
  .split("\n")
  .map(JSON.parse)
  .map(normalizeFrame);

test("streamed and loaded recordings produce the same state at every mechanical event", () => {
  const replay = new Playback(frames);
  const live = new Playback([], "live");
  for (let i = 0; i < frames.length; i++) {
    const frame = frames[i];
    live.append([frame]);
    if (frames[i + 1]?.t !== frame.t)
      assert.deepEqual(live.sample(frame.t), replay.sample(frame.t));
  }
});

test("reconnecting deduplicates events without losing subsequent events", () => {
  const replay = new Playback(frames.slice(0, 5));
  replay.append(frames.slice(2));
  assert.equal(replay.frames.length, frames.length);
  assert.equal(replay.sample(replay.duration).state.car, "departed");
});

test("pause freezes time and reset restores the arrival state", () => {
  const replay = new Playback(frames);
  replay.tick(8);
  replay.paused = true;
  const time = replay.time;
  replay.tick(8);
  assert.equal(replay.time, time);
  replay.reset();
  assert.equal(replay.time, 0);
  assert.equal(replay.sample(frames[0].t).state.car, "approaching");
});

test("replay loops while live viewing never restarts agents or rewinds", () => {
  const replay = new Playback(frames);
  replay.tick(23);
  assert.ok(replay.time < replay.duration);
  const live = new Playback(frames, "live");
  live.tick(live.duration + 100);
  assert.equal(live.sample().state.car, "departed");
});

test("an event gap is reported rather than silently skipping car state", () => {
  const replay = new Playback();
  assert.throws(() => replay.append([frames[1]]), /gap/);
});

test("a failure freezes active mechanical motion before it can look completed", () => {
  const stopped = frames.findIndex((frame) => frame.name === "car_stopped");
  const start = frames.findIndex(
    (frame, index) => index > stopped && frame.name === "crew_task_started",
  );
  const history = frames.slice(0, start + 1);
  const time = history.at(-1).t + 0.1;
  history.push({
    n: history.length,
    t: time,
    name: "pit_held",
    data: {
      state: { ...history.at(-1).data.state, held: "Equipment failure" },
    },
  });
  const replay = new Playback(history);
  assert.deepEqual(replay.sample(time), replay.sample(time + 100));
  assert.deepEqual(
    phaseTimers(replay.sample(time)),
    phaseTimers(replay.sample(time + 100)),
  );
  assert.equal(replay.sample(time + 100).state.car, "stopped");
});

test("phase timers follow simulation milestones and freeze at release", () => {
  const sample = {
    time: 50,
    milestones: {
      car_approaching: 0,
      car_stopped: 10,
      pit_service_completed: 30,
      pit_released: 40,
    },
  };
  assert.deepEqual(phaseTimers(sample), {
    preparation: 10,
    service: 20,
    clearance: 10,
    total: 40,
  });
  assert.deepEqual(phaseTimers({ ...sample, time: 100 }), phaseTimers(sample));
  assert.deepEqual(
    phaseTimers({ time: 5, milestones: { car_approaching: 0 } }),
    { preparation: 5, service: 0, clearance: 0, total: 5 },
  );
});

test("live clock follows server pauses while replay omits decision latency", () => {
  const frames = [
    { n: 0, t: 0, name: "run_metadata", data: { version: 4 } },
    { n: 1, t: 2, name: "pit_clock", data: { running: false } },
  ];
  const live = new Playback(frames, "live");
  live.tick(10);
  assert.equal(live.time, 2);
  live.append([
    { n: 2, t: 2, name: "pit_clock", data: { running: true, until: 3 } },
  ]);
  live.tick(1);
  assert.equal(live.time, 3);
  live.tick(10);
  assert.equal(live.time, 3);
  live.append([{ n: 3, t: 2.5, name: "pit_clock", data: { running: false } }]);
  assert.equal(live.time, 2.5);
  const replay = new Playback(live.frames);
  replay.tick(1);
  assert.ok(replay.time > 0);
});

test("the heading follows recorded titles, ignoring clock and task events", async () => {
  const { phaseTitle } = await import("../src/playback.js");
  const replay = new Playback(frames);
  let last;
  for (const frame of frames) {
    if (frame.name === "pit_title") last = frame.data.title;
    const next = frames[frame.n + 1];
    if (last && next?.t !== frame.t)
      assert.equal(phaseTitle(replay.sample(frame.t)), last);
  }
});

test("the heading shows the title werk.on_event set", async () => {
  const { phaseTitle } = await import("../src/playback.js");
  const replay = new Playback([
    { n: 0, t: 0, name: "run_metadata", data: { version: 6 } },
    { n: 1, t: 1, name: "car_lifted", data: {} },
    { n: 2, t: 1, name: "pit_title", data: { title: "car_lifted" } },
    { n: 3, t: 2, name: "car_unbraced", data: {} },
  ]);
  assert.equal(phaseTitle(replay.sample(1)), "car_lifted");
  assert.equal(phaseTitle(replay.sample(2)), "car_lifted");
});

test("completion colors survive seeking and HOLD only preserves finished phases", async () => {
  const { phaseCompleted } = await import("../src/playback.js");
  const replay = new Playback(frames);
  const end = phaseCompleted(replay.sample(replay.duration));
  assert.deepEqual(end, {
    preparation: true,
    service: true,
    clearance: true,
    total: true,
  });
  assert.deepEqual(phaseCompleted(replay.sample(0)), {
    preparation: false,
    service: false,
    clearance: false,
    total: false,
  });
  const service = frames.find(
    (frame) => frame.name === "pit_service_completed",
  );
  const held = new Playback([
    ...frames.slice(0, service.n + 1),
    { n: service.n + 1, t: service.t, name: "pit_held", data: {} },
  ]);
  assert.deepEqual(phaseCompleted(held.sample(service.t + 100)), {
    preparation: true,
    service: true,
    clearance: false,
    total: false,
  });
});
