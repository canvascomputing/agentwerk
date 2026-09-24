import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { Playback } from "../src/playback.js";

const frames = readFileSync(
  new URL("../recordings/showcase.jsonl", import.meta.url),
  "utf8",
)
  .trim()
  .split("\n")
  .map(JSON.parse);

test("streamed and loaded recordings produce the same state at every mechanical event", () => {
  const replay = new Playback(frames);
  const live = new Playback([], "live");
  for (const frame of frames) {
    live.append([frame]);
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
    (frame, index) => index > stopped && frame.name === "action_started",
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
  assert.equal(replay.sample(time + 100).state.car, "stopped");
});
