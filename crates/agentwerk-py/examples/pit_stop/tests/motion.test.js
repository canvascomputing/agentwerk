import assert from "node:assert/strict";
import test from "node:test";
import { actionMotion, arrivalPose, workProgress } from "../src/motion.js";

test("curved arrivals straighten and brake smoothly onto the same marks", () => {
  for (const offset of [-0.45, 0, 0.45]) {
    for (const braking of [1.8, 2.5]) {
      const data = { offset, braking };
      const poses = Array.from({ length: 101 }, (_, i) =>
        arrivalPose(data, i / 100),
      );
      assert.equal(poses[0].x, -19);
      assert.equal(poses[0].z, offset);
      assert.ok(Math.abs(poses.at(-1).x) < 1e-10);
      assert.ok(Math.abs(poses.at(-1).z) < 1e-10);
      assert.ok(Math.abs(poses.at(-1).heading) < 1e-10);
      assert.ok(poses.some((pose) => Math.abs(pose.heading) > 0.01));
      for (let i = 1; i < poses.length; i++) {
        assert.ok(poses[i].x > poses[i - 1].x);
        assert.ok(Math.abs(poses[i].z) < 0.8);
      }
      const step = 0.0001;
      const end = arrivalPose(data, 1).x;
      const before = arrivalPose(data, 1 - step).x;
      const earlier = arrivalPose(data, 1 - 2 * step).x;
      assert.ok(Math.abs((end - before) / step) < 0.001);
      assert.ok(Math.abs((end - 2 * before + earlier) / step ** 2) < 0.1);
      const middle = arrivalPose(data, 0.4);
      const next = arrivalPose(data, 0.4 + step);
      const tangent = -Math.atan2(next.z - middle.z, next.x - middle.x);
      assert.ok(Math.abs(middle.heading - tangent) < 0.001);
    }
  }
});

test("legacy recordings retain their straight arrival", () => {
  const pose = arrivalPose({}, 0.5);
  assert.deepEqual(pose, { x: -4.75, z: 0, heading: 0 });
});

const event = {
  t: 10,
  data: {
    phases: [
      { kind: "wait", duration: 1, points: [[0, 3]] },
      {
        kind: "move",
        duration: 2,
        points: [
          [0, 3],
          [2, 3],
          [2, 1],
        ],
      },
      { kind: "work", duration: 2, points: [[2, 1]] },
      {
        kind: "withdraw",
        duration: 1,
        points: [
          [2, 1],
          [2, 4],
        ],
      },
    ],
  },
};

test("mechanical motion waits for arrival at work, then persists through withdrawal", () => {
  assert.equal(workProgress(event, 12), 0);
  assert.equal(workProgress(event, 14), 0.5);
  assert.equal(workProgress(event, 15.5), 1);
  assert.deepEqual(actionMotion(event, 16).position, [2, 4]);
});

test("recorded paths drive heading and movement independently of frame rate", () => {
  assert.deepEqual(actionMotion(event, 11.5).position, [1, 3]);
  assert.equal(actionMotion(event, 11.5).heading, Math.PI / 2);
  assert.deepEqual(actionMotion(event, 12.5).position, [2, 2]);
  assert.equal(actionMotion(event, 12.5).heading, Math.PI);
  assert.equal(actionMotion(event, 14).walking, false);
});

test("observed phases hold their endpoint until the next event, despite clock drift", () => {
  const observed = { ...event, phaseEvent: { t: 12, data: { phase: 1 } } };
  assert.deepEqual(actionMotion(observed, 12).position, [0, 3]);
  assert.deepEqual(actionMotion(observed, 14.2).position, [2, 1]);
  assert.equal(actionMotion(observed, 14.2).kind, "move");
  assert.equal(workProgress(observed, 14.2), 0);
});
