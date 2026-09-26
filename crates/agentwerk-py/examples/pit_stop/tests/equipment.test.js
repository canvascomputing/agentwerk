import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import * as THREE from "three";
import { createEquipment, animateEquipment } from "../src/equipment.js";

const layout = JSON.parse(
  readFileSync(new URL("../layout.json", import.meta.url)),
);

test("resting tools contact the cupboard top and fit within its edges", () => {
  for (const [id, slot] of Object.entries(layout.slots)) {
    if (!id.startsWith("tool-")) continue;
    const items = {
      [id]: {
        kind: id.includes("gunner") ? "gunner" : "wing",
        owner: `slot:${id}`,
      },
    };
    const scene = new THREE.Scene();
    const world = { scene, items: createEquipment(scene, items) };
    animateEquipment(world, {
      state: { items },
      metadata: { version: 3, layout },
      tasks: {},
    });
    const bounds = new THREE.Box3().setFromObject(world.items[id]);
    assert.ok(Math.abs(bounds.min.y - 0.73) < 1e-7);
    assert.ok(bounds.min.x >= -7.65 && bounds.max.x <= -4.35);
    assert.ok(
      bounds.min.z >= slot[2] - 0.345 && bounds.max.z <= slot[2] + 0.345,
    );
  }
});

test("fresh and returned tires share two platforms and rest on their tops", () => {
  for (const corner of Object.keys(layout.corners)) {
    const slot = layout.slots[`fresh-${corner}`];
    assert.deepEqual(layout.slots[`used-${corner}`], slot);
    assert.deepEqual(
      layout.crew[`wheel-off-${corner}`].station,
      layout.crew[`wheel-on-${corner}`].station,
    );
    const items = { wheel: { kind: "old", owner: `slot:used-${corner}` } };
    const scene = new THREE.Scene();
    const world = { scene, items: createEquipment(scene, items) };
    animateEquipment(world, {
      state: { items },
      metadata: { version: 3, layout },
      tasks: {},
    });
    const bounds = new THREE.Box3().setFromObject(world.items.wheel);
    assert.ok(Math.abs(bounds.min.y - 0.18) < 1e-7);
    assert.ok(bounds.min.x >= 5.05 && bounds.max.x <= 8.35);
    assert.ok(
      bounds.min.z >= slot[2] - 0.345 && bounds.max.z <= slot[2] + 0.345,
    );
  }
});

test("recorded jack geometry keeps the rolling handle ahead of the operator", async () => {
  const { ownerPose } = await import("../src/equipment.js");
  const handle = [-0.72, 0.78, 0],
    grip_forward = 0.33;
  const root = new THREE.Group();
  const world = { workers: { operator: { root } } };
  const sample = {
    metadata: { version: 6, layout: { jack: { handle, grip_forward } } },
  };
  for (const heading of [-Math.PI, -Math.PI / 2, 0, Math.PI / 2, Math.PI]) {
    root.rotation.y = heading;
    const pose = ownerPose(world, sample, "crew:operator", "jack");
    const grip = new THREE.Vector3(...handle)
      .applyAxisAngle(new THREE.Vector3(0, 1, 0), pose.heading)
      .add(pose.position);
    assert.ok(
      Math.abs(
        grip.x * Math.sin(heading) + grip.z * Math.cos(heading) - grip_forward,
      ) < 1e-8,
    );
    assert.equal(pose.position.y, 0);
  }
});
