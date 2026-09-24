import * as THREE from "three";
import { box, cylinder, material, rod } from "./geometry.js";
import { CORNERS, tire } from "./car.js";
import { actionMotion } from "./motion.js";

const smooth = (p) => p * p * (3 - 2 * p);

export function createEquipment(scene, items) {
  return Object.fromEntries(
    Object.entries(items).map(([id, item]) => {
      const object =
        item.kind === "fresh" || item.kind === "old"
          ? tire(item.kind === "fresh")
          : createTool(item.kind);
      scene.add(object);
      return [id, object];
    }),
  );
}

function createTool(kind) {
  const tool = new THREE.Group();
  const steel = material("#b8c7c5", { metalness: 0.75, roughness: 0.3 });
  const blue = material("#246fde", { metalness: 0.3, roughness: 0.45 });
  const grip = material("#123667");
  if (kind === "gunner") {
    const barrel = cylinder(tool, 0.085, 0.28, blue, [0, 0, 0.06]);
    barrel.rotation.x = Math.PI / 2;
    box(tool, [0.085, 0.19, 0.09], grip, [0, -0.09, 0]);
    rod(tool, [0, 0, 0.18], [0, 0, 0.36], 0.03, steel);
  } else {
    rod(tool, [0, 0, -0.12], [0, 0, 0.38], 0.035, blue);
    box(tool, [0.12, 0.055, 0.08], grip, [0, 0, 0.38]);
  }
  return tool;
}

export function ownerPose(world, sample, owner, kind) {
  const [type, id] = owner.split(":");
  if (type === "slot")
    return {
      position: new THREE.Vector3(...sample.metadata.layout.slots[id]),
      heading: 0,
    };
  if (type === "hub") {
    const [x, z] = CORNERS[id];
    return {
      position: world.car.chassis.localToWorld(new THREE.Vector3(x, 0.49, z)),
      heading: world.car.root.rotation.y,
    };
  }
  const worker = world.workers[id];
  const isTire = kind === "fresh" || kind === "old";
  const pose = {
    position: worker.root.localToWorld(
      isTire
        ? new THREE.Vector3(0, 0.61, 0.6)
        : new THREE.Vector3(0.17, 0.8, 0.36),
    ),
    heading: worker.root.rotation.y,
  };
  const event = sample.actions[id];
  const motion = event ? actionMotion(event, sample.time) : null;
  if (!isTire && motion?.kind === "work") {
    const corner = CORNERS[worker.member.station];
    const goal = corner
      ? world.car.chassis.localToWorld(
          new THREE.Vector3(corner[0], 0.49, corner[1]),
        )
      : world.car.chassis.localToWorld(
          new THREE.Vector3(
            3.1,
            0.5,
            worker.member.station === "left" ? -0.75 : 0.75,
          ),
        );
    const toward = goal.clone().sub(pose.position).normalize();
    const contact = goal.clone().addScaledVector(toward, -0.35);
    const blend = smooth(
      Math.min(1, motion.progress * 5, (1 - motion.progress) * 5),
    );
    pose.position.lerp(contact, blend);
    const targetHeading = Math.atan2(toward.x, toward.z);
    const delta = Math.atan2(
      Math.sin(targetHeading - pose.heading),
      Math.cos(targetHeading - pose.heading),
    );
    pose.heading += delta * blend;
  }
  return pose;
}

export function equipmentPose(world, sample, id) {
  const item = sample.state.items[id];
  const isTire = item.kind === "fresh" || item.kind === "old";
  const pose = ownerPose(world, sample, item.owner, item.kind);
  for (const event of Object.values(sample.actions)) {
    const motion = actionMotion(event, sample.time);
    const transfer = motion.phase?.transfer;
    if (transfer?.item !== id) continue;
    const from = ownerPose(world, sample, transfer.from, item.kind);
    const to = ownerPose(world, sample, transfer.to, item.kind);
    const p = smooth(motion.progress);
    pose.position.copy(from.position).lerp(to.position, p);
    pose.position.y += Math.sin(Math.PI * p) * (isTire ? 0.025 : 0.12);
    // A tire's two faces are interchangeable; avoid sweeping it through the carrier with a half-turn.
    const symmetry = isTire ? 2 : 1;
    const angle = (to.heading - from.heading) * symmetry;
    const turn = Math.atan2(Math.sin(angle), Math.cos(angle)) / symmetry;
    pose.heading = from.heading + turn * p;
    pose.transfer = transfer;
  }
  return pose;
}

export function animateEquipment(world, sample) {
  if (!sample.state.items) return;
  world.scene.updateMatrixWorld(true);
  for (const [id, object] of Object.entries(world.items)) {
    const item = sample.state.items[id];
    const pose = equipmentPose(world, sample, id);
    object.position.copy(pose.position);
    object.rotation.set(0, pose.heading, 0);
    if (item.owner.startsWith("hub:"))
      object.rotation.z = -world.car.root.position.x / 0.49;
  }
}
