import * as THREE from "three";
import { CORNERS } from "./car.js";
import { actionMotion } from "./motion.js";
import { equipmentPose } from "./equipment.js";

const smooth = (p) => p * p * (3 - 2 * p);

export function poseCharacter(worker, sample) {
  const { id, role, station } = worker.member;
  const current = sample.state.crew[id];
  const event = sample.actions[id];
  const action = event?.data.action;
  const motion = event ? actionMotion(event, sample.time) : null;
  const [x, z] = motion?.position ?? current.position;
  worker.root.position.set(x, 0, z);
  let target = CORNERS[station] ?? [role === "wing" ? 3 : 0, 0];
  if (["collect", "return", "stow"].includes(action)) {
    const slot =
      role === "wheel-on"
        ? `fresh-${station}`
        : role === "wheel-off"
          ? `used-${station}`
          : `tool-${id}`;
    const point = sample.metadata.layout.slots[slot];
    target = [point[0], point[2]];
  }
  worker.root.rotation.y =
    motion?.heading ??
    current.heading ??
    Math.atan2(target[0] - x, target[1] - z);
  const kind = motion?.kind;
  const walking = !!motion?.walking;
  const handles = (kind) =>
    [
      "reach",
      "grip",
      "lift_item",
      "align",
      "pull",
      "seat",
      "place",
      "release_item",
      "work",
    ].includes(kind);
  const holdingPosition = !event && !current.clear;
  const previous = handles(motion?.previousPhase?.kind) ? 1 : 0;
  const next = handles(kind) || holdingPosition ? 1 : 0;
  const transition = motion ? smooth(Math.min(1, motion.progress * 5)) : 1;
  const engaged = previous + (next - previous) * transition;
  const carrying = !!current.equipment || role === "chief";
  const rushing = role === "chief" && action === "release" && walking;
  const pace = sample.metadata.motion[id].pace;
  const rhythm =
    sample.time * (rushing ? 16 : carrying ? 8 : 10) * pace +
    worker.index * 1.7;
  const bend = engaged * 0.33 * (motion?.phase?.pose ?? 1);
  const crouch = role !== "chief" && role !== "steadier" ? engaged : 0;
  const ramp = motion ? smooth(Math.min(1, motion.progress * 5)) : 1;
  const reach = role === "chief" ? 0 : bend * (kind === "reach" ? ramp : 1);
  worker.torso.rotation.x = reach;
  worker.torso.position.set(
    0,
    0.57 - crouch * 0.13 + (walking ? Math.abs(Math.sin(rhythm)) * 0.022 : 0),
    engaged * (role === "wheel-on" || role === "wheel-off" ? 0.04 : 0.14),
  );
  worker.head.rotation.y = walking ? Math.sin(rhythm * 0.15) * 0.08 : 0;
  for (let i = 0; i < 2; i++) {
    const stride = Math.sin(rhythm + i * Math.PI);
    worker.legs[i].position.y = 0.59 - crouch * 0.13;
    worker.legs[i].rotation.x = walking
      ? stride * (rushing ? 0.6 : carrying ? 0.35 : 0.5)
      : -crouch * 0.75;
    worker.knees[i].rotation.x = walking
      ? Math.max(0, -stride) * 0.65
      : crouch * 1.15;
    worker.arms[i].rotation.set(
      walking ? -stride * 0.45 : -0.06,
      0,
      i ? -0.06 : 0.06,
    );
    worker.elbows[i].rotation.set(engaged || carrying ? -0.9 : -0.15, 0, 0);
  }
  worker.tool.visible = false;
  worker.carried.visible = false;
}

function aimHand(worker, index, target) {
  const arm = worker.arms[index];
  const elbow = worker.elbows[index];
  worker.root.updateWorldMatrix(true, true);
  const shoulder = arm.getWorldPosition(new THREE.Vector3());
  const direction = target.clone().sub(shoulder);
  const length = Math.max(0.02, Math.min(0.428, direction.length()));
  direction.normalize();
  const pole = new THREE.Vector3(0, 0, 1).applyQuaternion(
    worker.root.quaternion,
  );
  pole.addScaledVector(direction, -pole.dot(direction)).normalize();
  const along = (0.205 ** 2 - 0.225 ** 2 + length ** 2) / (2 * length);
  const height = Math.sqrt(Math.max(0, 0.205 ** 2 - along ** 2));
  const joint = shoulder
    .clone()
    .addScaledVector(direction, along)
    .addScaledVector(pole, height);
  const end = shoulder.clone().addScaledVector(direction, length);
  const down = new THREE.Vector3(0, -1, 0);
  const parent = arm.parent.getWorldQuaternion(new THREE.Quaternion()).invert();
  arm.quaternion.setFromUnitVectors(
    down,
    joint.clone().sub(shoulder).normalize().applyQuaternion(parent),
  );
  worker.root.updateWorldMatrix(true, true);
  const inverse = arm.getWorldQuaternion(new THREE.Quaternion()).invert();
  elbow.quaternion.setFromUnitVectors(
    down,
    end.sub(joint).normalize().applyQuaternion(inverse),
  );
}

export function poseHands(world, sample) {
  for (const [id, worker] of Object.entries(world.workers)) {
    const event = sample.actions[id];
    const motion = event?.data.phases ? actionMotion(event, sample.time) : null;
    const current = sample.state.crew[id];
    const action = event?.data.action;
    if (worker.releaseSign) {
      const raised =
        action !== "release"
          ? 0
          : motion?.kind === "work"
            ? smooth(motion.progress)
            : motion?.kind === "turn"
              ? 1
              : motion?.kind === "withdraw"
                ? 1 - smooth(motion.progress)
                : 0;
      const board = worker.releaseSign.root;
      board.position.set(0, 0.22 + raised * 0.36, 0.28);
      for (const i of [0, 1])
        aimHand(
          worker,
          i,
          board.localToWorld(new THREE.Vector3(i ? 0.35 : -0.35, 0, 0.015)),
        );
      continue;
    }
    if (!sample.state.items) continue;
    let item = current.equipment ?? motion?.phase?.transfer?.item;
    if (!item && motion?.kind === "release_item")
      item =
        action === "fit"
          ? `fresh-${worker.member.station}`
          : action === "stow"
            ? `old-${worker.member.station}`
            : `tool-${id}`;
    if (!item && ["reach", "align"].includes(motion?.kind))
      item =
        action === "remove"
          ? `old-${worker.member.station}`
          : worker.member.role === "wheel-on"
            ? `fresh-${worker.member.station}`
            : `tool-${id}`;
    if (item && sample.state.items[item]) {
      const pose = equipmentPose(world, sample, item);
      const isTire = ["fresh", "old"].includes(sample.state.items[item].kind);
      for (const i of isTire ? [0, 1] : [1]) {
        const offset = new THREE.Vector3(
          isTire ? (i ? 0.3 : -0.3) : 0,
          isTire ? 0.28 : -0.07,
          isTire ? -0.21 : 0,
        );
        offset.applyAxisAngle(
          new THREE.Vector3(0, 1, 0),
          isTire ? worker.root.rotation.y : pose.heading,
        );
        let target = pose.position.clone().add(offset);
        if (["reach", "release_item"].includes(motion?.kind)) {
          const rest = worker.hands[i].getWorldPosition(new THREE.Vector3());
          target = rest.lerp(
            target,
            motion.kind === "reach"
              ? smooth(motion.progress)
              : 1 - smooth(motion.progress),
          );
        }
        aimHand(worker, i, target);
      }
    } else if (worker.member.role === "steadier" && !current.clear) {
      const side = worker.member.station === "left" ? -1 : 1;
      for (const i of [0, 1])
        aimHand(
          worker,
          i,
          world.car.chassis.localToWorld(
            new THREE.Vector3(i ? 0.2 : -0.2, 0.65, side * 0.9),
          ),
        );
    } else if (worker.member.role === "jack") {
      const jack = world.jacks[worker.member.station];
      for (const i of [0, 1])
        aimHand(
          worker,
          i,
          jack.root.localToWorld(
            new THREE.Vector3(-0.72, 0.78, i ? 0.18 : -0.18),
          ),
        );
    }
  }
}
