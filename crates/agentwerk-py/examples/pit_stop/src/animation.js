import { workerAt } from "./playback.js";
import { taskMotion, arrivalPose, workProgress } from "./motion.js";
import { poseCharacter, poseHands } from "./character-motion.js";
import { animateEquipment } from "./equipment.js";
import { CORNERS } from "./car.js";

const clamp = (value) => Math.max(0, Math.min(1, value));
const smooth = (value) => {
  const p = clamp(value);
  return p * p * (3 - 2 * p);
};
const lerp = (a, b, t) => a + (b - a) * t;

function progress(event, time) {
  return event?.data.duration
    ? clamp((time - event.t) / event.data.duration)
    : 0;
}

function vehiclePosition(carEvent, time, state) {
  if (!carEvent) return -19;
  const p = progress(carEvent, time);
  if (state.car === "departed") return 22;
  if (carEvent.name === "car_departing") return 22 * p * p;
  return 0;
}

function jackHeight(sample, end) {
  const active = sample.tasks[workerAt(sample, "jack", end)];
  if (active && ["lift", "lower"].includes(active.data.task)) {
    const p = smooth(workProgress(active, sample.time));
    return active.data.task === "lift" ? p : 1 - p;
  }
  return sample.state.jacks[end] === "up" ? 1 : 0;
}

function workerTarget(worker, task, x, z) {
  const { role, station } = worker.member;
  if (task === "collect") return [x, Math.sign(z) * 4.6];
  if (CORNERS[station]) return CORNERS[station];
  if (role === "jack") return [station === "front" ? 3 : -3, 0];
  return [role === "wing" ? 2.9 : 0, 0];
}

function animateLegacyWorker(worker, sample) {
  const { id, role, station } = worker.member;
  const current = sample.state.crew[id];
  const event = sample.tasks[id];
  const task = event?.data.task;
  const p = progress(event, sample.time);
  const goesHome = [
    "remove",
    "fit",
    "tighten",
    "adjust",
    "clear",
    "lower",
    "release",
  ].includes(task);
  let reach = current.clear ? 0 : 1;
  if (event) {
    if (["clear", "lower"].includes(task)) reach = 1 - smooth((p - 0.4) / 0.6);
    else if (task === "tighten") reach = 1 - smooth((p - 0.75) / 0.25);
    else
      reach = smooth(p / 0.25) * (goesHome ? 1 - smooth((p - 0.75) / 0.25) : 1);
  }
  const x = lerp(worker.home[0], worker.work[0], reach);
  const z = lerp(worker.home[1], worker.work[1], reach);
  worker.root.position.set(x, 0, z);
  const target = workerTarget(worker, task, x, z);
  worker.root.rotation.y = Math.atan2(target[0] - x, target[1] - z);
  const walking =
    event &&
    ((!["clear", "lower", "tighten"].includes(task) && p < 0.25) ||
      (goesHome && p > 0.75));
  const work = reach * (walking ? 0.2 : 1);
  worker.torso.rotation.x = work * (role === "steadier" ? 0.3 : 0.5);
  worker.torso.position.y = 0.57 - work * (role === "steadier" ? 0.03 : 0.12);
  for (let index = 0; index < 2; index++) {
    worker.legs[index].rotation.x = walking
      ? Math.sin(p * 32 + index * Math.PI) * 0.42
      : work * -0.15;
    worker.arms[index].rotation.x = -work * 0.35;
  }
  if (
    role === "chief" &&
    (task === "release" || current.done.includes("release"))
  )
    worker.arms[0].rotation.x = -2.4;
  worker.tool.visible = !!event;
  worker.carried.visible = false;
  if (role === "wheel-on")
    worker.carried.visible =
      !current.done.includes("fit") && (!event || p < 0.54);
  if (role === "wheel-off")
    worker.carried.visible =
      current.done.includes("remove") || (task === "remove" && p > 0.5);
  if (event && (task === "remove" || task === "fit")) {
    const [wx, wz] = CORNERS[station];
    // During the handoff, the wheel crosses the gap between hub and hands.
    const transfer =
      task === "remove"
        ? smooth((p - 0.5) / 0.22)
        : 1 - smooth((p - 0.32) / 0.22);
    worker.root.updateWorldMatrix(true, false);
    const local = worker.root.worldToLocal(
      worker.root.position.clone().set(wx, 0.49, wz),
    );
    worker.carried.position.set(
      lerp(local.x, 0, transfer),
      lerp(local.y, 0.47, transfer),
      lerp(local.z, 0.68, transfer),
    );
  } else worker.carried.position.set(0, 0.47, 0.68);
  worker.carried.rotation.y = -worker.root.rotation.y;
}

function animateWorker(worker, sample) {
  if (sample.state.items) return poseCharacter(worker, sample);
  const { id, role, station } = worker.member;
  const current = sample.state.crew[id];
  if (!current.position) return animateLegacyWorker(worker, sample);
  const event = sample.tasks[id];
  const task = event?.data.task;
  const motion = event ? taskMotion(event, sample.time) : null;
  const [x, z] = motion?.position ?? current.position;
  worker.root.position.set(x, 0, z);
  const target = workerTarget(worker, task, x, z);
  worker.root.rotation.y =
    motion?.heading ?? Math.atan2(target[0] - x, target[1] - z);
  const walking = motion?.walking;
  const working = motion?.kind === "work" && task !== "collect";
  const engaged = working || (!current.clear && !event);
  const pace = sample.metadata.motion[id].pace;
  const rhythm = sample.time * 10 * pace + worker.index * 1.7;
  const crouch =
    engaged && ["gunner", "wheel-off", "wheel-on", "wing"].includes(role)
      ? 1
      : 0;
  const bend = engaged ? (role === "steadier" ? 0.3 : 0.38) : 0;
  const carrying =
    current.equipment?.includes("tire") ||
    (task === "remove" && motion.kind === "work" && motion.work > 0.35) ||
    (task === "fit" &&
      motion.work < 0.5 &&
      motion.kind !== "pickup" &&
      motion.kind !== "wait");
  worker.torso.rotation.x = bend;
  worker.torso.position.y =
    0.57 -
    crouch * 0.16 +
    (walking
      ? Math.abs(Math.sin(rhythm)) * 0.025
      : Math.sin(rhythm * 0.15) * 0.008);
  for (let i = 0; i < 2; i++) {
    const stride = Math.sin(rhythm + i * Math.PI);
    worker.legs[i].position.y = 0.59 - crouch * 0.16;
    worker.legs[i].rotation.x = walking ? stride * 0.5 : -crouch * 0.85;
    worker.knees[i].rotation.x = walking
      ? Math.max(0, -stride) * 0.65
      : crouch * 1.25;
    worker.arms[i].rotation.x =
      engaged || carrying ? -0.65 : walking ? -stride * 0.45 : -0.06;
    worker.elbows[i].rotation.x = engaged || carrying ? -0.9 : -0.15;
    worker.arms[i].rotation.z = i ? -0.06 : 0.06;
    if (working && ["gunner", "wing", "jack"].includes(role))
      worker.elbows[i].rotation.x += Math.sin(rhythm * 1.8) * 0.08;
  }
  if (
    role === "chief" &&
    ((task === "release" && motion.work > 0) ||
      current.done.includes("release"))
  ) {
    worker.arms[0].rotation.x = -2.6;
    worker.elbows[0].rotation.x = -0.2;
  }
  worker.tool.visible = current.equipment === "tool";
  worker.carried.visible = !!carrying;
  worker.carried.position.set(0, 0.58, 0.56);
  worker.carried.rotation.y = -worker.root.rotation.y;
  if (motion?.kind === "work" && ["remove", "fit"].includes(task)) {
    const [wx, wz] = CORNERS[station];
    const p = motion.work;
    const transfer =
      task === "remove"
        ? smooth((p - 0.35) / 0.5)
        : 1 - smooth((p - 0.15) / 0.5);
    worker.carried.visible = task === "remove" ? p > 0.35 : p < 0.65;
    worker.root.updateWorldMatrix(true, false);
    const local = worker.root.worldToLocal(
      worker.root.position.clone().set(wx, 0.75, wz),
    );
    worker.carried.position.set(
      lerp(local.x, 0, transfer),
      lerp(local.y, 0.58, transfer),
      lerp(local.z, 0.56, transfer),
    );
  }
}

function animateVehicle(world, sample) {
  const { state, time, carEvent } = sample;
  const { car } = world;
  const arrival = carEvent?.name === "car_arriving";
  const pose = arrival
    ? arrivalPose(carEvent.data, progress(carEvent, time))
    : { x: vehiclePosition(carEvent, time, state), z: 0, heading: 0 };
  car.root.position.set(pose.x, 0, pose.z);
  car.root.rotation.y = pose.heading;
  const front = jackHeight(sample, "front");
  const rear = jackHeight(sample, "rear");
  car.chassis.position.y = (front + rear) * 0.13;
  car.chassis.rotation.z = (front - rear) * 0.045;
}

function animateWheels(world, sample) {
  const { state, time, tasks } = sample;
  for (const [corner, wheels] of Object.entries(world.car.wheels)) {
    if (state.items) {
      wheels.old.visible = wheels.fresh.visible = false;
      continue;
    }
    const status = state.wheels[corner];
    const removing = tasks[`wheel-off-${corner}`];
    const fitting = tasks[`wheel-on-${corner}`];
    wheels.old.visible =
      ["old-secured", "loose"].includes(status) &&
      !(removing && workProgress(removing, time) > 0.35);
    wheels.fresh.visible =
      ["fitted", "secured"].includes(status) ||
      !!(fitting && workProgress(fitting, time) > 0.65);
    for (const wheel of Object.values(wheels))
      wheel.rotation.z = -world.car.root.position.x / 0.49;
  }
}

function animateFlaps(world, sample) {
  const { state, time, tasks } = sample;
  for (const [side, flap] of Object.entries(world.car.flaps)) {
    const active = tasks[workerAt(sample, "wing", side)];
    const angle =
      active?.data.task === "adjust"
        ? smooth(workProgress(active, time)) * (active.data.value ?? 12)
        : state.wings[side];
    flap.rotation.z = (angle * Math.PI) / 180;
  }
}

function animateJacks(world, sample) {
  const { state, time, tasks } = sample;
  for (const [end, jack] of Object.entries(world.jacks)) {
    const worker = world.workers[workerAt(sample, "jack", end)];
    const event = tasks[workerAt(sample, "jack", end)];
    const isUp = state.jacks[end] === "up";
    const sign = end === "front" ? 1 : -1;
    if (sample.metadata.version >= 5) {
      jack.arm.rotation.z = sign * jackHeight(sample, end) * 0.4;
      continue;
    }
    if (sample.metadata.version >= 4) {
      jack.root.position.set(sign * 3.5, 0, 0);
    } else if (sample.state.crew[workerAt(sample, "jack", end)].position) {
      const operating =
        isUp ||
        (event &&
          workProgress(event, time) > 0 &&
          workProgress(event, time) < 1);
      if (operating) jack.root.position.set(sign * 3.5, 0, 0);
      else
        jack.root.position.set(
          worker.root.position.x - sign * 0.45,
          0,
          worker.root.position.z,
        );
    } else {
      const reach = isUp ? 1 : 0;
      jack.root.position.set(
        lerp(worker.home[0], sign * 3.5, reach),
        0,
        lerp(worker.home[1], 0, reach),
      );
    }
    jack.arm.rotation.z = sign * jackHeight(sample, end) * 0.4;
  }
}

function animateReleaseSign(world, state) {
  const released = ["released", "departing", "departed"].includes(state.car);
  const sign = world.workers.chief?.releaseSign;
  if (sign) sign.plate.material = sign.surfaces[released ? "GO" : "STOP"];
}

export function animateScene(world, sample) {
  if (!sample.state) return;
  animateVehicle(world, sample);
  animateWheels(world, sample);
  animateFlaps(world, sample);
  for (const worker of Object.values(world.workers))
    animateWorker(worker, sample);
  animateJacks(world, sample);
  animateEquipment(world, sample);
  poseHands(world, sample);
  animateReleaseSign(world, sample.state);
  world.renderer.render(world.scene, world.camera);
}
