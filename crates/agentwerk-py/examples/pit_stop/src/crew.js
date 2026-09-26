import * as THREE from "three";
import layout from "../layout.json";
import { CORNERS, tire } from "./car.js";
import { box, cylinder, material, mesh, rod } from "./geometry.js";

export const COLORS = {
  gunner: "#d8443d",
  "wheel-off": "#3d70c7",
  "wheel-on": "#deded6",
  jack: "#e4c573",
  steadier: "#333b45",
  wing: "#72a0df",
  chief: "#deded6",
};
const dark = material("#17252b");
const metal = material("#a2b1b1", { metalness: 0.8, roughness: 0.25 });

export function positions(member) {
  if (CORNERS[member.station]) {
    const [x, z] = CORNERS[member.station];
    const sign = Math.sign(z);
    const offset = { gunner: -0.88, "wheel-off": 0, "wheel-on": 0.88 }[
      member.role
    ];
    return {
      home: [x + offset, sign * 3.3],
      work: [x + offset * 0.7, sign * 1.95],
    };
  }
  if (member.role === "jack") {
    const sign = member.station === "front" ? 1 : -1;
    return { home: [sign * 4.5, -2.5], work: [sign * 3.9, 0] };
  }
  if (member.role === "chief") return { home: [0.2, -4.2], work: [0.2, -4.2] };
  const sign = member.station === "left" ? -1 : 1;
  return member.role === "wing"
    ? { home: [4, sign * 3.25], work: [3.25, sign * 1.65] }
    : { home: [0, sign * 3.3], work: [0, sign * 1.48] };
}

function clothingTexture(color, index) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 32;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, 32, 32);
  ctx.fillStyle = "#ffffff18";
  for (let y = 0; y < 32; y += 4) ctx.fillRect(0, y, 32, 1);
  ctx.fillStyle = "#10182038";
  ctx.fillRect(15, 0, 2, 32);
  ctx.fillRect(5, 10, 7, 6);
  if (index % 3 === 0) ctx.fillRect(22, 10, 7, 6);
  const texture = new THREE.CanvasTexture(canvas);
  texture.magFilter = THREE.NearestFilter;
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function createReleaseSign(torso) {
  const root = new THREE.Group();
  torso.add(root);
  root.position.set(0, 0.22, 0.28);
  const surfaces = {
    STOP: material("#ce3430", { name: "STOP", roughness: 0.85 }),
    GO: material("#238746", { name: "GO", roughness: 0.85 }),
  };
  box(root, [0.72, 0.38, 0.06], dark, [0, 0, 0]);
  const plate = box(root, [0.68, 0.34, 0.025], surfaces.STOP, [0, 0, 0.04]);
  box(
    root,
    [0.58, 0.25, 0.015],
    material("#e4e1d5", { roughness: 0.9 }),
    [0, 0, 0.06],
  );
  return { root, plate, surfaces };
}

export function createCrew(member, seed = 27, recordedLayout = layout) {
  const index = Object.keys(recordedLayout.crew).indexOf(member.id);
  const root = new THREE.Group();
  const shirts = ["#ce3430", "#2869bd", "#ebc440", "#19212a", "#e9e9df"];
  const pants = ["#173250", "#17212a", "#e2e3d9", "#235daa", "#bc2928"];
  let appearance = (seed ^ Math.imul(index + 1, 2654435761)) >>> 0;
  function pick(colors) {
    appearance = (Math.imul(appearance, 1664525) + 1013904223) >>> 0;
    return colors[appearance % colors.length];
  }
  const skins = ["#c68f68", "#82543b", "#e0b294", "#ad7450", "#684532"];
  const shirt = material("#ffffff", {
    map: clothingTexture(pick(shirts), index),
    roughness: 0.95,
    flatShading: true,
  });
  const trousers = material(pick(pants), { roughness: 1 });
  const skin = material(skins[index % skins.length], { roughness: 0.95 });
  const accent = material(COLORS[member.role]);
  const torso = new THREE.Group();
  torso.position.y = 0.57;
  root.add(torso);
  const chest = mesh(
    torso,
    new THREE.CylinderGeometry(0.22, 0.17, 0.4, 4, 1),
    shirt,
    [0, 0.22, 0],
  );
  chest.rotation.y = Math.PI / 4;
  chest.scale.z = 0.66;
  box(torso, [0.29, 0.055, 0.21], dark, [0, 0.025, 0]);
  box(torso, [0.055, 0.05, 0.025], metal, [0, 0.025, 0.12]);
  box(torso, [0.07, 0.11, 0.015], accent, [-0.095, 0.31, 0.105]);
  if (index % 4 === 0) {
    for (const sign of [-1, 1])
      box(torso, [0.075, 0.34, 0.025], accent, [sign * 0.12, 0.23, 0.11]);
  }
  cylinder(torso, 0.065, 0.09, skin, [0, 0.465, 0]);
  const head = box(torso, [0.205, 0.23, 0.205], skin, [0, 0.59, 0.018]);
  box(torso, [0.055, 0.06, 0.045], skin, [0, 0.59, 0.135]);
  const hair = material(
    ["#272623", "#513c2e", "#a88a58", "#3c3430"][index % 4],
  );
  box(torso, [0.215, 0.065, 0.22], hair, [0, 0.715, 0]);
  box(torso, [0.217, 0.13, 0.045], hair, [0, 0.66, -0.09]);
  if (index % 3 !== 1) {
    const cap = material(pick(shirts));
    box(torso, [0.23, 0.085, 0.23], cap, [0, 0.735, 0]);
    box(torso, [0.21, 0.025, 0.12], cap, [0, 0.708, 0.15]);
  }
  if (member.role === "chief") {
    for (const sign of [-1, 1])
      box(torso, [0.055, 0.105, 0.1], dark, [sign * 0.13, 0.63, 0]);
    rod(torso, [0.14, 0.61, 0], [0.1, 0.53, 0.16], 0.016, dark);
  }
  const arms = [],
    elbows = [],
    legs = [],
    knees = [],
    hands = [];
  for (const sign of [-1, 1]) {
    const arm = new THREE.Group();
    arm.position.set(sign * 0.215, 0.38, 0);
    torso.add(arm);
    box(arm, [0.115, 0.22, 0.13], shirt, [0, -0.09, 0]);
    const elbow = new THREE.Group();
    elbow.position.y = -0.205;
    arm.add(elbow);
    box(elbow, [0.09, 0.21, 0.1], index % 2 ? shirt : skin, [0, -0.09, 0]);
    const hand = box(elbow, [0.09, 0.1, 0.105], skin, [0, -0.225, 0]);
    arms.push(arm);
    elbows.push(elbow);
    hands.push(hand);
    const leg = new THREE.Group();
    leg.position.set(sign * 0.095, 0.59, 0);
    root.add(leg);
    box(leg, [0.145, 0.27, 0.17], trousers, [0, -0.12, 0]);
    const knee = new THREE.Group();
    knee.position.y = -0.26;
    leg.add(knee);
    box(knee, [0.12, 0.245, 0.145], trousers, [0, -0.115, 0]);
    box(knee, [0.15, 0.105, 0.28], dark, [0, -0.245, 0.045]);
    legs.push(leg);
    knees.push(knee);
  }
  const tool = new THREE.Group();
  hands[1].add(tool);
  if (member.role === "gunner") {
    const gun = cylinder(tool, 0.075, 0.27, metal, [0, -0.02, 0.07]);
    gun.rotation.x = Math.PI / 2;
    box(tool, [0.065, 0.16, 0.085], dark, [0, -0.06, 0]);
    rod(tool, [0, -0.02, 0.15], [0, -0.02, 0.31], 0.025, metal);
  }
  if (member.role === "wing") rod(tool, [0, 0, 0], [0, 0, 0.48], 0.024, metal);
  const carried = tire(member.role === "wheel-on");
  root.add(carried);
  carried.visible = false;
  const place = recordedLayout.crew[member.id] ?? positions(member);
  root.position.set(place.home[0], 0, place.home[1]);
  const releaseSign = member.role === "chief" ? createReleaseSign(torso) : null;
  return {
    root,
    torso,
    head,
    arms,
    elbows,
    legs,
    knees,
    hands,
    tool,
    carried,
    releaseSign,
    member,
    index,
    ...place,
  };
}

export function createJack(end, profile) {
  const root = new THREE.Group();
  const blue = material("#276fce", { metalness: 0.5, roughness: 0.4 });
  box(root, [0.78, 0.13, 0.42], blue, [0, 0.12, 0]);
  const arm = box(root, [0.62, 0.07, 0.12], metal, [0.13, 0.24, 0]);
  const handle = profile?.handle ?? [-0.72, 0.78, 0];
  root.userData.handle = handle;
  rod(root, [-0.28, 0.16, 0], handle, 0.036, metal);
  rod(
    root,
    [handle[0], handle[1], -0.18],
    [handle[0], handle[1], 0.18],
    0.038,
    dark,
  );
  for (const sign of [-1, 1]) {
    const wheel = cylinder(root, 0.12, 0.09, dark, [-0.27, 0.13, sign * 0.26]);
    wheel.rotation.x = Math.PI / 2;
  }
  root.rotation.y = end === "front" ? Math.PI : 0;
  return { root, arm };
}
