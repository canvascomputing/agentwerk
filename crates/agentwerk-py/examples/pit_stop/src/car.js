import layout from "../layout.json";
import * as THREE from "three";
import { box, cylinder, decal, material, mesh, rod } from "./geometry.js";

export const CORNERS = layout.corners;

const carbon = material("#182326", { roughness: 0.4, metalness: 0.3 });
const silver = material("#b9c6c3", { roughness: 0.25, metalness: 0.75 });
const rubber = material("#111819", { roughness: 0.93 });
const ivory = material("#eeeae2", { roughness: 0.28, metalness: 0.35 });
const red = material("#d52932", { roughness: 0.35, metalness: 0.3 });

const blue = material("#1858af", { roughness: 0.35, metalness: 0.3 });
const yellow = material("#ffd333", { roughness: 0.4 });

export function tire(fresh = false) {
  const group = new THREE.Group();
  const tread = cylinder(group, 0.49, 0.39, rubber);
  tread.rotation.x = Math.PI / 2;
  for (const side of [-1, 1]) {
    const rim = cylinder(group, 0.27, 0.025, carbon, [0, 0, side * 0.204]);
    rim.rotation.x = Math.PI / 2;
    const hub = cylinder(group, 0.075, 0.035, silver, [0, 0, side * 0.223]);
    hub.rotation.x = Math.PI / 2;
    const marking = material(fresh ? "#b8d691" : "#d8ae68");
    mesh(group, new THREE.TorusGeometry(0.375, 0.012, 6, 48), marking, [
      0,
      0,
      side * 0.208,
    ]);
    for (let i = 0; i < 10; i++) {
      const angle = (i * Math.PI) / 5;
      const spoke = box(group, [0.025, 0.4, 0.012], silver, [
        0,
        0,
        side * 0.22,
      ]);
      spoke.rotation.z = angle;
    }
  }
  return group;
}

function body(parent, outline, depth, surface, y) {
  const shape = new THREE.Shape();
  shape.moveTo(outline[0][0], -outline[0][1]);
  for (const [x, z] of outline.slice(1)) shape.lineTo(x, -z);
  shape.closePath();
  const geometry = new THREE.ExtrudeGeometry(shape, {
    depth,
    bevelEnabled: true,
    bevelSegments: 3,
    steps: 1,
    bevelSize: 0.09,
    bevelThickness: 0.07,
  });
  const object = mesh(parent, geometry, surface, [0, y, 0]);
  object.rotation.x = -Math.PI / 2;
  return object;
}

function agentMark(parent) {
  // Preserve the standing figure from Apparat Fabrik's FIGURE sprite.
  const figure = [
    "  ##  ",
    " #### ",
    " #  # ",
    " #### ",
    "######",
    " #### ",
    " #  # ",
    " #  # ",
  ];
  const canvas = document.createElement("canvas");
  canvas.width = 6;
  canvas.height = 8;
  const context = canvas.getContext("2d");
  context.fillStyle = "#f3eee0";
  figure.forEach((row, y) =>
    [...row].forEach((cell, x) => {
      if (cell === "#") context.fillRect(x, y, 1, 1);
    }),
  );
  const texture = new THREE.CanvasTexture(canvas);
  texture.magFilter = texture.minFilter = THREE.NearestFilter;
  texture.colorSpace = THREE.SRGBColorSpace;
  const surface = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
  });
  const mark = mesh(
    parent,
    new THREE.PlaneGeometry(0.63, 0.84),
    surface,
    [-2.65, 1.137, 0],
  );
  mark.rotation.x = -Math.PI / 2;
  mark.castShadow = false;
}

function createBodywork(chassis) {
  body(
    chassis,
    [
      [-2.5, -0.55],
      [-1.5, -0.85],
      [0.5, -0.73],
      [1.2, -0.3],
      [3, -0.18],
      [3.12, 0],
      [3, 0.18],
      [1.2, 0.3],
      [0.5, 0.73],
      [-1.5, 0.85],
      [-2.5, 0.55],
    ],
    0.12,
    carbon,
    0.27,
  );
  body(
    chassis,
    [
      [-2.1, -0.3],
      [-1.3, -0.66],
      [0.25, -0.59],
      [0.85, -0.31],
      [2.9, -0.13],
      [3, 0],
      [2.9, 0.13],
      [0.85, 0.31],
      [0.25, 0.59],
      [-1.3, 0.66],
      [-2.1, 0.3],
    ],
    0.29,
    ivory,
    0.45,
  );
  body(
    chassis,
    [
      [-2.25, -0.2],
      [-1.4, -0.32],
      [-0.7, -0.27],
      [-0.6, 0.27],
      [-1.4, 0.32],
      [-2.25, 0.2],
    ],
    0.3,
    red,
    0.72,
  );
  body(
    chassis,
    [
      [0.72, -0.19],
      [2.8, -0.11],
      [3, 0],
      [2.8, 0.11],
      [0.72, 0.19],
    ],
    0.05,
    red,
    0.8,
  );
  for (const sign of [-1, 1]) {
    body(
      chassis,
      [
        [-1.5, sign * 0.75],
        [-0.5, sign * 0.86],
        [0.48, sign * 0.69],
        [0.2, sign * 0.46],
        [-1.3, sign * 0.42],
      ],
      0.23,
      blue,
      0.5,
    );
    box(chassis, [0.55, 0.13, 0.21], carbon, [0.16, 0.8, sign * 0.58]);
    for (let i = 0; i < 6; i++)
      box(chassis, [0.035, 0.013, 0.22], carbon, [
        -1.15 + i * 0.1,
        0.83,
        sign * 0.57,
      ]);
  }
  for (const sign of [-1, 1])
    box(chassis, [1.65, 0.018, 0.055], yellow, [-0.5, 0.825, sign * 0.7]);
}

function createCockpit(chassis) {
  const cockpit = mesh(
    chassis,
    new THREE.SphereGeometry(0.46, 32, 16),
    carbon,
    [0.13, 0.87, 0],
  );
  cockpit.scale.set(1.35, 0.35, 0.72);
  const helmet = mesh(
    chassis,
    new THREE.SphereGeometry(0.18, 24, 16),
    yellow,
    [-0.08, 0.99, 0],
  );
  helmet.scale.y = 0.85;
  const visor = mesh(
    chassis,
    new THREE.SphereGeometry(0.183, 20, 12, 0, Math.PI),
    material("#172f33", { metalness: 0.8, roughness: 0.12 }),
    [-0.06, 1, 0],
  );
  visor.rotation.y = Math.PI / 2;
  visor.scale.y = 0.35;
  const halo = mesh(
    chassis,
    new THREE.TorusGeometry(0.41, 0.038, 8, 40, Math.PI * 1.65),
    carbon,
    [0.15, 1.06, 0],
  );
  halo.rotation.x = Math.PI / 2;
  halo.scale.x = 1.45;
  rod(chassis, [0.64, 0.83, 0], [0.67, 1.06, 0], 0.035, carbon);
}

function createWings(chassis) {
  box(chassis, [0.8, 0.09, 1.9], carbon, [-2.65, 1.08, 0]);
  box(chassis, [0.65, 0.09, 1.9], blue, [-2.6, 0.93, 0]);
  for (const sign of [-1, 1]) {
    box(chassis, [0.87, 0.48, 0.055], yellow, [-2.6, 0.91, sign * 0.97]);
    rod(
      chassis,
      [-2.2, 0.5, sign * 0.25],
      [-2.65, 1.05, sign * 0.45],
      0.045,
      carbon,
    );
    box(chassis, [0.72, 0.3, 0.04], yellow, [2.96, 0.44, sign * 1.04]);
  }
  box(chassis, [0.6, 0.08, 2.05], carbon, [2.98, 0.3, 0]);
  const flaps = {};
  for (const [side, sign] of [
    ["left", -1],
    ["right", 1],
  ]) {
    const flap = box(chassis, [0.38, 0.055, 0.85], ivory, [
      2.93,
      0.4,
      sign * 0.56,
    ]);
    flaps[side] = flap;
  }
  return flaps;
}

function createWheels(chassis) {
  const wheels = {};
  const hubs = {};
  for (const [corner, [x, z]] of Object.entries(CORNERS)) {
    for (const from of [x - 0.6, x + 0.5])
      rod(
        chassis,
        [from, 0.43, Math.sign(z) * 0.36],
        [x, 0.49, z],
        0.035,
        carbon,
      );
    rod(chassis, [x, 0.48, 0], [x, 0.48, z], 0.045, silver);
    const hub = cylinder(chassis, 0.17, 0.12, silver, [x, 0.49, z]);
    hub.rotation.x = Math.PI / 2;
    hubs[corner] = hub;
    const wheel = tire();
    wheel.position.set(x, 0.49, z);
    chassis.add(wheel);
    const fresh = tire(true);
    fresh.position.copy(wheel.position);
    chassis.add(fresh);
    wheels[corner] = { old: wheel, fresh };
  }
  return { wheels, hubs };
}

export function createCar() {
  const root = new THREE.Group();
  const chassis = new THREE.Group();
  root.add(chassis);

  createBodywork(chassis);
  createCockpit(chassis);
  const flaps = createWings(chassis);
  const { wheels, hubs } = createWheels(chassis);

  agentMark(chassis);
  decal(chassis, "27", 0.75, 0.55, [1.55, 0.88, 0], "#182326", 150);
  return { root, chassis, wheels, hubs, flaps };
}
