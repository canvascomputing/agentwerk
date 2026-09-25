import layout from "../layout.json";
import * as THREE from "three";
import { box, cylinder, material, mesh, rod } from "./geometry.js";
import { createCar, tire } from "./car.js";
import { createEquipment } from "./equipment.js";
import { createCrew, createJack } from "./crew.js";

function asphaltTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 512;
  const context = canvas.getContext("2d");
  const pixels = context.createImageData(512, 512);
  let seed = 27;
  for (let i = 0; i < pixels.data.length; i += 4) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    const value = 70 + (seed % 30);
    pixels.data.set([value, value + 7, value + 6, 255], i);
  }
  context.putImageData(pixels, 0, 0);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(10, 10);
  return texture;
}

function streetPaint(scene) {
  const glyphs = {
    a: ["00000", "01110", "00001", "01111", "10001", "01111", "00000"],
    g: ["00000", "01111", "10001", "10001", "01111", "00001", "01110"],
    e: ["00000", "01110", "10001", "11111", "10000", "01111", "00000"],
    n: ["00000", "11110", "10001", "10001", "10001", "10001", "00000"],
    t: ["00100", "11111", "00100", "00100", "00100", "00011", "00000"],
    w: ["00000", "10001", "10001", "10101", "10101", "01010", "00000"],
    r: ["00000", "10110", "11001", "10000", "10000", "10000", "00000"],
    k: ["10000", "10010", "10100", "11000", "10100", "10010", "00000"],
  };
  const canvas = document.createElement("canvas");
  canvas.width = 1152;
  canvas.height = 224;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#b51f29";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#fff7e3";
  [..."agentwerk"].forEach((letter, index) =>
    glyphs[letter].forEach((row, y) =>
      [...row].forEach((pixel, x) => {
        if (pixel === "1")
          ctx.fillRect(80 + index * 108 + x * 18, 35 + y * 22, 18, 22);
      }),
    ),
  );
  let seed = 57;
  for (let i = 0; i < 8000; i++) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    const x = seed % canvas.width;
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    ctx.fillStyle = i % 2 ? "#00000009" : "#ffffff09";
    ctx.fillRect(x, seed % canvas.height, 1, 1);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.repeat.set(6, 1);
  const paint = material("#ffffff", {
    map: texture,
    transparent: true,
    depthWrite: false,
    roughness: 1,
  });
  const word = mesh(
    scene,
    new THREE.PlaneGeometry(51.6, 1.65),
    paint,
    [0.35, 0.025, -4.05],
  );
  word.rotation.x = -Math.PI / 2;
  word.receiveShadow = true;
}

function pitMarkings(scene) {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 128;
  const context = canvas.getContext("2d");
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, 512, 128);
  context.globalCompositeOperation = "destination-out";
  let seed = 81;
  for (let index = 0; index < 1800; index++) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    const x = seed % 512;
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    context.globalAlpha = 0.2 + (seed % 60) / 100;
    context.fillRect(x, seed % 128, 1 + (seed % 4), 1 + (seed % 3));
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.colorSpace = THREE.SRGBColorSpace;
  const paint = material("#c99412", {
    map: texture,
    transparent: true,
    depthWrite: false,
    roughness: 1,
  });
  function strip(width, depth, x, z) {
    const line = mesh(scene, new THREE.PlaneGeometry(width, depth), paint, [
      x,
      0.026,
      z,
    ]);
    line.rotation.x = -Math.PI / 2;
    line.castShadow = false;
  }
  for (const sign of [-1, 1]) {
    strip(9.2, 0.16, 0, sign * 2.4);
    for (const x of [-2.05, 2.05]) strip(0.16, 0.52, x, sign * 2.66);
  }
}

function paintApron(scene, { paint, yellow, black }) {
  pitMarkings(scene);
  for (const z of [-5.0, 5.1])
    box(scene, [70, 0.013, 0.07], yellow, [0, 0.012, z]);
  for (let x = -30; x < 30; x += 3)
    box(scene, [1.35, 0.013, 0.055], paint, [x, 0.012, 6.9]);
  for (let x = -18; x < 18; x += 4) {
    box(scene, [0.025, 0.011, 25], black, [x, 0.002, -4]);
  }
  for (const z of [-8, -4, 4, 8])
    box(scene, [60, 0.011, 0.022], black, [0, 0.002, z]);
  streetPaint(scene);
}

function buildGarage(scene, { black, steel }) {
  const garage = material("#2c3d40");
  box(scene, [27, 0.25, 5], garage, [0, 0, -8]);
  box(scene, [27, 1.8, 0.25], black, [0, 0.9, -10.4]);
  for (let x = -12; x <= 12; x += 6) {
    box(scene, [0.22, 1.8, 5], steel, [x, 0.9, -8]);
    box(
      scene,
      [5.6, 0.03, 0.16],
      material("#d8e6d5", { emissive: "#a8c5bc", emissiveIntensity: 0.45 }),
      [x + 3, 1.9, -7.3],
    );
  }
}

function buildStations(
  scene,
  { black, steel },
  recordedLayout,
  modern,
  version,
) {
  const cabinetBlue = material("#205cb2", { roughness: 0.55 });
  const topBlue = material("#367ed2", { metalness: 0.25, roughness: 0.5 });
  for (const [x, z] of Object.values(recordedLayout.stations)) {
    const height = modern && x > 0 ? 0.1 : 0.65;
    const cabinet = box(scene, [3.2, height, 0.62], cabinetBlue, [
      x,
      height / 2,
      z,
    ]);
    for (let i = 0; i < (height > 0.2 ? 4 : 0); i++)
      box(scene, [2.95, 0.025, 0.02], steel, [x, 0.15 + i * 0.13, z + 0.33]);
    box(scene, [3.3, 0.06, 0.69], topBlue, [x, height + 0.05, z]);
    for (const sign of [-1, 1])
      cylinder(scene, 0.09, 0.12, black, [x + sign * 1.2, 0.1, z + 0.2]);
    if (!modern) {
      for (let j = 0; j < 3; j++) {
        const wheel = tire(true);
        wheel.rotation.x = Math.PI / 2;
        wheel.position.set(x - 1.9, 0.3 + j * 0.38, z);
        scene.add(wheel);
      }
    }
    cabinet.castShadow = true;
  }
  if (modern && version < 3) {
    for (const [name, point] of Object.entries(recordedLayout.slots)) {
      if (name.startsWith("used-"))
        box(scene, [0.85, 0.05, 0.65], cabinetBlue, [point[0], 0.04, point[2]]);
    }
  }
  for (const x of [-11, 11]) {
    cylinder(scene, 0.12, 1.1, steel, [x, 0.55, -2.9]);
    cylinder(scene, 0.17, 0.55, black, [x, 0.3, -2.9]);
    rod(scene, [x, 1.1, -2.9], [x, 1.45, -2.9], 0.026, steel);
  }
}

function garageDetails(scene, { black, steel }) {
  const red = material("#c73932");
  const cloth = material("#c5b68e", { roughness: 1 });
  for (let turn = 0; turn < 3; turn++) {
    const hose = mesh(
      scene,
      new THREE.TorusGeometry(0.3 + turn * 0.075, 0.035, 6, 32),
      black,
      [-8.6, 0.08, -2.4],
    );
    hose.rotation.x = Math.PI / 2;
  }
  cylinder(scene, 0.15, 0.6, red, [8.8, 0.33, 2.6]);
  cylinder(scene, 0.08, 0.12, steel, [8.8, 0.69, 2.6]);
  rod(scene, [8.68, 0.78, 2.6], [8.92, 0.78, 2.6], 0.035, black);
  box(scene, [0.43, 0.025, 0.3], black, [-4.65, 0.745, -4.5]);
  const rag = box(scene, [0.3, 0.035, 0.23], cloth, [-7.4, 0.75, -4.5]);
  rag.rotation.y = 0.2;
}

function environment(scene, metadata) {
  const concrete = material("#818f8b", { map: asphaltTexture(), roughness: 1 });
  box(scene, [100, 0.18, 70], concrete, [0, -0.12, 0]);
  const surfaces = {
    paint: material("#c2c8a5", { roughness: 0.96 }),
    yellow: material("#c0ad73"),
    black: material("#1b272a"),
    steel: material("#748b8b", { metalness: 0.55 }),
  };

  paintApron(scene, surfaces);
  buildGarage(scene, surfaces);
  if (metadata?.version >= 3) garageDetails(scene, surfaces);
  buildStations(
    scene,
    surfaces,
    metadata?.layout ?? layout,
    !!metadata?.state.items,
    metadata?.version ?? 1,
  );
}

export function createScene(canvas, crew, metadata) {
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: false,
    preserveDrawingBuffer: true,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.35;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#182528");
  scene.fog = new THREE.Fog("#182528", 32, 65);
  const camera = new THREE.OrthographicCamera(-12, 12, 7, -7, 0.1, 100);
  camera.position.set(0, 27, 18);
  camera.lookAt(0, 0, -0.2);
  scene.add(new THREE.HemisphereLight("#e1efea", "#334039", 2.5));
  const key = new THREE.DirectionalLight("#f6e8ca", 4.5);
  key.position.set(-6, 14, -7);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  Object.assign(key.shadow.camera, {
    left: -16,
    right: 16,
    top: 13,
    bottom: -13,
    near: 1,
    far: 45,
  });
  key.shadow.normalBias = 0.035;
  key.shadow.bias = -0.0002;
  key.shadow.radius = 3;
  scene.add(key);
  const fill = new THREE.DirectionalLight("#93bec8", 1.4);
  fill.position.set(7, 9, 5);
  scene.add(fill);
  environment(scene, metadata);
  const car = createCar();
  scene.add(car.root);
  const workers = Object.fromEntries(
    crew.map((member) => {
      const worker = createCrew(member, metadata?.seed);
      scene.add(worker.root);
      return [member.id, worker];
    }),
  );
  const jacks = Object.fromEntries(
    ["front", "rear"].map((end) => {
      const jack = createJack(end);
      scene.add(jack.root);
      return [end, jack];
    }),
  );
  const items = createEquipment(scene, metadata?.state.items ?? {});
  function resize() {
    const { width, height } = canvas.getBoundingClientRect();
    renderer.setSize(width, height, false);
    const aspect = width / height;
    const halfWidth = Math.max(9.3, 4.15 * aspect);
    camera.left = -halfWidth;
    camera.right = halfWidth;
    camera.top = halfWidth / aspect;
    camera.bottom = -halfWidth / aspect;
    camera.updateProjectionMatrix();
  }
  resize();
  return {
    renderer,
    scene,
    camera,
    car,
    workers,
    jacks,
    items,
    resize,
  };
}
