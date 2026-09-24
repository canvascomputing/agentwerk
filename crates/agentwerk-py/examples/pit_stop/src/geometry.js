import * as THREE from "three";

export function material(color, options = {}) {
  return new THREE.MeshStandardMaterial({ color, roughness: 0.65, ...options });
}

export function mesh(parent, geometry, surface, position = [0, 0, 0]) {
  const object = new THREE.Mesh(geometry, surface);
  object.position.set(...position);
  object.castShadow = true;
  object.receiveShadow = true;
  parent.add(object);
  return object;
}

export function box(parent, dimensions, surface, position) {
  return mesh(parent, new THREE.BoxGeometry(...dimensions), surface, position);
}

export function cylinder(
  parent,
  radius,
  length,
  surface,
  position,
  top = radius,
) {
  return mesh(
    parent,
    new THREE.CylinderGeometry(top, radius, length, 24),
    surface,
    position,
  );
}

export function rod(parent, start, end, radius, surface) {
  const startPoint = new THREE.Vector3(...start);
  const endPoint = new THREE.Vector3(...end);
  const object = cylinder(
    parent,
    radius,
    startPoint.distanceTo(endPoint),
    surface,
    startPoint.clone().add(endPoint).multiplyScalar(0.5).toArray(),
  );
  object.quaternion.setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    endPoint.sub(startPoint).normalize(),
  );
  return object;
}

export function decal(
  parent,
  text,
  width,
  height,
  position,
  color = "#d5ddd1",
  size = 80,
) {
  const canvas = document.createElement("canvas");
  canvas.width = 1024;
  canvas.height = 256;
  const context = canvas.getContext("2d");
  context.font = `600 ${size}px Arial`;
  context.fillStyle = color;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, 512, 128);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const surface = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -2,
  });
  const object = mesh(
    parent,
    new THREE.PlaneGeometry(width, height),
    surface,
    position,
  );
  object.rotation.x = -Math.PI / 2;
  object.castShadow = false;
  return object;
}
