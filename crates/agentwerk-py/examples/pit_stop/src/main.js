import "@fontsource/silkscreen/latin-400.css";
import "./style.css";
import { Box3, Vector3 } from "three";
import {
  Playback,
  phaseTimers,
  phaseTitle,
  phaseCompleted,
} from "./playback.js";
import { createScene } from "./scene.js";
import { animateScene } from "./animation.js";
import { updateServiceDiagram } from "./hud.js";

const ui = Object.fromEntries(
  [
    "phase",
    "hold-reason",
    "progress",
    "error",
    "timers",
    "service-car",
    "service-side",
  ].map((id) => [id, document.getElementById(id)]),
);

function updateHud(playback, sample) {
  const { state, time } = sample;
  if (!state) return;
  ui.phase.textContent = phaseTitle(sample);
  updateServiceDiagram(ui["service-car"], ui["service-side"], sample);
  const timers = phaseTimers(sample);
  const completed = phaseCompleted(sample);
  for (const [name, seconds] of Object.entries(timers)) {
    const cell = ui.timers.querySelector(`[data-timer="${name}"]`);
    cell.querySelector("output").textContent = `${seconds.toFixed(1)}s`;
    cell.dataset.completed = String(completed[name]);
  }
  ui["hold-reason"].hidden = !state.held;
  ui["hold-reason"].textContent = state.held ?? "";
  ui.progress.style.width = `${Math.min(100, (time / playback.duration) * 100)}%`;
}

function showError(message) {
  ui.error.hidden = false;
  ui.error.textContent = message;
}

function currentItem(sample, actor) {
  return sample.state.items[sample.state.crew[actor].equipment];
}

function projectedBounds(object, camera) {
  const box = new Box3().setFromObject(object);
  const points = [];
  for (const x of [box.min.x, box.max.x])
    for (const y of [box.min.y, box.max.y])
      for (const z of [box.min.z, box.max.z])
        points.push(new Vector3(x, y, z).project(camera));
  return {
    left: Math.min(...points.map((p) => p.x)),
    right: Math.max(...points.map((p) => p.x)),
    bottom: Math.min(...points.map((p) => p.y)),
    top: Math.max(...points.map((p) => p.y)),
  };
}

function inspectScene(world, playback) {
  const sample = playback.sample();
  const crew = world
    ? Object.fromEntries(
        Object.entries(world.workers).map(([id, worker]) => [
          id,
          {
            position: [worker.root.position.x, worker.root.position.z],
            bounds: projectedBounds(worker.root, world.camera),
            heading: worker.root.rotation.y,
            legs: worker.legs.map((leg) => leg.rotation.x),
            boardGrip: worker.releaseSign
              ? worker.hands.map((hand, i) =>
                  hand
                    .getWorldPosition(hand.position.clone())
                    .distanceTo(
                      worker.releaseSign.root.localToWorld(
                        hand.position.clone().set(i ? 0.35 : -0.35, 0, 0.015),
                      ),
                    ),
                )
              : null,
            tool: sample.state.items
              ? currentItem(sample, id)?.kind === "gunner" ||
                currentItem(sample, id)?.kind === "wing"
              : worker.tool.visible,
            tire: sample.state.items
              ? ["fresh", "old"].includes(currentItem(sample, id)?.kind)
              : worker.carried.visible,
          },
        ]),
      )
    : {};
  const wheels = world
    ? Object.fromEntries(
        Object.entries(world.car.wheels).map(([corner, pair]) => [
          corner,
          sample.state.items
            ? {
                old:
                  sample.state.items[`old-${corner}`].owner === `hub:${corner}`,
                fresh:
                  sample.state.items[`fresh-${corner}`].owner ===
                  `hub:${corner}`,
              }
            : { old: pair.old.visible, fresh: pair.fresh.visible },
        ]),
      )
    : {};

  return {
    items: Object.fromEntries(
      Object.entries(world?.items ?? {}).map(([id, object]) => [
        id,
        {
          position: object.position.toArray(),
          heading: object.rotation.y,
          visible: object.visible,
          grip: object.userData.handle
            ? object
                .localToWorld(new Vector3(...object.userData.handle))
                .toArray()
            : null,
        },
      ]),
    ),
    state: sample.state,
    carX: world?.car.root.position.x,
    carZ: world?.car.root.position.z,
    carHeading: world?.car.root.rotation.y,
    releaseSign: world?.workers.chief.releaseSign.plate.material.name,
    crew,
    frontHeight: world?.car.chassis.position.y,
    wheels,
  };
}

function connectLiveFeed(playback, onFrame) {
  const source = new EventSource("/events");
  source.onmessage = (event) => {
    try {
      playback.append([JSON.parse(event.data)]);
      onFrame();
    } catch (error) {
      source.close();
      showError(error.message);
    }
  };
  return source;
}

async function start() {
  const response = await fetch("/api/run");
  if (!response.ok)
    throw new Error("The pit-stop server could not load this run.");
  const config = await response.json();
  const playback = new Playback(config.frames, config.mode);
  if (config.mode === "live") {
    playback.time = config.elapsed;
    playback.clockRunning = config.running ?? true;
  }
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  playback.paused = reduced;
  let world = null;

  function render() {
    const sample = playback.sample();
    if (!world && sample.metadata) {
      try {
        world = createScene(
          document.getElementById("scene"),
          sample.metadata.crew,
          sample.metadata,
        );
      } catch {
        throw new Error(
          "This scene needs WebGL. Enable browser hardware acceleration and reload, or view the GIF in the project README.",
        );
      }
    }
    if (world) animateScene(world, sample);
    updateHud(playback, sample);
    return sample;
  }
  window.addEventListener("keydown", (event) => {
    if (event.repeat || event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.code === "Space") {
      event.preventDefault();
      playback.paused = !playback.paused;
    } else if (event.code === "KeyR" && playback.mode === "replay") {
      playback.reset();
      render();
    }
  });
  const source =
    config.mode === "live"
      ? connectLiveFeed(playback, () => {
          if (!world) render();
        })
      : null;
  window.addEventListener("resize", () => {
    world?.resize();
    render();
  });
  document
    .getElementById("scene")
    .addEventListener("webglcontextlost", (event) => {
      event.preventDefault();
      playback.paused = true;
      showError("The graphics context was lost. Reload to restore the scene.");
    });
  // Capture and browser tests advance the real renderer with a fixed clock.
  window.pitStop = {
    playback,
    seek(seconds) {
      playback.paused = true;
      playback.time = seconds;
      return render();
    },
    inspect() {
      return inspectScene(world, playback);
    },
  };
  if (reduced && config.mode === "replay")
    playback.time =
      playback.frames.find((frame) => frame.name === "car_stopped")?.t ?? 0;
  render();
  let previous = performance.now();
  function tick(now) {
    playback.tick((now - previous) / 1000);
    previous = now;
    try {
      if (!playback.paused) render();
      requestAnimationFrame(tick);
    } catch (error) {
      showError(error.message);
      source?.close();
    }
  }
  requestAnimationFrame(tick);
}

start().catch((error) => showError(error.message));
