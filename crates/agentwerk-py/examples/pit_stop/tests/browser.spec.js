import { test, expect } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.pitStop);
});

test("car arrives, exposes hubs, fits wheels, and departs only after release", async ({
  page,
}) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const frames = await page.evaluate(() => window.pitStop.playback.frames);
  async function seek(frame, offset = 0) {
    return page.evaluate((t) => {
      window.pitStop.seek(t);
      return window.pitStop.inspect();
    }, frame.t + offset);
  }
  const initial = await seek(frames[0]);
  expect(initial.carX).toBeLessThan(-10);
  const arrival = frames.find((frame) => frame.name === "car_arriving");
  const entering = await seek(arrival, arrival.data.duration / 2);
  expect(entering.carX).toBeLessThan(0);
  expect(entering.carX).toBeGreaterThan(-10);
  expect(entering.state.car).toBe("approaching");
  const movingCrew = Object.entries(entering.crew).filter(
    ([id, worker]) =>
      Math.hypot(
        worker.position[0] - initial.crew[id].position[0],
        worker.position[1] - initial.crew[id].position[1],
      ) > 0.2,
  );
  expect(movingCrew.length).toBeGreaterThanOrEqual(2);
  expect(
    await seek(frames.find((f) => f.name === "car_stopped")),
  ).toMatchObject({ carX: 0, state: { car: "stopped" } });
  const removal = frames.find(
    (f) => f.name === "action_completed" && f.data.action === "remove",
  );
  const empty = await seek(removal);
  const corner = removal.data.actor.replace("wheel-off-", "");
  expect(empty.wheels[corner]).toEqual({ old: false, fresh: false });
  expect(empty.frontHeight).toBeGreaterThan(0.2);
  const release = frames.find(
    (f) => f.name === "action_completed" && f.data.action === "release",
  );
  expect((await seek(release, -0.01)).state.car).toBe("stopped");
  const departure = frames.find((f) => f.name === "car_departing");
  const moving = await seek(departure, 2);
  expect(moving.carX).toBeGreaterThan(0);
  expect(moving.state.car).toBe("departing");
  const end = await seek(frames.at(-1));
  expect(end.carX).toBeGreaterThan(18);
  expect(
    Object.values(end.wheels).every((wheel) => wheel.fresh && !wheel.old),
  ).toBe(true);
  expect(errors).toEqual([]);
  await expect(page.locator("#error")).toBeHidden();
});

test("pause, replay, and resizing retain a usable scene", async ({ page }) => {
  await page.keyboard.press("Space");
  const time = await page.evaluate(() => window.pitStop.playback.time);
  await page.waitForTimeout(150);
  expect(await page.evaluate(() => window.pitStop.playback.time)).toBe(time);
  await page.keyboard.press("r");
  expect(await page.evaluate(() => window.pitStop.playback.time)).toBe(0);
  await page.setViewportSize({ width: 600, height: 800 });
  await expect(page.locator("#scene")).toBeVisible();
  await expect(page.locator("#error")).toBeHidden();
});

test("reduced motion starts paused on the stopped car", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.reload();
  await page.waitForFunction(() => window.pitStop);
  expect(await page.evaluate(() => window.pitStop.playback.paused)).toBe(true);
  expect((await page.evaluate(() => window.pitStop.inspect())).state.car).toBe(
    "stopped",
  );
});

test("unavailable WebGL displays a useful error", async ({ page }) => {
  await page.addInitScript(() => {
    const original = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type, ...args) {
      return type.includes("webgl") ? null : original.call(this, type, ...args);
    };
  });
  await page.reload();
  await expect(page.locator("#error")).toContainText("WebGL");
});

test("recorded travel, equipment, release signal, and car transforms survive a full reset", async ({
  page,
}) => {
  const result = await page.evaluate(() => {
    const pit = window.pitStop;
    const frames = pit.playback.frames;
    const inspect = (time) => {
      pit.seek(time);
      return pit.inspect();
    };
    const initial = inspect(frames[0].t);
    const poses = frames
      .filter(
        (frame) =>
          frame.name.startsWith("car_") ||
          frame.name === "action_effect" ||
          (frame.name === "action_phase" && frame.data.actor === "chief"),
      )
      .map((frame) => inspect(frame.t));
    const collect = frames.find(
      (frame) =>
        frame.name === "action_completed" && frame.data.action === "collect",
    );
    const equipped = inspect(collect.t).crew[collect.data.actor];
    const removal = frames.find(
      (frame) =>
        frame.name === "action_started" && frame.data.action === "stow",
    );
    let elapsed = 0;
    const returning = removal.data.phases.find((phase) => {
      if (phase.kind === "return") return true;
      elapsed += phase.duration;
      return false;
    });
    const carrying = inspect(removal.t + elapsed + returning.duration / 2).crew[
      removal.data.actor
    ];
    const dropped = frames.find(
      (frame) =>
        frame.name === "action_completed" &&
        frame.data.actor === removal.data.actor &&
        frame.data.action === "stow",
    );
    const afterDrop = inspect(dropped.t).crew[removal.data.actor];
    const end = inspect(frames.at(-1).t);
    const reset = inspect(frames[0].t);
    return { initial, reset, poses, equipped, carrying, afterDrop, end };
  });
  for (const pose of result.poses) {
    for (const gap of pose.crew.chief.boardGrip)
      expect(gap).toBeLessThan(0.015);
    expect(
      Number.isFinite(pose.carX) &&
        Number.isFinite(pose.carZ) &&
        Number.isFinite(pose.carHeading),
    ).toBe(true);
    if (pose.state.car === "stopped") {
      expect(pose.carX).toBe(0);
      expect(pose.carZ).toBe(0);
      expect(pose.carHeading).toBeCloseTo(0);
      expect(pose.releaseSign).toBe("STOP");
    }
    if (["departing", "departed"].includes(pose.state.car)) {
      expect(pose.releaseSign).toBe("GO");
      expect(Math.abs(pose.crew.chief.position[1])).toBeGreaterThan(1.7);
    }
  }
  expect(result.equipped.tool).toBe(true);
  expect(result.carrying.tire).toBe(true);
  expect(result.afterDrop.tire).toBe(false);
  expect(result.end.releaseSign).toBe("GO");
  expect(result.end.crew.chief.position).toEqual([9, -5.8]);
  expect(result.reset).toEqual(result.initial);
  await expect(page.getByLabel("Crew roles")).toHaveCount(0);
});

test("each item remains visible and continuous across recorded handoffs", async ({
  page,
}) => {
  const checks = await page.evaluate(() => {
    const pit = window.pitStop;
    const frames = pit.playback.frames;
    const result = [];
    for (const frame of frames.filter((f) => f.name === "action_transfer")) {
      pit.seek(frame.t - 0.00001);
      const before = pit.inspect().items[frame.data.item];
      pit.seek(frame.t + 0.00001);
      const after = pit.inspect().items[frame.data.item];
      result.push({
        item: frame.data.item,
        before,
        after,
        count: Object.keys(pit.inspect().items).length,
      });
    }
    return result;
  });
  expect(checks.length).toBe(28);
  for (const check of checks) {
    expect(check.count).toBe(14);
    expect(check.before.visible && check.after.visible).toBe(true);
    expect(
      Math.hypot(
        ...check.before.position.map((p, i) => p - check.after.position[i]),
      ),
    ).toBeLessThan(0.01);
  }
});

test("tire pickups keep the tread clear of the carrier without a half-turn", async ({
  page,
}) => {
  const pickups = await page.evaluate(() => {
    const pit = window.pitStop;
    return pit.playback.frames
      .filter(
        (frame) =>
          frame.name === "action_phase" &&
          frame.data.kind === "grip" &&
          frame.data.actor.startsWith("wheel-on-"),
      )
      .map((frame) => {
        pit.seek(frame.t);
        const sample = pit.playback.sample();
        const phase =
          sample.actions[frame.data.actor].data.phases[frame.data.phase];
        const id = phase.transfer.item;
        const before = pit.inspect().items[id];
        pit.seek(frame.t + phase.duration / 2);
        const view = pit.inspect();
        return {
          before,
          tire: view.items[id],
          worker: view.crew[frame.data.actor],
        };
      });
  });
  expect(pickups).toHaveLength(4);
  for (const { before, tire, worker } of pickups) {
    expect(Math.abs(tire.heading - before.heading)).toBeLessThan(0.05);
    const forward =
      (tire.position[0] - worker.position[0]) * Math.sin(worker.heading) +
      (tire.position[2] - worker.position[1]) * Math.cos(worker.heading);
    expect(forward - 0.195).toBeGreaterThan(0.21);
  }
});

test("compact overhead stays fixed while side view shows independent jack lift", async ({
  page,
}) => {
  await page.setViewportSize({ width: 800, height: 450 });
  await page.evaluate(() => window.pitStop.seek(25));
  await expect(page.locator("#steps")).toHaveCount(0);
  await expect(page.locator(".status-copy > *")).toHaveCount(2);
  await expect(page.locator(".wheel-meter")).toHaveCount(0);
  await expect(page.locator("#service-car .mini-tire")).toHaveCount(4);
  await expect(page.locator("#service-side .mini-tire")).toHaveCount(2);
  const removed = await page.evaluate(() => {
    const frame = window.pitStop.playback.frames.find(
      (f) =>
        f.name === "action_phase" &&
        f.data.action === "lift" &&
        f.data.kind === "work",
    );
    window.pitStop.seek(frame.t + 0.3);
    return window.pitStop.playback.frames.find(
      (f) => f.name === "action_effect" && f.data.action === "remove",
    );
  });
  await expect(page.locator("#service-side")).toHaveAttribute(
    "data-lifted",
    "true",
  );
  const lift = await page
    .locator("#service-side")
    .evaluate((e) => [Number(e.dataset.front), Number(e.dataset.rear)]);
  expect(Math.abs(lift[0] - lift[1])).toBeGreaterThan(0.05);
  expect(
    await page.locator("#service-car .car-diagram").getAttribute("transform"),
  ).toBeNull();
  expect(
    await page.locator("#service-car").getAttribute("data-lifted"),
  ).toBeNull();
  await expect(page.locator("#service-side .side-car")).toHaveAttribute(
    "transform",
    /rotate\((?!0 )/,
  );
  await page.evaluate((time) => window.pitStop.seek(time), removed.t);
  const corner = removed.data.actor.replace("wheel-off-", "");
  await expect(
    page.locator(`#service-car [data-corner="${corner}"]`),
  ).toHaveAttribute("data-status", "empty");
  await expect(
    page.locator(`#service-car [data-corner="${corner}"] .mini-tire`),
  ).toHaveAttribute("opacity", "0");
  await page.evaluate(() =>
    window.pitStop.seek(window.pitStop.playback.frames.at(-1).t),
  );
  await expect(page.locator("#service-side")).toHaveAttribute(
    "data-lifted",
    "false",
  );
  await expect(page.locator("#service-side .side-car")).toHaveAttribute(
    "transform",
    "translate(0 0) rotate(0 40 47)",
  );
  await expect(
    page.locator('#service-car [data-status="secured"]'),
  ).toHaveCount(4);
  await expect(page.locator(".status-copy #activity")).toContainText("kit");
  expect(
    await page
      .locator(".progress")
      .evaluate((e) => e.getBoundingClientRect().height),
  ).toBe(6);
  const scene = await page.locator("#scene").boundingBox();
  const hud = await page.locator(".status").boundingBox();
  expect(hud.y).toBeGreaterThanOrEqual(scene.y + scene.height);
  await page.setViewportSize({ width: 390, height: 700 });
  expect(
    await page.locator(".status").evaluate((e) => e.scrollWidth <= innerWidth),
  ).toBe(true);
});

test("older recordings retain equipment animation and complete departure", async ({
  page,
}) => {
  const { readFileSync } = await import("node:fs");
  const frames = JSON.parse(
    readFileSync(new URL("fixtures/legacy.json", import.meta.url), "utf8"),
  );
  await page.route("**/api/run", (route) =>
    route.fulfill({ json: { mode: "replay", frames } }),
  );
  await page.reload();
  await page.waitForFunction(() => window.pitStop);
  const result = await page.evaluate(() => {
    const pit = window.pitStop;
    const frames = pit.playback.frames;
    pit.seek(frames[0].t);
    const initial = pit.inspect();
    const collect = frames.find(
      (f) => f.name === "action_completed" && f.data.action === "collect",
    );
    pit.seek(collect.t);
    const equipped = pit.inspect().crew[collect.data.actor].tool;
    pit.seek(frames.at(-1).t);
    return { initial, equipped, end: pit.inspect() };
  });
  expect(result.initial.items).toEqual({});
  expect(result.equipped).toBe(true);
  expect(result.end.carX).toBeGreaterThan(18);
  await expect(page.locator("#error")).toBeHidden();
});
