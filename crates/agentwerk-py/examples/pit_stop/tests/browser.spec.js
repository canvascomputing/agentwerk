import { test, expect } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.pitStop);
});

test("recorded agents prepare, service the car, report, and depart after GO", async ({
  page,
}) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const result = await page.evaluate(() => {
    const pit = window.pitStop;
    const frames = pit.playback.frames;
    const at = (time) => {
      pit.seek(time);
      return pit.inspect();
    };
    const initial = at(0);
    const arrival = frames.find((f) => f.name === "car_arriving");
    const entering = at(arrival.t + arrival.data.duration / 2);
    const removed = frames.find(
      (f) => f.name === "crew_work" && f.data.work === "remove",
    );
    const empty = at(removed.t);
    const release = frames.find((f) => f.name === "pit_released");
    const beforeRelease = at(release.t - 0.001);
    const end = at(frames.at(-1).t);
    return {
      initial,
      entering,
      corner: removed.data.target,
      empty,
      beforeRelease,
      end,
      reset: at(0),
    };
  });
  expect(result.initial.carX).toBeLessThan(-10);
  expect(result.entering.carX).toBeLessThan(0);
  expect(result.entering.carX).toBeGreaterThan(-10);
  expect(
    Object.entries(result.entering.crew).filter(
      ([id, w]) =>
        Math.hypot(
          ...w.position.map((x, i) => x - result.initial.crew[id].position[i]),
        ) > 0.2,
    ).length,
  ).toBeGreaterThan(1);
  expect(result.empty.wheels[result.corner]).toEqual({
    old: false,
    fresh: false,
  });
  expect(result.empty.frontHeight).toBeGreaterThan(0.2);
  expect(result.beforeRelease.releaseSign).toBe("STOP");
  expect(result.end.releaseSign).toBe("GO");
  expect(result.end.carX).toBeGreaterThan(18);
  expect(Object.values(result.end.wheels).every((w) => w.fresh && !w.old)).toBe(
    true,
  );
  expect(result.reset).toEqual(result.initial);
  expect(errors).toEqual([]);
  await expect(page.locator("#error")).toBeHidden();
});

test("timers freeze at release, seek reproducibly, and fit the sub-banner", async ({
  page,
}) => {
  const read = async (time) => {
    await page.evaluate((t) => window.pitStop.seek(t), time);
    return page.locator("#timers").textContent();
  };
  const times = await page.evaluate(() => {
    const frames = window.pitStop.playback.frames;
    return {
      release: frames.find((f) => f.name === "pit_released").t,
      end: frames.at(-1).t,
    };
  });
  const end = await read(times.end);
  expect(await read(times.release)).toBe(end);
  expect(await read(0)).toMatch(/Total\s*0.0s/);
  expect(await read(times.end)).toBe(end);
  for (const viewport of [
    { width: 800, height: 450 },
    { width: 390, height: 700 },
  ]) {
    await page.setViewportSize(viewport);
    const scene = await page.locator("#scene").boundingBox();
    const hud = await page.locator(".status").boundingBox();
    expect(hud.y).toBeGreaterThanOrEqual(scene.y + scene.height - 1);
    expect(
      await page
        .locator(".status")
        .evaluate((e) => e.scrollWidth <= innerWidth),
    ).toBe(true);
    await expect(page.locator("#timers")).toBeVisible();
  }
});

test("all equipment stays visible and continuous through handoffs", async ({
  page,
}) => {
  const transfers = await page.evaluate(() => {
    const pit = window.pitStop;
    return pit.playback.frames
      .filter((f) => f.name === "crew_transfer")
      .map((frame) => {
        pit.seek(frame.t - 0.00001);
        const before = pit.inspect().items[frame.data.item];
        pit.seek(frame.t + 0.00001);
        const after = pit.inspect().items[frame.data.item];
        return {
          before,
          after,
          count: Object.keys(pit.inspect().items).length,
          inventory: Object.keys(pit.inspect().state.items).length,
        };
      });
  });
  expect(transfers.length).toBeGreaterThanOrEqual(28);
  for (const { before, after, count, inventory } of transfers) {
    expect(count).toBe(inventory);
    expect(before.visible && after.visible).toBe(true);
    expect(
      Math.hypot(...before.position.map((p, i) => p - after.position[i])),
    ).toBeLessThan(0.02);
  }
});

test("walk and run produce different strides along the recorded path", async ({
  page,
}) => {
  const poses = await page.evaluate(() => {
    const pit = window.pitStop;
    const start = pit.playback.frames.find(
      (frame) =>
        frame.name === "crew_task_started" && frame.data.task === "move",
    );
    const index = start.data.phases.findIndex((phase) => phase.kind === "move");
    const phase = start.data.phases[index];
    const began = pit.playback.frames.find(
      (frame) =>
        frame.name === "crew_task_phase" &&
        frame.n > start.n &&
        frame.data.actor === start.data.actor &&
        frame.data.phase === index,
    );
    const time = began.t + phase.duration * 0.4;
    const original = phase.pace;
    const pose = (pace) => {
      phase.pace = pace;
      pit.seek(time);
      return pit.inspect().crew[start.data.actor];
    };
    const walk = pose("walk"),
      run = pose("run");
    phase.pace = original;
    return { walk, run };
  });
  expect(poses.walk.position).toEqual(poses.run.position);
  expect(Math.abs(poses.walk.legs[0] - poses.run.legs[0])).toBeGreaterThan(
    0.05,
  );
});

test("work diagrams follow task targets rather than worker identities", async ({
  page,
}) => {
  const removed = await page.evaluate(() => {
    const pit = window.pitStop;
    const frame = pit.playback.frames.find(
      (f) => f.name === "crew_work" && f.data.work === "remove",
    );
    pit.seek(frame.t);
    return frame.data.target;
  });
  await expect(
    page.locator(`#service-car [data-corner="${removed}"]`),
  ).toHaveAttribute("data-status", "empty");
  await expect(page.locator("#service-side")).toHaveAttribute(
    "data-lifted",
    "true",
  );
  await page.evaluate(() =>
    window.pitStop.seek(window.pitStop.playback.frames.at(-1).t),
  );
  await expect(page.locator("#service-side")).toHaveAttribute(
    "data-lifted",
    "false",
  );
  await expect(
    page.locator('#service-car [data-status="secured"]'),
  ).toHaveCount(4);
});

test("pause, replay, and reduced motion preserve the selected simulation time", async ({
  page,
}) => {
  await page.evaluate(() => {
    window.pitStop.playback.paused = false;
  });
  await page.keyboard.press("Space");
  const time = await page.evaluate(() => window.pitStop.playback.time);
  await page.waitForTimeout(100);
  expect(await page.evaluate(() => window.pitStop.playback.time)).toBe(time);
  await page.keyboard.press("r");
  expect(await page.evaluate(() => window.pitStop.playback.time)).toBe(0);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.reload();
  await page.waitForFunction(() => window.pitStop);
  expect(await page.evaluate(() => window.pitStop.playback.paused)).toBe(true);
  expect((await page.evaluate(() => window.pitStop.inspect())).state.car).toBe(
    "stopped",
  );
});

for (const fixture of ["legacy", "legacy-v3"]) {
  test(`${fixture} retains equipment animation and complete departure`, async ({
    page,
  }) => {
    const { readFileSync } = await import("node:fs");
    const frames = JSON.parse(
      readFileSync(
        new URL(`fixtures/${fixture}.json`, import.meta.url),
        "utf8",
      ),
    );
    await page.route("**/api/run", (route) =>
      route.fulfill({ json: { mode: "replay", frames } }),
    );
    await page.reload();
    await page.waitForFunction(() => window.pitStop);
    const result = await page.evaluate(() => {
      const pit = window.pitStop;
      const collect = pit.playback.frames.find(
        (f) => f.name === "crew_task_completed" && f.data.task === "collect",
      );
      pit.seek(collect.t);
      const equipped = pit.inspect().crew[collect.data.actor].tool;
      pit.seek(pit.playback.frames.at(-1).t);
      return { equipped, end: pit.inspect() };
    });
    expect(result.equipped).toBe(true);
    expect(result.end.carX).toBeGreaterThan(18);
    await expect(page.locator("#error")).toBeHidden();
  });
}

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

test("title and green timer cells follow milestones on desktop, mobile, and HOLD", async ({
  page,
}) => {
  for (const viewport of [
    { width: 1440, height: 900 },
    { width: 390, height: 700 },
  ]) {
    await page.setViewportSize(viewport);
    await page.evaluate(() => window.pitStop.seek(0));
    await expect(page.locator('#timers [data-completed="true"]')).toHaveCount(
      0,
    );
    const cells = await page.locator("#timers > div").evaluateAll((cells) =>
      cells.map((cell) => {
        const { x, y, width } = cell.getBoundingClientRect();
        return { x, y, width };
      }),
    );
    expect(cells[0].y).toBe(cells[1].y);
    expect(cells[2].y).toBe(cells[3].y);
    expect(cells[2].x).toBe(cells[0].x);
    expect(cells[1].x).toBeGreaterThan(cells[0].x + cells[0].width);
    await page.evaluate(() =>
      window.pitStop.seek(window.pitStop.playback.duration),
    );
    await expect(page.locator('#timers [data-completed="true"]')).toHaveCount(
      4,
    );
    await expect(page.locator("#phase")).toHaveText("car_departed");
  }
  const frozen = await page.evaluate(() => {
    const pit = window.pitStop;
    const service = pit.playback.frames.find(
      (frame) => frame.name === "pit_service_completed",
    );
    pit.playback.frames = pit.playback.frames.slice(0, service.n + 1);
    pit.playback.append([
      {
        n: service.n + 1,
        t: service.t,
        name: "pit_held",
        data: { state: { ...service.data.state, held: "Test HOLD" } },
      },
    ]);
    pit.seek(service.t);
    const content = document.querySelector("#timers").textContent;
    pit.seek(service.t + 100);
    return content;
  });
  await expect(page.locator("#timers")).toHaveText(frozen);
  await expect(page.locator("#phase")).toHaveText("pit_held");
  await expect(page.locator('#timers [data-completed="true"]')).toHaveCount(2);
  await expect(page.locator('[data-timer="service"] output')).toHaveCSS(
    "color",
    "rgb(164, 216, 149)",
  );
  await expect(page.locator('[data-timer="service"]')).toHaveCSS(
    "border-top-color",
    "rgba(147, 201, 130, 0.6)",
  );
});

test("version four retains its fixed jacks and recorded equipment", async ({
  page,
}) => {
  const { readFileSync } = await import("node:fs");
  const frames = JSON.parse(
    readFileSync(new URL("fixtures/legacy-v4.json", import.meta.url), "utf8"),
  );
  await page.route("**/api/run", (route) =>
    route.fulfill({ json: { mode: "replay", frames } }),
  );
  await page.reload();
  await page.waitForFunction(() => window.pitStop);
  const end = await page.evaluate(() => {
    const pit = window.pitStop;
    pit.seek(pit.playback.duration);
    return pit.inspect();
  });
  expect(end.carX).toBeGreaterThan(18);
  expect(Object.keys(end.items)).toHaveLength(
    Object.keys(frames[0].data.state.items).length,
  );
  await expect(page.locator("#error")).toBeHidden();
});

test("jacks roll with their owners, engage at the car, and return off the road", async ({
  page,
}) => {
  const result = await page.evaluate(() => {
    const pit = window.pitStop;
    const frames = pit.playback.frames;
    const jacks = Object.entries(frames[0].data.state.items).filter(
      ([, item]) => item.kind === "jack",
    );
    const poses = jacks.map(([id, item]) => {
      const mounted = frames.find(
        (frame) =>
          frame.name === "crew_transfer" &&
          frame.data.item === id &&
          frame.data.to.startsWith("mount:"),
      );
      pit.seek(mounted.t);
      const engaged = pit.inspect().items[id].position;
      const lower = frames.find(
        (frame) =>
          frame.name === "crew_work" &&
          frame.data.work === "lower" &&
          frame.data.target === item.end,
      );
      pit.seek(lower.t);
      const down = pit.inspect().state.items[id].owner;
      const moves = frames.filter(
        (frame) =>
          frame.name === "crew_task_started" &&
          frame.data.task === "move" &&
          frame.data.state.crew[frame.data.actor].equipment === id,
      );
      const rolling = moves.map((frame) => {
        pit.seek(frame.t + frame.data.duration / 2);
        const state = pit.inspect();
        return {
          jack: state.items[id].position,
          worker: state.crew[frame.data.actor].position,
          lift: Number(
            document.querySelector("#service-side").dataset[item.end],
          ),
        };
      });
      pit.seek(pit.playback.duration);
      const parked = pit.inspect().items[id].position;
      pit.seek(0);
      return {
        engaged,
        down,
        rolling,
        parked,
        reset: pit.inspect().items[id].position,
        storage: pit.playback.frames[0].data.layout.slots[item.storage],
      };
    });
    return {
      poses,
      expected: jacks.length,
      offset:
        -frames[0].data.layout.jack.handle[0] +
        frames[0].data.layout.jack.grip_forward,
    };
  });
  expect(result.expected).toBe(2);
  for (const pose of result.poses) {
    expect(Math.abs(pose.engaged[0])).toBeCloseTo(3.5);
    expect(pose.engaged[2]).toBeCloseTo(0);
    expect(pose.down).toMatch(/^mount:/);
    expect(pose.rolling.length).toBeGreaterThanOrEqual(2);
    for (const { jack, worker, lift } of pose.rolling) {
      expect(lift).toBe(0);
      expect(jack[1]).toBeCloseTo(0);
      expect(Math.hypot(jack[0] - worker[0], jack[2] - worker[1])).toBeCloseTo(
        result.offset,
      );
    }
    expect(pose.parked).toEqual(pose.storage);
    expect(pose.reset).toEqual(pose.storage);
    expect(Math.abs(pose.parked[2])).toBeGreaterThan(3);
  }
});

test("the title belongs to the illustrations and all parked crew remain visible", async ({
  page,
}) => {
  for (const viewport of [
    { width: 1440, height: 900 },
    { width: 800, height: 450 },
    { width: 390, height: 700 },
  ]) {
    await page.setViewportSize(viewport);
    for (const end of [false, true]) {
      const crew = await page.evaluate((end) => {
        const pit = window.pitStop;
        pit.seek(end ? pit.playback.duration : 0);
        return pit.inspect().crew;
      }, end);
      expect(Object.keys(crew)).toHaveLength(19);
      for (const worker of Object.values(crew)) {
        expect(worker.bounds.left).toBeGreaterThan(-1);
        expect(worker.bounds.right).toBeLessThan(1);
        expect(worker.bounds.bottom).toBeGreaterThan(-1);
        expect(worker.bounds.top).toBeLessThan(1);
      }
    }
    const title = await page.locator("#phase").boundingBox();
    const diagrams = await page.locator(".service-views").boundingBox();
    const timers = await page.locator("#timers").boundingBox();
    expect(title.y + title.height).toBeLessThanOrEqual(diagrams.y);
    expect(
      Math.abs(title.x + title.width / 2 - diagrams.x - diagrams.width / 2),
    ).toBeLessThan(1);
    if (viewport.width > 550)
      expect(timers.x + timers.width).toBeLessThanOrEqual(title.x);
    else expect(timers.y + timers.height).toBeLessThanOrEqual(title.y);
  }
});

test("waiting workers face their hubs and the Chief faces the driver", async ({
  page,
}) => {
  const results = await page.evaluate(() => {
    const pit = window.pitStop;
    const frames = pit.playback.frames;
    const reports = frames.filter(
      (f) =>
        f.name === "pit_report" &&
        f.data.step === "prepare" &&
        (f.data.actor === "chief" || f.data.actor.startsWith("wheel-on-")),
    );
    return reports.map((frame) => {
      pit.seek(frame.t);
      const worker = pit.inspect().crew[frame.data.actor];
      const target =
        frame.data.actor === "chief"
          ? [0, 0]
          : frames[0].data.layout.corners[frame.data.target];
      const delta = target.map((v, i) => v - worker.position[i]);
      return (
        (Math.sin(worker.heading) * delta[0] +
          Math.cos(worker.heading) * delta[1]) /
        Math.hypot(...delta)
      );
    });
  });
  expect(results).toHaveLength(5);
  for (const alignment of results) expect(alignment).toBeGreaterThan(0.999);
});

test("engaged jack handles remain in front of the operator", async ({
  page,
}) => {
  const grips = await page.evaluate(() => {
    const pit = window.pitStop;
    return pit.playback.frames
      .filter(
        (f) =>
          f.name === "crew_work" && ["lift", "lower"].includes(f.data.work),
      )
      .map((frame) => {
        pit.seek(frame.t);
        const scene = pit.inspect(),
          worker = scene.crew[frame.data.actor];
        const grip = scene.items[`jack-${frame.data.target}`].grip;
        const dx = grip[0] - worker.position[0],
          dz = grip[2] - worker.position[1];
        return {
          forward:
            dx * Math.sin(worker.heading) + dz * Math.cos(worker.heading),
          side: dx * Math.cos(worker.heading) - dz * Math.sin(worker.heading),
        };
      });
  });
  expect(grips).toHaveLength(4);
  for (const grip of grips) {
    expect(grip.forward).toBeGreaterThan(0.3);
    expect(Math.abs(grip.side)).toBeLessThan(0.01);
  }
});

test("version five retains its original jack transport offset", async ({
  page,
}) => {
  const { readFileSync } = await import("node:fs");
  const frames = JSON.parse(
    readFileSync(new URL("fixtures/legacy-v5.json", import.meta.url), "utf8"),
  );
  await page.route("**/api/run", (route) =>
    route.fulfill({ json: { mode: "replay", frames } }),
  );
  await page.reload();
  await page.waitForFunction(() => window.pitStop);
  const distance = await page.evaluate(() => {
    const pit = window.pitStop;
    const move = pit.playback.frames.find(
      (f) =>
        f.name === "crew_task_started" &&
        f.data.task === "move" &&
        f.data.state.crew[f.data.actor].equipment?.startsWith("jack-"),
    );
    pit.seek(move.t + move.data.duration / 2);
    const state = pit.inspect(),
      worker = state.crew[move.data.actor],
      item = state.items[move.data.state.crew[move.data.actor].equipment];
    return Math.hypot(
      item.position[0] - worker.position[0],
      item.position[2] - worker.position[1],
    );
  });
  expect(distance).toBeCloseTo(0.65);
});
