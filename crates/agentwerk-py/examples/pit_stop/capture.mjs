/** Capture the real renderer at its playback speed within the README budget. */
import { chromium } from "@playwright/test";
import { mkdir, mkdtemp, rename, rm, stat, writeFile } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { REPLAY_SECONDS } from "./src/playback.js";

const root = fileURLToPath(new URL("../../../../", import.meta.url));
const output = resolve(process.argv[2] ?? `${root}/.context/pit-stop.gif`);
const screenshots = resolve(root, ".context");
await mkdir(screenshots, { recursive: true });
const framesDir = await mkdtemp(resolve(screenshots, "pit-capture-"));
const temporary = resolve(framesDir, "preview.gif");
const frameCount = REPLAY_SECONDS * 15;
const ffmpeg = execFileSync(
  "uv",
  [
    "run",
    "python",
    "-c",
    "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())",
  ],
  { encoding: "utf8" },
).trim();
const browser = await chromium.launch();
const page = await browser.newPage({
  viewport: { width: 800, height: 450 },
  deviceScaleFactor: 1,
});
try {
  await page.goto(process.env.PIT_STOP_URL ?? "http://127.0.0.1:8423");
  await page.waitForFunction(() => window.pitStop?.playback.frames.length > 0);
  await page.evaluate(() => document.fonts.ready);
  const { duration, frames } = await page.evaluate(() => ({
    duration: window.pitStop.playback.duration,
    frames: window.pitStop.playback.frames,
  }));
  for (let index = 0; index < frameCount; index++) {
    await page.evaluate(
      (time) => window.pitStop.seek(time),
      (duration * index) / frameCount,
    );
    await writeFile(
      resolve(framesDir, `${String(index).padStart(4, "0")}.png`),
      await page.screenshot(),
    );
    if (index % 60 === 0) console.log(`Captured ${index}/${frameCount} frames`);
  }
  let encoded = false;
  for (const [fps, colors] of [
    [15, 192],
    [15, 128],
    [15, 96],
    [12, 96],
    [12, 64],
  ]) {
    execFileSync(ffmpeg, [
      "-y",
      "-loglevel",
      "error",
      "-framerate",
      "15",
      "-i",
      resolve(framesDir, "%04d.png"),
      "-filter_complex",
      `fps=${fps},split[a][b];[a]palettegen=max_colors=${colors}:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle`,
      "-loop",
      "0",
      temporary,
    ]);
    const bytes = (await stat(temporary)).size;
    console.log(`${fps} fps / ${colors} colors: ${bytes} bytes`);
    if (bytes >= 2_000_000) continue;
    await rename(temporary, output);
    encoded = true;
    break;
  }
  if (!encoded)
    throw new Error(
      "GIF exceeds 2,000,000 bytes; existing output was preserved.",
    );
  const removal = frames.find(
    (frame) => frame.name === "crew_task_phase" && frame.data.kind === "pull",
  );
  const moments = {
    service: removal?.t ?? duration / 3,
    departure: frames.find((frame) => frame.name === "car_departing").t + 1.5,
  };
  for (const [name, time] of Object.entries(moments)) {
    await page.evaluate((time) => window.pitStop.seek(time), time);
    await page.screenshot({
      path: resolve(screenshots, `pit-stop-${name}.png`),
    });
  }
  console.log(`Saved ${output} (${(await stat(output)).size} bytes)`);
} finally {
  await browser.close();
  await rm(framesDir, { recursive: true, force: true });
}
