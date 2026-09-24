import { defineConfig } from "@playwright/test";

const port = Number(process.env.PIT_STOP_TEST_PORT ?? 8423);
const baseURL = `http://127.0.0.1:${port}`;

export default defineConfig({
  testDir: "tests",
  testMatch: "browser.spec.js",
  workers: 1,
  use: {
    baseURL,
    viewport: { width: 1440, height: 900 },
  },
  projects: [
    { name: "chromium", use: { browserName: "chromium" } },
    { name: "webkit", use: { browserName: "webkit" } },
  ],
  webServer: {
    command: `uv run main.py --no-browser --port ${port}`,
    url: baseURL,
    reuseExistingServer: true,
  },
});
