# Pit Stop

Nineteen agentwerk agents complete 52 actions to service a racing car in a 3D
browser scene. The car leaves only after the chief authorizes release.

## Run the recording

Requires Python 3.10+, [uv](https://docs.astral.sh/uv/), Rust, and Node.js 22.12+.
The first sync builds the Python extension.

```sh
cd crates/agentwerk-py/examples/pit_stop
uv sync --frozen
npm ci
npm run build
uv run main.py
```

Open `http://127.0.0.1:8423`. Playback needs no credentials or network after setup.
Press **Space** to pause/resume or **R** to replay. Reduced motion starts paused.

Real agents using `qwen3-coder-next` produced the recording.
Its approximately 70-second run plays in a 12-second loop with a reset interval;
uniform acceleration preserves ordering and concurrency.
This is an illustrative mechanical simulation, not a measured Formula 1 stop.

## Run live agents

For an OpenAI-compatible endpoint, set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and
`OPENAI_MODEL` or `MODEL`. Set `LITELLM_PROVIDER=openai` if other provider keys
are present.

```sh
uv run main.py --live --seed 42
```

With optional Sigrid shell integration:

```sh
sigrid && MODEL=qwen3-coder-next LITELLM_PROVIDER=openai uv run main.py --live
```

Credentials stay in the host process; the browser and recording receive only simulation state and selected
lifecycle events. Live runs stop after one departure. The viewer stays open
until Ctrl-C, and replay never makes model calls.

Live events default to `.agentwerk/pit-stop.jsonl`:

```sh
uv run main.py --replay .agentwerk/pit-stop.jsonl
uv run main.py --live --record-only --record .agentwerk/new-stop.jsonl
```

`--seed` fixes motion and work timing, not model or network timing. Omit it for
a fresh seed. Metadata records the seed and configured model; only recordings
reproduce runs exactly. Older recordings retain their original motion.
Use `--port 8424` or `--no-browser` to change serving behavior.
Failed work or obstructed routes hold the car.

## Read the example

Start at [main.py](main.py); [orchestration.py](orchestration.py) configures the
agents, tools, tasks, and handoffs. Each agent calls `perform` once for its assigned
action. Agents use host completion (`Agent.interactive()`), so their only tool is
`perform`. A validated tool result completes the task through Werk; no additional
model turn or `finish` call is needed. One-shot Conditions create dependent tasks
from mechanical readiness events, and Werk runs independent crew members in parallel.

Mechanical state and active routes are projections of Werk's ordered event log.
The reducer records completed phases, transfers, mechanical effects, and holds;
`PitStop.rebuild()` reconstructs the same projection from that log. Live sessions
are saved under `.agentwerk/pit-stop-<id>/`. The browser receives derived snapshots
and recorded paths, so it cannot change the car or authorize departure.

Crew follow curved, eased paths around the cupboards and tire platforms. Tools
lie on the cupboard surfaces and rotate through handoffs. Fresh and used tires
share two platforms: an old tire returns to the position vacated by its replacement.

One Werk runs all 52 tasks under a 300-second, 160-turn limit and Policy token
limits. Mechanical effects and ownership changes are validated by the host;
rendering and task completion claims cannot authorize them. All 52 actions,
lowered jacks, cleared crew, and the chief's release are required for departure.

No external models, fonts, or CDN assets load at runtime. The bundled Silkscreen
font is OFL-licensed; its license ships in `public/licenses/`.

## Develop and verify

Run `uv run main.py --no-browser`, then `npm run dev`. Vite proxies requests to Python.

```sh
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
npm test
npx playwright install chromium webkit
npm run test:browser
npx prettier --check 'src/*.{js,css}' 'tests/*.js' '*.js' '*.mjs' index.html
```

Set `PIT_STOP_TEST_PORT` if 8423 is occupied. Tests check service prerequisites,
failure holds, and playback in Chromium and WebKit. No tests make external model calls.

## Capture the README GIF

With the built viewer running at port 8423:

```sh
npm run capture -- ../../../../assets/demo.gif
```

Capture renders 180 frames at 800×450 for a 12-second loop. It tries 15 fps with
smaller palettes, then 12 fps to stay below **2,000,000 bytes**. The current GIF
uses 15 fps and approximately 1.88 MB. Oversized output never replaces the existing GIF.

FFmpeg comes from the locked `imageio-ffmpeg` dependency. Set `PIT_STOP_URL` for
another viewer. Screenshots and GIFs without an output path go under `.context/`.
