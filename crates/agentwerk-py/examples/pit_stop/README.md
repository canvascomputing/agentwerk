# Pit Stop

Nineteen agentwerk agents service a racing car in a browser-rendered pit box.
The car arrives, stops, and leaves only after the crew completes all 52 actions
and the chief authorizes release.

## Run the recording

Requires Python 3.10+, [uv](https://docs.astral.sh/uv/), Rust, and Node.js 22.12+.
Rust builds the local agentwerk Python extension on the first sync.

```sh
cd crates/agentwerk-py/examples/pit_stop
uv sync --frozen
npm ci
npm run build
uv run main.py
```

The browser opens at `http://127.0.0.1:8423`. The default recording needs no
provider credentials or network connection after setup. Press **Space** to pause/resume or **R** to replay. Reduced-motion preferences start the scene paused.

The bundled recording was produced by real agentwerk agents using
`qwen3-coder-next` through Cortecs. Its approximately 72-second run plays in a
12-second loop, including a brief reset interval. All timestamps share one
uniform speed multiplier, preserving the recorded ordering and concurrency.
This is an illustrative mechanical simulation, not a measured Formula 1 stop.

## Run live agents

Use agentwerk's normal provider environment variables. For an OpenAI-compatible
endpoint, set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` or `MODEL`.
Set `LITELLM_PROVIDER=openai` when other provider keys are also present.

```sh
uv run main.py --live --seed 42
```

With Sigrid's shell integration installed, use its selected credentials:

```sh
sigrid && MODEL=qwen3-coder-next LITELLM_PROVIDER=openai uv run main.py --live
```

The example itself has no dependency on Sigrid. Credentials stay in the host
process; the browser and recording receive only simulation state and selected
lifecycle events. Live runs stop after one departure. The viewer stays open
until Ctrl-C, and replay never makes model calls.

Live events are saved to `.agentwerk/pit-stop.jsonl` by default:

```sh
uv run main.py --replay .agentwerk/pit-stop.jsonl
uv run main.py --live --record-only --record .agentwerk/new-stop.jsonl
```

`--seed` controls walking pace, routes, reaction delays, work durations, and the
arrival path. Omit it for a fresh seed, saved in recording metadata. The model
name in that metadata comes from agentwerk's provider configuration. Model and
network timing remain live; only a recording reproduces an entire run exactly.
Older recordings retain their original motion behavior.

Use `--port 8424` to change the port or `--no-browser` to serve without opening
a tab. A missing build produces the command needed to create it. Failed work
holds the car instead of synthesizing a successful release.

## Read the example

Start with [orchestration.py](orchestration.py). It composes one `Werk`, labeled
agents, role-specific tools, schema-bound tasks, and result hooks. Each agent
calls `perform`, checks its result, then calls `finish`. The result hook verifies
that the mechanical action really occurred before scheduling the next work.

| Component | Responsibility |
| --- | --- |
| `orchestration.py` | Agent setup, role tools, task handoffs, run limits, event observation |
| `simulation.py` | Car state, equipment possession, physical clearance, mechanical prerequisites and actions |
| `movement.py` / `layout.json` | Seeded motion, shared positions, and reservations at crossings |
| `prompts/` | Seven crew roles and the `perform` tool contract |
| `feed.py` / `main.py` | Recording, replay loading, local HTTP/SSE server and CLI |
| `src/playback.js` | Ordered event history and the shared live/replay clock |
| `src/motion.js` / `src/animation.js` | Sample recorded routes and derive physical poses from action phases |
| `src/equipment.js` / `src/character-motion.js` | Persistent items, continuous handoffs, articulated hand contact and crew poses |
| `src/hud.js` | Per-corner progress derived from recorded work |
| `src/car.js` / `src/crew.js` | Car, removable wheels, crew and jack models |
| `src/scene.js` / `src/geometry.js` | Environment, lighting, camera and geometry helpers |
| `src/main.js` | Connect playback, rendering, keyboard controls and status |

Ten collectors retrieve tools and fresh tires while the car approaches. The car
starts after two collectors begin moving, so model latency cannot leave everyone
standing still during entry. `car_stopped` unlocks lifting and bracing; both jacks
and steadiers must prepare the car before wheel service.

| Crew | Tasks per worker |
| --- | --- |
| Four gunners | Collect → loosen → tighten → return tool |
| Four wheel-off operators | Remove and step aside → store used tire |
| Four wheel-on operators | Collect and stage → fit and step aside → withdraw |
| Two wing adjusters | Collect → adjust → return tool |
| Two jack operators | Lift → lower |
| Two steadiers | Brace → clear |
| Chief | Enter with the pit board → verify readiness and release |

One Werk runs all 52 tasks. Result hooks schedule each handoff once and check the
assigned action against validated state. Fitting overlaps used-tire storage;
tightening overlaps the fitter's withdrawal. Native task events report active
agents, and simulation events travel through the same Werk observation channel.
The run has a 300-second, 160-turn limit; token limits remain bounded by Policy.
The chief enters from the apron gripping a rectangular pit board directly with
both hands. It faces the driver ahead of the nose while the car is serviced.
After the final checks, the chief gives a brief signal and retreats briskly to
the garage, lowering the board along the way. Its rim turns green and departure
is enabled only after the withdrawal finishes. The board has no camera-facing text; the chief provides
the release cue without a traffic light.

Every tool and tire has one owner: a station slot, a worker, or a hub. Recorded
reach, grip, pull, seat, and placement phases move that same item through each
handoff. Tires travel just ahead of the body with a shallow lift and hands on
the rear tread, keeping the carry clear of the torso. Mechanical effects and
ownership changes are validated by the host;
rendering and task completion claims cannot authorize them. All 52 actions,
lowered jacks, cleared crew, and the chief's release are required for departure.

Routes include seeded pace, reactions, handling duration, and smooth turns.
Three apron paths and local yields reserve space for travel and handling,
including stationary crew. An obstructed route holds the car. The renderer
follows actual phase timestamps, so delayed events cannot make equipment snap
back to its previous location.

The angular 3D crew have individual clothing, skin tones, caps, and articulated
work poses inspired by GTA 2. The car uses a red, blue, yellow, and white livery,
with repeating pixel-lettered red sponsor paint and a physical release signal.
Crew clothing varies deterministically between red, blue, yellow, black, and
white; tools, cupboards, and tire-return plates use blue. The closer fixed camera
frames tool pickup and used-tire returns on the left, fresh-tire collection on
the right, and mechanical work around the car.

Geometry and textures are authored in code. No external models, fonts, or CDN
assets are fetched at runtime. The retro HUD bundles the OFL-licensed Silkscreen
font locally. The HUD keeps two text lines: the phase and activity counts.
A compact overhead car stays fixed, highlighting tire work in amber, showing
empty hubs during replacement, and turning secured tires green. A separate side
view rises and tilts with the front and rear jacks independently. The wheels sit
at their actual corners; there are no corner abbreviations or separate meters.
A six-pixel progress bar sits at the bottom.
The borderless HUD has its own space below the scene. Open yellow rails and
alignment ticks mark the box. The car follows a gentle curve and brakes smoothly
onto the marks. The font license ships in `public/licenses/`.

## Develop and verify

Keep `uv run main.py --no-browser` running, then use `npm run dev` for frontend
editing. Vite proxies the recording and event routes to the Python server.

```sh
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
npm test
npx playwright install chromium webkit
npm run test:browser
npx prettier --check 'src/*.{js,css}' 'tests/*.js' '*.js' '*.mjs' index.html
```

Set `PIT_STOP_TEST_PORT` to a free port when another viewer is using 8423.

Python tests cover all 52 missing-action cases, concurrent service, failure holds,
seeded routes, equipment prerequisites, physical clearance, duplicate scheduling,
and false completion claims through a local provider.
Browser tests exercise real WebGL rendering in Chromium and WebKit, wheel
replacement, arrival and release, playback controls, reduced motion, resizing,
and unavailable WebGL. Tests make no external model calls.

## Capture the README GIF

With the built viewer running at port 8423:

```sh
npm run capture -- ../../../../assets/demo.gif
```

The capture command renders 180 frames at 800×450 for a 12-second loop. It tries
15 fps with progressively smaller palettes, then 12 fps if needed to stay below
**2,000,000 bytes**. The current GIF uses 15 fps and is approximately 1.86 MB.
An oversized encode never replaces the existing output.

FFmpeg comes from the locked `imageio-ffmpeg` development dependency. Review
screenshots go under the repository's `.context/` directory. Set `PIT_STOP_URL`
to capture another local viewer. Without an output argument, the GIF also goes
under `.context/`.
