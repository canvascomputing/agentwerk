# Pit Stop

Nineteen agentwerk agents service a racing car in a 3D browser scene: 4 gunners, 4 wheel-off operators, 4 wheel-on operators, 2 jack operators, 2 steadiers, 2 wing mechanics, and 1 chief. They choose equipment, travel pace, and physical tasks. The Chief Mechanic reviews their `finish` reports before deciding GO or HOLD.

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
The crew starts and returns on the upper banner and between the lower equipment
stations, with the whole group kept in view.

The bundled recording comes from real agents. Its simulation timeline plays in a
12-second loop with a reset interval; uniform acceleration preserves ordering and
concurrency. This is an illustrative simulation, not measured Formula 1 performance.

## Run live agents

For an OpenAI-compatible endpoint, set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and
`OPENAI_MODEL` or `MODEL`. Set `LITELLM_PROVIDER=openai` if other provider keys
are present.

```sh
uv run main.py --live --seed 42
```

With optional Sigrid shell integration:

```sh
sigrid && MODEL=qwen3.8-flash-next LITELLM_PROVIDER=openai uv run main.py --live
```

Credentials stay in the host process; the browser and recording receive only simulation state and selected
lifecycle events. Live runs stop after one departure. The viewer stays open
until Ctrl-C, and replay never makes model calls.

Live events default to `.agentwerk/pit-stop.jsonl`:

```sh
uv run main.py --replay .agentwerk/pit-stop.jsonl
uv run main.py --live --record-only --record .agentwerk/new-stop.jsonl
```

`--seed` fixes assignments, equipment placement, arrival timing, and wing targets,
not agent decisions or model timing. Omit it for
a fresh seed. Metadata records the seed and configured model; only recordings
reproduce runs exactly. Older recordings retain their original motion.
Use `--port 8424` or `--no-browser` to change serving behavior.
Failed work or obstructed routes hold the car.

## Read the example

Start at [main.py](main.py). [orchestration.py](orchestration.py) creates one Werk,
nineteen agents, task result schemas, one-shot Conditions, and result hooks.
Crew members receive objectives and observations rather than prescribed tool calls.

- `move(destination, pace)` walks or runs to a named location. The planner records
  curved paths, travel pace, arrival turns, and yielding reservations. Crew members face
  their assigned work while waiting; parked crew face the car.
- `operate(task, ...)` picks up, drops, or uses equipment at the crew member's actual
  position. Wrong equipment can be picked up; incompatible use is rejected without
  mechanical changes. Every response supplies fresh observations for recovery.
- `finish({"status":"completed"})` reports completion; use `"blocked"` when stuck. The host records physical evidence and checks
  its location, inventory, and completed work before opening dependent tasks.

The Chief, jack operators, and steadiers approach first. Gunners and wing mechanics
join once the car is stable; wheel-off and wheel-on crew members follow their corner's
validated handoffs. Later crew members prepare at holding positions away from the car.
Tasks include who is waiting and estimated walking/running times.

Steadiers report clear from nearby safe positions so lowering can start while they
walk back to parking. Jack operators stand behind the handles, back their lowered
jacks clear of the car, and roll them into storage beside the equipment stations. Equipment returns can overlap other service.

The Chief holds the STOP board in front of the car, reviews every report, steps
aside, and calls `finish(decision, reviewed_tasks, reason)`. GO requires verified
reports, requested service, lowered jacks, stored equipment, and all crew clear.
The review task reads statuses with `{{ find_results(task.status = finished)[*].status }}`
and verified reports with `{{ find_events(event.name = pit_report)[*].data }}`.
A claim cannot change mechanical state. Blocked or contradictory reports prevent GO.

The prompts live in [prompts/](prompts/). Crew members decide whether time permits
walking, which inventory item fits the task, where to stage, and how to recover
from a rejected call. `storage` in an inventory item names its original storage
slot; tools and tires may use another compatible empty slot. Jacks return to their
designated slots. `busy_destinations` identifies
positions occupied or reserved by other crew members.

### Custom events

Observe the pit stop using the same `werk.on_event` API as agent lifecycle events:

```python
def log_pit_stop(_, event):
    data = event.get_data()

    match event.get_name():
        case "car_approaching":
            print(f"Car arrives in {data['arrives_in_seconds']}s.")
        case "pit_service_completed":
            print("Service complete.")
        case "pit_released":
            print("GO.")
        case "pit_held":
            print(f"HOLD: {data['message']}")


werk.on_event(log_pit_stop)
```

`car_approaching` opens preparation and supplies `arrives_in_seconds`, the time
until the car stops in the pit box. `car_arriving` and `car_stopped` follow.
`pit_prepared` records all validated preparation reports. Service emits
`pit_service_started` and `pit_service_completed`; physical clearance emits
`pit_crew_clear`. The Chief's accepted verdict emits `pit_released` or `pit_held`.
A released car emits `car_departing` and `car_departed`. Milestones occur once;
`pit_prepared` can occur after arrival if the crew is late.

Physical crew events, reports, and clock changes also pass through Werk. State and
active routes rebuild from its ordered log with `PitStop.rebuild()`. Sessions are
saved under `.agentwerk/pit-stop-<id>/`. The browser only observes snapshots,
recorded paths, and events; it cannot authorize release.

### Simulation time

[sim_clock.py](sim_clock.py) advances all physical work together and pauses while
runnable agents decide. Arrival has a seeded deadline on that clock and does not
wait for crew readiness. The live scene pauses for decisions; replay omits their
wall-clock latency. Version 6 records arrival turns, visible parking positions,
and corrected jack grips and clearance. Earlier recordings retain their original
motion and equipment poses.

The title above the car illustrations shows the latest custom milestone. The
separate 2×2 timer grid shows:

- Preparation: approach notification until the car stops.
- Service: car stopped until the wheels and wings meet their targets.
- Clearance: service complete until the Chief's GO is accepted.
- Total: approach notification until accepted GO.

Completed timers freeze and turn green. HOLD freezes all timers and preserves
only the completed phases in green. Pause, seek, and replay
use the same recorded times. Policy still uses wall time: 900 seconds, 1,000 turns,
1,000,000 input tokens, and bounded request retries. Recoverable tool errors return
to the agent; terminal failures and policy exhaustion hold the car.

No external models, fonts, or CDN assets load during replay. The bundled Silkscreen
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
smaller palettes, then 12 fps to stay below **2,000,000 bytes**. Oversized output never replaces the existing GIF.

FFmpeg comes from the locked `imageio-ffmpeg` dependency. Set `PIT_STOP_URL` for
another viewer. Screenshots and GIFs without an output path go under `.context/`.
