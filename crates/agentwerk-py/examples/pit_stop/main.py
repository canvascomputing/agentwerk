"""Launch the pit-stop browser, or record one live agentwerk run."""

import argparse
import asyncio
import webbrowser
from pathlib import Path

from aiohttp import web

from feed import Feed, application, read_recording

HERE = Path(__file__).resolve().parent


async def main(args):
    feed = Feed(args.record) if args.live else None
    recording = None if args.live else read_recording(args.replay)
    runner = None
    try:
        if not args.record_only:
            dist = HERE / "dist"
            if not (dist / "index.html").exists():
                raise SystemExit(
                    "Build the browser first: cd "
                    + str(HERE)
                    + " && npm ci && npm run build"
                )
            runner = web.AppRunner(
                application(feed, recording, dist), shutdown_timeout=1
            )
            await runner.setup()
            await web.TCPSite(runner, "127.0.0.1", args.port).start()
            url = f"http://127.0.0.1:{args.port}"
            print(f"Pit Stop: {url}", flush=True)
            if not args.no_browser:
                webbrowser.open(url)
        if args.live:
            from orchestration import run_stop

            succeeded = await run_stop(feed, seed=args.seed)
            print(
                "Car departed." if succeeded else "Car held in the pit box.", flush=True
            )
            if args.record_only:
                return 0 if succeeded else 1
        await asyncio.Event().wait()
    finally:
        if runner:
            await runner.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--live",
        action="store_true",
        help="Run real agents using provider environment variables",
    )
    mode.add_argument(
        "--replay", type=Path, default=HERE / "recordings" / "showcase.jsonl"
    )
    parser.add_argument(
        "--record", type=Path, default=Path(".agentwerk/pit-stop.jsonl")
    )
    parser.add_argument(
        "--record-only",
        action="store_true",
        help="Run once without serving the browser (requires --live)",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed assignments, equipment, arrival timing, and service targets",
    )
    parser.add_argument("--port", type=int, default=8423)
    args = parser.parse_args()
    if args.record_only and not args.live:
        parser.error("--record-only requires --live")
    try:
        raise SystemExit(asyncio.run(main(args)))
    except KeyboardInterrupt:
        pass
