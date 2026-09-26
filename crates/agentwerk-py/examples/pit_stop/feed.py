"""Record public simulation events and serve them to local browsers."""

import asyncio
import json
import time
from pathlib import Path
from threading import RLock

from aiohttp import web


class Feed:
    def __init__(self, record_file=None):
        self.frames = []
        self.clock = None
        self.started = time.monotonic()
        self.lock = RLock()
        self.record = Path(record_file) if record_file else None
        if record_file:
            self.record.parent.mkdir(parents=True, exist_ok=True)
            self.record.write_text("")

    def push(self, name, data):
        with self.lock:
            frame = {
                "n": len(self.frames),
                "t": self.clock.seconds
                if self.clock
                else time.monotonic() - self.started,
                "name": name,
                "data": data,
            }
            self.frames.append(frame)
            if self.record:
                with self.record.open("a") as record:
                    record.write(json.dumps(frame, separators=(",", ":")) + "\n")

    def after(self, number):
        with self.lock:
            return list(self.frames[number + 1 :])


def read_recording(path):
    frames = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    if not frames or frames[0].get("name") != "run_metadata":
        raise ValueError("Recording must start with run_metadata")
    previous = -1
    for number, frame in enumerate(frames):
        if (
            frame.get("n") != number
            or not isinstance(frame.get("t"), (float, int))
            or frame["t"] < previous
        ):
            raise ValueError(
                "Recording events must have ordered sequence numbers and timestamps"
            )
        previous = frame["t"]
    return frames


def application(feed, recording, dist):
    async def config(_):
        return web.json_response(
            {
                "mode": "live" if feed else "replay",
                "frames": feed.after(-1) if feed else recording,
                "elapsed": (
                    feed.clock.seconds
                    if feed.clock
                    else time.monotonic() - feed.started
                )
                if feed
                else 0,
                "running": feed.clock.running if feed and feed.clock else True,
            }
        )

    async def events(request):
        if feed is None:
            raise web.HTTPNotFound()
        response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        )
        await response.prepare(request)
        try:
            number = int(request.headers.get("Last-Event-ID", "-1"))
        except ValueError:
            number = -1
        try:
            while True:
                for frame in feed.after(max(-1, number)):
                    await response.write(
                        f"id: {frame['n']}\ndata: {json.dumps(frame)}\n\n".encode()
                    )
                    number = frame["n"]
                await response.write(b": heartbeat\n\n")
                await asyncio.sleep(0.25)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return response

    async def index(_):
        return web.FileResponse(dist / "index.html")

    app = web.Application()
    app.router.add_get("/api/run", config)
    app.router.add_get("/events", events)
    app.router.add_get("/", index)
    app.router.add_static("/", dist)
    return app
