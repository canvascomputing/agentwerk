"""Exercise agentwerk against a local provider; only model output is scripted."""

import asyncio
import json
import re
import time

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from feed import Feed
from orchestration import build_crew, run_stop
from simulation import PitStop


def scripted_provider(skip_work, wrong_action=False):
    async def respond(request):
        body = await request.json()
        assert {entry["function"]["name"] for entry in body["tools"]} == {"perform"}
        content = next(
            message["content"]
            for message in body["messages"]
            if message["role"] == "user"
        )
        if isinstance(content, list):
            content = " ".join(part.get("text", "") for part in content)
        action = re.search(r'"action"\s*:\s*"([^"\n]+)"', content).group(1)
        if wrong_action and '"chief"' in content:
            action = "release"
        name = "finish" if skip_work else "perform"
        arguments = {"action": action}
        if skip_work:
            actor = re.search(r'"actor"\s*:\s*"([^"\n]+)"', content).group(1)
            arguments.update(actor=actor, completed=True)
        chunk = {
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        return web.Response(
            text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
            content_type="text/event-stream",
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", respond)
    return app


def configure_provider(monkeypatch, server):
    monkeypatch.setenv("LITELLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", str(server.make_url("/")).rstrip("/"))
    monkeypatch.setenv("MODEL", "test")


@pytest.mark.parametrize("skip_work", [False, True])
async def test_conditions_require_mechanical_work_and_create_each_action_once(
    monkeypatch, skip_work, assert_rebuilt
):
    app = scripted_provider(skip_work)
    async with TestServer(app) as server:
        configure_provider(monkeypatch, server)
        prepared = asyncio.Event()
        collections = set()
        loop = asyncio.get_running_loop()

        def observe(name, data):
            if name == "action_completed" and data["action"] == "collect":
                collections.add(data["actor"])
                if len(collections) == 10:
                    loop.call_soon_threadsafe(prepared.set)

        pit = PitStop(observe, sleep=lambda _: None)
        werk = build_crew(pit)
        werk.start()
        if not skip_work:
            await asyncio.wait_for(prepared.wait(), timeout=5)
            assert pit.snapshot()["car"] == "approaching"
            assert not pit.eligible()
        pit.arrive()
        await werk.finish()
        state = pit.snapshot()
        if skip_work:
            assert state["held"]
            assert state["car"] in ("approaching", "stopped")
            assert all(not worker["done"] for worker in state["crew"].values())
        else:
            assert state["car"] == "released"
            assert len(werk.find_tasks("task.status = finished")) == 52
            assert len(werk.find_events("event.name = request_started")) == 52
            assert_rebuilt(pit)
            assert not pit.active


async def test_live_metadata_records_the_model_selected_by_agentwerk(monkeypatch):
    async with TestServer(scripted_provider(False)) as server:
        configure_provider(monkeypatch, server)
        monkeypatch.delenv("MODEL")
        monkeypatch.setenv("OPENAI_MODEL", "provider-selected-model")
        original_pit_stop = PitStop
        monkeypatch.setattr(
            "orchestration.PitStop",
            lambda seed, werk: original_pit_stop(
                sleep=lambda duration: time.sleep(min(duration, 0.003)),
                seed=seed,
                werk=werk,
            ),
        )

        feed = Feed()
        assert await run_stop(feed, seed=42)
        assert feed.frames[0]["name"] == "run_metadata"
        assert feed.frames[0]["data"]["model"] == "provider-selected-model"
        active = set()
        peak = 0
        for frame in feed.frames:
            if frame["name"] == "action_started":
                active.add(frame["data"]["actor"])
                peak = max(peak, len(active))
            elif frame["name"] == "action_completed":
                active.remove(frame["data"]["actor"])
        assert peak > 1


async def test_perform_rejects_an_allowed_action_that_was_not_assigned(monkeypatch):
    async with TestServer(scripted_provider(False, wrong_action=True)) as server:
        configure_provider(monkeypatch, server)
        pit = PitStop(sleep=lambda _: None, seed=42)
        werk = build_crew(pit)
        await werk.finish()
        state = pit.snapshot()
        assert "outside its assigned task" in state["held"]
        assert state["car"] == "approaching"
        assert state["crew"]["chief"]["done"] == []
