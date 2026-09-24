"""Exercise agentwerk against a local provider; only model output is scripted."""

import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from feed import Feed
from orchestration import build_crew, run_stop
from simulation import PitStop


def scripted_provider(skip_work):
    async def respond(request):
        body = await request.json()
        finish = next(
            tool["function"]
            for tool in body["tools"]
            if tool["function"]["name"] == "finish"
        )
        action = finish["parameters"]["properties"]["action"]["enum"][0]
        performed = any(message["role"] == "tool" for message in body["messages"])
        name = "finish" if performed or skip_work else "perform"
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
                                    "arguments": json.dumps({"action": action}),
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
async def test_result_hooks_require_mechanical_work_and_schedule_each_handoff_once(
    monkeypatch, skip_work
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
        werk, schedule = build_crew(pit)
        schedule()
        schedule()
        assert len(werk.find_tasks("task.status = todo")) == 11
        werk.start()
        if not skip_work:
            await asyncio.wait_for(prepared.wait(), timeout=5)
            assert pit.snapshot()["car"] == "approaching"
            assert not pit.eligible()
        pit.arrive()
        schedule()
        schedule()
        await werk.finish()
        state = pit.snapshot()
        if skip_work:
            assert state["held"]
            assert state["car"] == "stopped"
            assert all(not worker["done"] for worker in state["crew"].values())
        else:
            assert state["car"] == "released"
            assert len(werk.find_tasks("task.status = finished")) == 52


async def test_live_metadata_records_the_model_selected_by_agentwerk(monkeypatch):
    async with TestServer(scripted_provider(False)) as server:
        configure_provider(monkeypatch, server)
        monkeypatch.delenv("MODEL")
        monkeypatch.setenv("OPENAI_MODEL", "provider-selected-model")
        original_pit_stop = PitStop
        monkeypatch.setattr(
            "orchestration.PitStop",
            lambda publish, seed: original_pit_stop(
                publish, sleep=lambda _: None, seed=seed
            ),
        )

        feed = Feed()
        assert await run_stop(feed, seed=42)
        assert feed.frames[0]["name"] == "run_metadata"
        assert feed.frames[0]["data"]["model"] == "provider-selected-model"
