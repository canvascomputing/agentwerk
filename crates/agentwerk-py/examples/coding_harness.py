"""Plan, implement, and verify one repository change with two agents.

Usage: python coding_harness.py <TASK>
       python coding_harness.py --resume
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

from agentwerk import (
    Agent,
    CommandTool,
    Condition,
    EditFileTool,
    Event,
    GlobTool,
    GrepTool,
    ListDirectoryTool,
    ReadFileTool,
    Task,
    Werk,
    WriteFileTool,
)

PLAN = "plan"
CODING = "coding"
SESSION_DIR = Path("./session")
CODER_TASK = "Implement this plan:\n\n{{ find_result(plan) }}"
PROMPTS = Path(__file__).parent / "prompts"
PLANNER_ROLE = (PROMPTS / "planner.md").read_text()
CODER_ROLE = (PROMPTS / "coder.md").read_text()


def parse_args(args: List[str]) -> Optional[str]:
    if args == ["--resume"]:
        return None
    if not args or "--resume" in args:
        raise ValueError("usage: python coding_harness.py <TASK> | python coding_harness.py --resume")

    request = " ".join(args).strip()
    if not request:
        raise ValueError("the coding task must not be empty")
    return request


def open_werk(session: Path, request: Optional[str]) -> Werk:
    if request is not None and session.exists():
        raise RuntimeError(
            f"session already exists at {session}; use --resume or move it before starting another change"
        )
    if request is None and not session.exists():
        raise RuntimeError(f"no session exists at {session}")
    return Werk(str(session))


def read_tools():
    return [ListDirectoryTool(), GlobTool(), GrepTool(), ReadFileTool()]


def git_tool() -> CommandTool:
    return CommandTool("git").allow("git status*").allow("git diff*")


def planner_agent(repo: Path) -> Agent:
    return (
        Agent.from_env()
        .label(PLAN)
        .role(PLANNER_ROLE)
        .dir(str(repo))
        .tools(read_tools())
        .tool(git_tool())
    )


def coder_agent(repo: Path) -> Agent:
    cargo = (
        CommandTool("cargo")
        .allow("cargo fmt*")
        .allow("cargo check*")
        .allow("cargo test*")
    )


def coder_condition() -> Condition:
    return Condition(
        "task.label = plan AND task.status = finished"
    ).task(Task(CODER_TASK, label=CODING))
    return (
        Agent.from_env()
        .label(CODING)
        .role(CODER_ROLE)
        .dir(str(repo))
        .interactive()
        .tools(read_tools())
        .tool(EditFileTool())
        .tool(WriteFileTool())
        .tool(git_tool())
        .tool(cargo)
    )


def next_stage(werk: Werk):
    planners = werk.find_tasks(PLAN)
    if len(planners) != 1:
        raise RuntimeError("the session must contain exactly one planner task")
    planner = planners[0]
    if planner.is_failed():
        raise RuntimeError("the planner task failed")
    if not planner.is_finished():
        return PLAN, planner.get_id()

    request = planner.get_task()
    plan = planner.get_result()
    if not isinstance(request, str):
        raise RuntimeError("the planner request is not text")
    if plan is None:
        raise RuntimeError("the planner result is missing")

    coders = werk.find_tasks(CODING)
    if len(coders) > 1:
        raise RuntimeError("the session contains more than one coder task")
    if not coders:
        return "create_coder", (request, plan)

    coder = coders[0]
    if coder.is_failed():
        raise RuntimeError("the coder task failed")
    if not coder.is_finished():
        return CODING, coder.get_id()

    result = coder.get_result()
    if result is None:
        raise RuntimeError("the coder result is missing")
    return "done", result


def stream_coder_text(_, event) -> None:
    if event.get_name() == Event.TEXT_CHUNK_RECEIVED and event.get_label() == CODING:
        print(event.get_data().get("content", ""), end="", flush=True)


async def run_harness(werk: Werk):
    while True:
        stage, value = next_stage(werk)
        if stage == PLAN:
            if await werk.finish_task(value) is None:
                raise RuntimeError("planner did not finish; resume the session to retry")
        elif stage == "create_coder":
            request, plan = value
            task = (
                f"Complete this request:\n\n{request}\n\n"
                f"Follow this plan:\n\n{json.dumps(plan, indent=2)}"
            )
            werk.add_task(Task(task, label=CODING))
        elif stage == CODING:
            print("coder> ", end="", flush=True)
            await werk.finish_task(value)
            print()
            try:
                reply = input("you> ").strip()
            except EOFError:
                return None
            if reply == "/finish":
                werk.set_task_finished(value, {"status": "finished"})
            elif reply == "/quit":
                return None
            elif reply:
                werk.add_reply(value, reply)
        else:
            return value


async def main(request: Optional[str]) -> None:
    repo = Path.cwd()
    session = repo / SESSION_DIR
    werk = open_werk(session, request)
    werk.add_agent(planner_agent(repo))
    werk.add_agent(coder_agent(repo))
    werk.add_condition(coder_condition())
    werk.on_event(stream_coder_text)

    if request is not None:
        werk.add_task(Task(request, label=PLAN))

    print("coding harness: /finish accepts the change, /quit leaves it resumable")
    result = await run_harness(werk)
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        asyncio.run(main(parse_args(sys.argv[1:])))
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
