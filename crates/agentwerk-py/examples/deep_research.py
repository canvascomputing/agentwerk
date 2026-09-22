"""Research one question, share the findings, and write a cited report.

Usage: python deep_research.py <QUESTION>
"""

import asyncio
import os
import sys
from pathlib import Path

from agentwerk import Agent, Condition, Event, FetchTool, Knowledge, Policy, Task, Werk

from web_search import brave_search_tool


PROMPTS = Path(__file__).with_name("prompts")
RESEARCHER_ROLE = (PROMPTS / "researcher.role.md").read_text()
WRITER_ROLE = (PROMPTS / "writer.role.md").read_text()
RESEARCH_TASK = "Research {{ question }} with emphasis on {{ focus }}."
REPORT_TASK = "Write a cited report answering:\n\n{{ question }}"
FOCUS = "latency and reliability"
RESEARCH = "research"
REPORT = "report"


def log_research(_, event) -> None:
    if event.get_name() == Event.KNOWLEDGE_WRITTEN:
        slug = event.get_data().get("slug", "")
        print(f"Saved research: {slug}", file=sys.stderr)


async def main(question: str) -> None:
    knowledge = Knowledge.load(".agentwerk/research")

    researcher = Agent.from_env()
    researcher.label(RESEARCH)
    researcher.role(RESEARCHER_ROLE)
    researcher.knowledge(knowledge)
    researcher.tool(brave_search_tool(os.environ["BRAVE_API_KEY"]))
    researcher.tool(FetchTool())

    writer = Agent.from_env()
    writer.label(REPORT)
    writer.role(WRITER_ROLE)
    writer.knowledge(knowledge)

    research_task = Task(RESEARCH_TASK, label=RESEARCH)
    report_task = Task(REPORT_TASK, label=REPORT)
    write_report = Condition("task.label = research AND task.status = finished")
    write_report.task(report_task)

    werk = Werk()
    werk.set_policy(Policy(max_time=300))
    werk.set_template("question", question)
    werk.set_template("focus", FOCUS)
    werk.on_event(log_research)
    werk.add_agent(researcher)
    werk.add_agent(writer)
    werk.add_condition(write_report)
    werk.add_task(research_task)
    await werk.finish()

    result = werk.find_result(REPORT)
    if result is None:
        raise RuntimeError("the writer produced no report")
    print(result.get("report", ""))


def question_from_args() -> str:
    if len(sys.argv) < 2 or sys.argv[1] in {"--help", "-h"}:
        raise SystemExit("Usage: python deep_research.py <QUESTION>")
    return " ".join(sys.argv[1:])


if __name__ == "__main__":
    asyncio.run(main(question_from_args()))
