//! Research one question, share the findings, and write a cited report.
//!
//! Usage: deep-research <QUESTION>

mod web_search;

use agentwerk::tools::FetchTool;
use agentwerk::{Agent, Condition, Event, Knowledge, Task, Werk};

use web_search::{brave_key_from_env, brave_search_tool};

const RESEARCHER_ROLE: &str = include_str!("prompts/researcher.role.md");
const WRITER_ROLE: &str = include_str!("prompts/writer.role.md");
const RESEARCH_TASK: &str = "Research {{ question }} with emphasis on {{ focus }}.";
const REPORT_TASK: &str = "Write a cited report answering:\n\n{{ question }}";
const FOCUS: &str = "latency and reliability";
const RESEARCH: &str = "research";
const REPORT: &str = "report";

#[tokio::main]
async fn main() {
    let question = question_from_args();
    let brave_key = brave_key_from_env().unwrap_or_else(|message| exit(&message));
    let stored_knowledge = Knowledge::load(".agentwerk/research");
    let knowledge = stored_knowledge.unwrap_or_else(|error| exit(&error.to_string()));

    let researcher = Agent::from_env()
        .label(RESEARCH)
        .role(RESEARCHER_ROLE)
        .knowledge(&knowledge)
        .tool(brave_search_tool(brave_key))
        .tool(FetchTool::new());

    let writer = Agent::from_env()
        .label(REPORT)
        .role(WRITER_ROLE)
        .knowledge(&knowledge);

    let research_task = Task::labeled(RESEARCH, RESEARCH_TASK);
    let report_task = Task::labeled(REPORT, REPORT_TASK);
    let finished_research = "task.label = research AND task.status = finished";
    let write_report = Condition::new(finished_research).task(report_task);

    let werk = Werk::new();
    werk.set_template("question", question);
    werk.set_template("focus", FOCUS);
    werk.on_event(|_, event| {
        if event.get_name() == Event::KNOWLEDGE_WRITTEN {
            let slug = event.get_data()["slug"].as_str().unwrap_or_default();
            eprintln!("Saved research: {slug}");
        }
    });
    werk.add_agent(researcher);
    werk.add_agent(writer);
    werk.add_condition(write_report);
    werk.add_task(research_task);
    werk.finish().await;

    let result = werk
        .find_result(REPORT)
        .unwrap_or_else(|| exit("the writer produced no report"));
    let report = result["report"].as_str().unwrap_or_default();
    println!("{report}");
}

fn question_from_args() -> String {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let help = arguments
        .first()
        .is_some_and(|argument| matches!(argument.as_str(), "--help" | "-h"));
    if arguments.is_empty() || help {
        eprintln!("Usage: deep-research <QUESTION>");
        std::process::exit(if help { 0 } else { 1 });
    }
    arguments.join(" ")
}

fn exit(message: &str) -> ! {
    eprintln!("Error: {message}");
    std::process::exit(1)
}
