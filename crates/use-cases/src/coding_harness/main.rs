//! Plan, implement, and verify one repository change with two agents.
//!
//! Usage: coding-harness <TASK>
//!        coding-harness --resume

use std::error::Error;
use std::io::{self, ErrorKind, Write};
use std::path::Path;
use std::sync::Arc;

use agentwerk::tools::{
    CommandTool, EditFileTool, GlobTool, GrepTool, ListDirectoryTool, ReadFileTool, WriteFileTool,
};
use agentwerk::{Agent, Condition, Event, Task, Werk};
use serde_json::{json, Value};

const PLAN: &str = "plan";
const CODING: &str = "coding";
const SESSION_DIR: &str = "./session";
const CODER_TASK: &str = "Implement this plan:\n\n{{ result: plan }}";
const PLANNER_ROLE: &str = include_str!("planner.md");
const CODER_ROLE: &str = include_str!("coder.md");

#[derive(Debug, PartialEq)]
enum RunMode {
    New(String),
    Resume,
}

enum Stage {
    Planner(String),
    CreateCoder { request: String, plan: Value },
    Coder(String),
    Done(Value),
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let mode = parse_args(std::env::args().skip(1)).unwrap_or_else(|message| {
        eprintln!("{message}");
        std::process::exit(2);
    });
    let repo = std::env::current_dir()?;
    let session = repo.join(SESSION_DIR);
    let werk = open_werk(&session, &mode)?;

    werk.add_agent(planner_agent(&repo));
    werk.add_agent(coder_agent(&repo));
    werk.add_condition(coder_condition());
    werk.on_event(|_, event| stream_coder_text(event));

    if let RunMode::New(request) = mode {
        werk.add_task(Task::new(request).label(PLAN));
    }

    println!("coding harness: /finish accepts the change, /quit leaves it resumable");
    if let Some(result) = run_harness(&werk).await? {
        println!("{}", serde_json::to_string_pretty(&result)?);
    }
    Ok(())
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<RunMode, String> {
    let args: Vec<String> = args.into_iter().collect();
    if args == ["--resume"] {
        return Ok(RunMode::Resume);
    }
    if args.is_empty() || args.iter().any(|arg| arg == "--resume") {
        return Err("usage: coding-harness <TASK> | coding-harness --resume".to_string());
    }

    let request = args.join(" ").trim().to_string();
    if request.is_empty() {
        return Err("the coding task must not be empty".to_string());
    }
    Ok(RunMode::New(request))
}

fn open_werk(session: &Path, mode: &RunMode) -> io::Result<Arc<Werk>> {
    match mode {
        RunMode::New(_) if session.exists() => Err(io::Error::new(
            ErrorKind::AlreadyExists,
            format!(
                "session already exists at {}; use --resume or move it before starting another change",
                session.display()
            ),
        )),
        RunMode::Resume if !session.exists() => Err(io::Error::new(
            ErrorKind::NotFound,
            format!("no session exists at {}", session.display()),
        )),
        RunMode::Resume => Werk::load(session),
        RunMode::New(_) => {
            let werk = Werk::new();
            werk.set_dir(session);
            Ok(werk)
        }
    }
}

async fn run_harness(werk: &Arc<Werk>) -> Result<Option<Value>, Box<dyn Error>> {
    loop {
        match next_stage(werk)? {
            Stage::Planner(id) => {
                werk.finish_task(id).await.ok_or_else(|| {
                    io::Error::other("planner did not finish; resume the session to retry")
                })?;
            }
            Stage::CreateCoder { request, plan } => {
                let plan = serde_json::to_string_pretty(&plan)?;
                let task =
                    format!("Complete this request:\n\n{request}\n\nFollow this plan:\n\n{plan}");
                werk.add_task(Task::new(task).label(CODING));
            }
            Stage::Coder(id) => {
                print!("coder> ");
                io::stdout().flush()?;
                werk.finish_task(id.clone()).await;
                println!();

                let Some(input) = read_line() else {
                    return Ok(None);
                };
                match input.as_str() {
                    "/finish" => {
                        werk.set_task_finished(&id, json!({ "status": "finished" }))?;
                    }
                    "/quit" => return Ok(None),
                    _ => {
                        werk.add_reply(&id, input);
                    }
                }
            }
            Stage::Done(result) => return Ok(Some(result)),
        }
    }
}

fn read_line() -> Option<String> {
    loop {
        print!("you> ");
        io::stdout().flush().ok()?;

        let mut input = String::new();
        if io::stdin().read_line(&mut input).ok()? == 0 {
            return None;
        }
        let input = input.trim().to_string();
        if !input.is_empty() {
            return Some(input);
        }
    }
}

fn stream_coder_text(event: &Event) {
    if event.get_name() == Event::TEXT_CHUNK_RECEIVED && event.get_label() == Some(CODING) {
        print!(
            "{}",
            event.get_data()["content"].as_str().unwrap_or_default()
        );
        let _ = io::stdout().flush();
    }
}

fn next_stage(werk: &Werk) -> io::Result<Stage> {
    let planners = werk.find_tasks(PLAN);
    if planners.len() != 1 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "the session must contain exactly one planner task",
        ));
    }
    let planner = &planners[0];
    if planner.is_failed() {
        return Err(io::Error::other("the planner task failed"));
    }
    if !planner.is_finished() {
        return Ok(Stage::Planner(planner.get_id().to_string()));
    }

    let request = planner
        .get_task()
        .as_str()
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "the planner request is not text"))?
        .to_string();
    let plan = planner
        .get_result()
        .cloned()
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "the planner result is missing"))?;

    let coders = werk.find_tasks(CODING);
    if coders.len() > 1 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "the session contains more than one coder task",
        ));
    }
    let Some(coder) = coders.first() else {
        return Ok(Stage::CreateCoder { request, plan });
    };
    if coder.is_failed() {
        return Err(io::Error::other("the coder task failed"));
    }
    if !coder.is_finished() {
        return Ok(Stage::Coder(coder.get_id().to_string()));
    }

    coder
        .get_result()
        .cloned()
        .map(Stage::Done)
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "the coder result is missing"))
}

fn planner_agent(repo: &Path) -> Agent {
    read_only_agent(Agent::from_env().label(PLAN).role(PLANNER_ROLE).dir(repo)).tool(git_tool())
}

fn coder_agent(repo: &Path) -> Agent {
    read_only_agent(
        Agent::from_env()
            .label(CODING)
            .role(CODER_ROLE)
            .dir(repo)
            .interactive(),
    )
    .tool(EditFileTool)
    .tool(WriteFileTool)
    .tool(git_tool())
    .tool(
        CommandTool::new("cargo")
            .allow("cargo fmt*")
            .allow("cargo check*")
            .allow("cargo test*"),
    )
}

fn read_only_agent(agent: Agent) -> Agent {
    agent
        .tool(ListDirectoryTool)
        .tool(GlobTool)
        .tool(GrepTool)
        .tool(ReadFileTool)
}

fn git_tool() -> CommandTool {
    CommandTool::new("git")
        .allow("git status*")
        .allow("git diff*")
}

fn coder_condition() -> Condition {
    Condition::new("task.label = plan AND task.status = finished")
        .task(Task::labeled(CODING, CODER_TASK))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn parse(values: &[&str]) -> Result<RunMode, String> {
        parse_args(values.iter().map(|value| value.to_string()))
    }

    fn temp_session() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "agentwerk-coding-harness-{}-{unique}",
            std::process::id()
        ))
    }

    #[test]
    fn arguments_require_a_task_or_resume() {
        assert_eq!(
            parse(&["fix", "the", "test"]),
            Ok(RunMode::New("fix the test".into()))
        );
        assert_eq!(parse(&["--resume"]), Ok(RunMode::Resume));
        assert!(parse(&[]).is_err());
        assert!(parse(&["--resume", "extra"]).is_err());
    }

    #[test]
    fn session_mode_prevents_overwrite_and_missing_resume() {
        let session = temp_session();
        assert!(open_werk(&session, &RunMode::Resume).is_err());

        let werk = open_werk(&session, &RunMode::New("change".into())).unwrap();
        werk.add_task(Task::new("change").label(PLAN));
        assert!(session.exists());
        assert!(open_werk(&session, &RunMode::New("other".into())).is_err());
        assert!(open_werk(&session, &RunMode::Resume).is_ok());

        fs::remove_dir_all(session).unwrap();
    }

    #[test]
    fn stages_create_one_coder_task_and_finish_with_its_result() {
        let werk = Werk::new();
        let planner = werk.add_task(Task::new("fix addition").label(PLAN));
        assert!(matches!(next_stage(&werk).unwrap(), Stage::Planner(id) if id == planner));

        let plan = json!({
            "plan": ["Implement addition", "Run cargo test"],
            "files": ["src/lib.rs"],
            "checks": ["cargo test"]
        });
        werk.set_task_finished(&planner, plan.clone()).unwrap();
        assert!(matches!(
            next_stage(&werk).unwrap(),
            Stage::CreateCoder { request, plan: found }
                if request == "fix addition" && found == plan
        ));

        let coder = werk.add_task(Task::new("implement the plan").label(CODING));
        assert!(matches!(next_stage(&werk).unwrap(), Stage::Coder(id) if id == coder));
        assert_eq!(werk.find_tasks(CODING).len(), 1);

        let result = json!({
            "summary": "Implemented addition.",
            "changed_files": ["src/lib.rs"],
            "checks": ["cargo test: passed"]
        });
        werk.set_task_finished(&coder, result.clone()).unwrap();
        assert!(matches!(next_stage(&werk).unwrap(), Stage::Done(found) if found == result));
        assert_eq!(werk.find_tasks(CODING).len(), 1);
    }

    #[test]
    fn finishing_the_plan_condition_creates_one_coder_task() {
        let werk = Werk::new();
        werk.add_condition(coder_condition());
        let plan = werk.add_task(Task::new("fix addition").label(PLAN));

        werk.set_task_finished(&plan, json!({ "plan": ["Fix it"] }))
            .unwrap();

        let coders = werk.find_tasks(CODING);
        assert_eq!(coders.len(), 1);
        assert_eq!(coders[0].get_task(), &json!(CODER_TASK));
    }

    #[test]
    fn stages_reject_duplicate_coder_tasks() {
        let werk = Werk::new();
        let planner = werk.add_task(Task::new("fix addition").label(PLAN));
        werk.set_task_finished(&planner, json!({ "plan": ["Fix it"] }))
            .unwrap();
        werk.add_task(Task::new("first").label(CODING));
        werk.add_task(Task::new("duplicate").label(CODING));

        let error = next_stage(&werk).err().unwrap();
        assert_eq!(error.kind(), ErrorKind::InvalidData);
        assert!(error.to_string().contains("more than one coder task"));
    }
}
