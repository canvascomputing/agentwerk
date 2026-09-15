//! Route a draft through an AQL condition, then select the edit with AQL.
//!
//! Usage: editorial-review [TEXT]

use agentwerk::{Agent, Condition, Task, Werk};

#[tokio::main]
async fn main() {
    let werk = Werk::new();

    let draft_writer = Agent::from_env()
        .label("draft")
        .role("Write the requested draft. Return only the drafted text.");
    werk.add_agent(draft_writer);

    let editor = Agent::from_env()
        .label("edit")
        .role("Edit the draft for clarity and brevity. Return only the final text.");
    let edit_request =
        "Edit this draft:\n\n{{ result: task.label = draft AND task.status = finished }}";
    let edit_task = Task::new(edit_request).label("edit");
    let route_to_editor = Condition::new("task.label = draft AND task.status = finished")
        .agent(editor)
        .task(edit_task);
    werk.add_condition(route_to_editor);

    let draft_task = Task::new(text_from_args()).label("draft");
    werk.add_task(draft_task);
    werk.finish().await;

    let Some(final_edit) = werk.find_result("task.label = edit AND task.status = finished") else {
        eprintln!("the editor produced no result");
        std::process::exit(1);
    };
    println!("{}", final_edit.as_str().unwrap_or_default());
}

fn text_from_args() -> String {
    let text = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if text.is_empty() {
        return "Announce a new software release in two sentences.".to_string();
    }

    text
}
