//! Chapter 2 - Bonus Quest: Kiro the Grey Hat
//!
//! Kiro gets two powerful vended tools: `shell` and `editor`. Tools that write
//! files and run commands should not fire unsupervised, so every call pauses for
//! your approval first.
//!
//! The Rust SDK has no vended approval step yet, so we build one from what it
//! does ship: a hook that raises an interrupt before each tool call, and a loop
//! that asks you on stdin and resumes the agent with your answer.
//!
//! Run with:
//!   cargo run --bin ch2-bonus-quest
//!
//! Kiro writes its scroll into the directory you run from.

use std::io::{self, BufRead, Write};
use std::sync::Arc;

use serde_json::json;
use strands_agents::hooks::{BeforeToolCallEvent, HookProvider};
use strands_agents::interrupt::Interrupt;
use strands_agents::models::{BedrockModel, ModelConfig};
use strands_agents::tools::ToolContext;
use strands_agents::types::content::ContentBlock;
use strands_agents::{AgentBuilder, Result};
// TODO: Step 1 - Import Shell and Editor from strands_agents_tools
use strands_agents_tools::{Editor, Shell};

/// Kiro's quest.
const QUERY: &str = "Create a magical scroll that generates the first 10 numbers of the Fibonacci sequence and demonstrate its power!";

/// What Kiro hears when you refuse a spell. It becomes the tool result, so the
/// model knows the call never ran.
const DENIED: &str = "The adventurer refused to let this spell be cast.";

/// Kiro's system prompt.
fn system_prompt(cwd: &str) -> String {
    format!(
        "You are Kiro the Grey Hat, a wizard who specializes in the ancient art of code magic.
    When asked to create spells (code), you inscribe them on parchment (files) in the directory {cwd}
    and then cast them to demonstrate their power."
    )
}

/// Only an explicit yes lets a tool run. Anything else, including an empty line, is a no.
fn is_approved(answer: &str) -> bool {
    matches!(answer.trim().to_lowercase().as_str(), "y" | "yes")
}

/// The gate. Before every tool call it raises an interrupt carrying the tool name
/// and input. The agent pauses; when it resumes, the same hook runs again and
/// `interrupt` now returns the human's answer instead of halting.
struct HumanInTheLoop;

impl HookProvider for HumanInTheLoop {
    fn on_before_tool_call(&self, event: &mut BeforeToolCallEvent, _ctx: &mut ToolContext) {
        let request = json!({ "tool": event.tool_name, "input": event.input });
        if let Ok(answer) = event.interrupt("approval", request)
            && !answer.as_str().is_some_and(is_approved)
        {
            event.cancel_tool = Some(DENIED.to_string());
        }
        // Err(InterruptHalt) is the first pass: the runtime records the pause for us.
    }

    fn name(&self) -> &str {
        "HumanInTheLoop"
    }
}

/// Show one pending tool call and read y/N from stdin.
fn ask(interrupt: &Interrupt, stdin: &mut impl BufRead) -> Result<ContentBlock> {
    let request = interrupt.reason.clone().unwrap_or_default();
    let tool = request["tool"].as_str().unwrap_or("an unknown tool");
    println!("\n--- Kiro wants to cast `{tool}` ---");
    println!("{}", serde_json::to_string_pretty(&request["input"])?);
    print!("Approve? [y/N] ");
    io::stdout().flush()?;

    let mut answer = String::new();
    stdin.read_line(&mut answer)?;
    Ok(ContentBlock::interrupt_response(
        &interrupt.id,
        json!(answer.trim()),
    ))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "strands_agents=warn".into()),
        )
        .init();

    let cwd = std::env::current_dir()?.display().to_string();
    let model = game_master_model().await?;

    let mut arcane_scribe = AgentBuilder::new()
        .model(Box::new(model))
        // TODO: Step 1 - Add the Shell and Editor tools to your agent
        .tool(Arc::new(Shell::new()))
        .tool(Arc::new(Editor::new()))
        // TODO: Step 2 - Ask for your approval before each tool call with the HumanInTheLoop hook
        .hook(Box::new(HumanInTheLoop))
        .system_prompt(system_prompt(&cwd))
        .build()
        .await?;

    println!("Adventurer: {QUERY}");
    let mut stdin = io::stdin().lock();
    let mut result = arcane_scribe.run(QUERY).await?;

    // Each pause hands back the calls waiting for a verdict. Answer them all, resume.
    while result.is_interrupt() {
        let answers = result
            .interrupts
            .iter()
            .map(|interrupt| ask(interrupt, &mut stdin))
            .collect::<Result<Vec<_>>>()?;
        result = arcane_scribe.run(answers).await?;
    }

    println!("\nKiro the Grey Hat: {}", result.content);
    Ok(())
}

/// Default Bedrock model for the whole adventure.
///
/// Verified live against the configured Bedrock account (us-west-2, 2026-10-19
/// via `aws bedrock get-inference-profile`): `ACTIVE`. Override it with
/// `MODEL_ID`, e.g. `export MODEL_ID=us.anthropic.claude-sonnet-5-5`.
const DEFAULT_MODEL_ID: &str = "us.anthropic.claude-haiku-5-5";

/// Resolve the model id from `MODEL_ID`, falling back to the verified default.
fn model_id() -> String {
    std::env::var("MODEL_ID").unwrap_or_else(|_| DEFAULT_MODEL_ID.to_string())
}

/// Build the storyteller model on Amazon Bedrock.
///
/// Honours `MODEL_ID` and the standard AWS environment (`AWS_REGION`,
/// `AWS_PROFILE`, ...). It sets no temperature: Claude Haiku 5.5 refuses the
/// parameter ("`temperature` is deprecated for this model").
async fn game_master_model() -> strands_agents::Result<BedrockModel> {
    let config = ModelConfig::new(model_id());
    BedrockModel::with_config(config).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_an_explicit_yes_approves() {
        for yes in ["y", "Y", "yes", " YES \n"] {
            assert!(is_approved(yes), "{yes:?} should approve");
        }
        for no in ["", "\n", "n", "no", "sure", "yy"] {
            assert!(!is_approved(no), "{no:?} should deny");
        }
    }

    #[test]
    fn ask_answers_the_interrupt_it_was_shown() -> Result<()> {
        let interrupt = Interrupt::new(
            "id-42",
            "approval",
            Some(json!({ "tool": "shell", "input": { "command": "ls" } })),
            Default::default(),
        );
        let block = ask(&interrupt, &mut io::Cursor::new("y\n"))?;
        let raw = serde_json::to_value(&block)?;
        assert!(raw.to_string().contains("id-42"), "{raw}");
        assert!(raw.to_string().contains("\"y\""), "{raw}");
        Ok(())
    }

    #[test]
    fn prompt_names_the_working_directory() {
        assert!(system_prompt("/tmp/scrolls").contains("in the directory /tmp/scrolls\n"));
    }
}
