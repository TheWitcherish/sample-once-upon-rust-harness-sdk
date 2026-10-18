//! Chapter 1 - The Art of Agent Summoning
//!
//! A side-by-side port of the Python workshop's `1_strands_basics/simple_agent.py`:
//!   <https://github.com/aws-samples/sample-once-upon-agentic-ai/tree/main/1_strands_basics>
//!
//! Three TODOs, the same three steps as the Python chapter.
//!
//! Run with:
//!   cargo run --bin ch1-summoning
//!
//! Environment overrides:
//!   MODEL_ID     a Bedrock inference-profile id (default: Claude Sonnet 5.5)
//!   AWS_REGION   the region your model lives in, e.g. us-east-1
//!   RUST_LOG     set to strands_agents=debug for the agent's inner monologue

use strands_agents::models::{BedrockModel, ModelConfig};
use strands_agents::{AgentBuilder, Result};

/// The Python chapter's exact first query.
const FIRST_QUERY: &str = "Hi, I am an adventurer ready for adventure!";

#[tokio::main]
async fn main() -> Result<()> {
    // TODO: Step 1 - Add debug logging to see what your agent is thinking
    //
    // Python: logging.getLogger("strands").setLevel(logging.DEBUG)
    // Rust:   the SDK emits `tracing` spans on the `strands_agents` target, so a
    //         subscriber with an env filter is the direct analogue. Default to
    //         info so the demo stays readable; RUST_LOG=strands_agents=debug
    //         opens the floodgates.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "strands_agents=info".into()),
        )
        .with_target(true)
        .init();

    println!("Summoning a game master on {}\n", model_id());

    // TODO: Step 2 - Create the agent with the following system prompt: "You are a Game Master for a Table-Top Role-Playing Game (TTRPG)"
    //
    // Python's Agent() defaults to Bedrock. Rust names the provider explicitly -
    // that is the only structural difference in this whole chapter.
    let model = game_master_model().await?;

    let mut agent = AgentBuilder::new()
        .model(Box::new(model))
        .name("Game Master")
        .system_prompt(GAME_MASTER_PROMPT)
        .build()
        .await?;

    // TODO: Step 3 - Invoke your agent with a basic query such as "Hi, I am an adventurer ready for adventure!"
    //
    // Python: agent("Hi, ...") prints as a side effect.
    // Rust:   agent.run(..) returns an AgentResult, so the caller decides what
    //         to do with it.
    println!("Adventurer: {FIRST_QUERY}\n");
    let result = agent.run(FIRST_QUERY).await?;
    println!("Game Master: {}\n", result.content);

    let usage = &result.event_loop_metrics.accumulated_usage;
    println!(
        "[stop_reason={} | input_tokens={} | output_tokens={}]\n",
        result.stop_reason, usage.input_tokens, usage.output_tokens
    );

    // The agent carries its own history, so a follow-up needs no plumbing.
    let follow_up = "I check the tavern for a quest board. What do I find?";
    println!("Adventurer: {follow_up}\n");
    let result = agent.run(follow_up).await?;
    println!("Game Master: {}\n", result.content);

    println!("Messages retained in history: {}\n", agent.messages().len());

    Ok(())
}

/// Default Bedrock model for the whole adventure.
///
/// Verified live against the configured Bedrock account (us-east-1, 2026-10-18
/// via `aws bedrock list-inference-profiles`): `ACTIVE`. Override it with
/// `MODEL_ID` for a cheaper run, e.g.
/// `export MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0`.
const DEFAULT_MODEL_ID: &str = "us.anthropic.claude-sonnet-5-5";

/// The system prompt shared by the whole adventure.
const GAME_MASTER_PROMPT: &str = "You are a Game Master for a Table-Top Role-Playing Game (TTRPG)";

/// Resolve the model id from `MODEL_ID`, falling back to the verified default.
fn model_id() -> String {
    std::env::var("MODEL_ID").unwrap_or_else(|_| DEFAULT_MODEL_ID.to_string())
}

/// Build the storyteller model on Amazon Bedrock.
///
/// Honours `MODEL_ID` and the standard AWS environment (`AWS_REGION`,
/// `AWS_PROFILE`, ...). It sets no temperature: Claude Sonnet 5.5 refuses the
/// parameter ("`temperature` is deprecated for this model").
async fn game_master_model() -> strands_agents::Result<BedrockModel> {
    let config = ModelConfig::new(model_id());
    BedrockModel::with_config(config).await
}
