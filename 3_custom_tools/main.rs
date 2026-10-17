//! Chapter 3 - The Art of Magical Forging: the Dice Roller
//!
//! A side-by-side port of the Python workshop's `3_custom_tools/`:
//!   <https://github.com/aws-samples/sample-once-upon-agentic-ai/tree/main/3_custom_tools>
//!
//! Chapter 2 handed the agent a tool someone else wrote. Here we forge our own:
//! a plain `roll_dice` function, wrapped in the SDK's `Tool` trait so the model
//! can see it, call it, and get a real random number back instead of inventing one.
//!
//! Run with:
//!   cargo run --bin ch3-custom-tools
//!
//! Environment overrides: same as Chapter 1 (MODEL_ID, AWS_REGION, RUST_LOG).

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use strands_agents::models::{BedrockModel, ModelConfig};
use strands_agents::tools::ToolResult;
use strands_agents::{AgentBuilder, Result, Tool, ToolContext, ToolSpec};

const LADY_LUCK_PROMPT: &str = "You are Lady Luck, the mystical keeper of dice and fortune in \
Table-Top Role-Playing Game (TTRPG) adventures. \
You speak with theatrical flair and always announce dice rolls with appropriate drama. \
You know all about TTRPG mechanics, ability scores, and can help players with character creation. \
When rolling ability scores, remember the traditional method: roll 4d6, drop the lowest die.";

const QUERY: &str = "Help me create a new TTRPG character! Roll the strength, wisdom, charisma \
and intelligence abilities scores using 4d6 drop lowest method.";

/// Roll a single die with the given number of faces.
///
/// The same function as the Python workshop's `roll_dice`, minus the decorator:
/// it knows nothing about agents, so it can be tested on its own.
fn roll_dice(faces: i64) -> std::result::Result<i64, String> {
    if faces < 1 {
        return Err("Dice must have at least 1 face".to_string());
    }
    Ok(rand::random_range(1..=faces))
}

/// The bridge between our function and the model.
///
/// Python's `@tool` reads the signature and docstring to build this for you.
/// In Rust we write the spec by hand: name, description, and a JSON schema the
/// model fills in when it decides to roll.
struct RollDice;

// TODO: Step 1 - Implement the Tool trait to transform RollDice into a tool
#[async_trait]
impl Tool for RollDice {
    fn spec(&self) -> ToolSpec {
        // TODO: Step 2 - Write the description, the faces parameter and the return value
        todo!("Step 1 - Tell the model about roll_dice: its name, description and input schema")
    }

    async fn execute(&self, _input: Value, _ctx: &mut ToolContext) -> Result<ToolResult> {
        todo!("Step 1 - Read faces from the model's JSON input and roll the die")
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ch3_custom_tools=info".into()),
        )
        .init();

    // What the SDK tells the model about our tool - the Rust twin of
    // Python's `print(roll_dice.tool_spec)`.
    println!("{}\n", serde_json::to_string_pretty(&RollDice.spec())?);

    let model = game_master_model().await?;

    let mut dice_master = AgentBuilder::new()
        .model(Box::new(model))
        .name("Lady Luck")
        .system_prompt(LADY_LUCK_PROMPT)
        // TODO: Step 3 - Add the tool to the agent
        .build()
        .await?;

    println!("Adventurer: {QUERY}\n");
    let result = dice_master.run(QUERY).await?;
    println!("Lady Luck: {}\n", result.content);

    // 4 abilities x 4d6 should mean 16 real rolls. Count them.
    let metrics = &result.event_loop_metrics;
    let rolls = metrics
        .tool_metrics
        .get("roll_dice")
        .map_or(0, |m| m.call_count);
    println!(
        "[roll_dice calls={rolls} | stop_reason={} | input_tokens={} | output_tokens={}]",
        result.stop_reason,
        metrics.accumulated_usage.input_tokens,
        metrics.accumulated_usage.output_tokens
    );

    Ok(())
}

/// Default Bedrock model for the whole adventure.
///
/// Verified live against the configured Bedrock account (us-east-1, 2026-10-05
/// via `aws bedrock list-inference-profiles`): `ACTIVE`, and the same model the
/// Python workshop teaches against. Override it with `MODEL_ID` for a cheaper
/// run, e.g. `export MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0`.
const DEFAULT_MODEL_ID: &str = "us.anthropic.claude-sonnet-4-6";

/// Resolve the model id from `MODEL_ID`, falling back to the verified default.
fn model_id() -> String {
    std::env::var("MODEL_ID").unwrap_or_else(|_| DEFAULT_MODEL_ID.to_string())
}

/// Build the storyteller model on Amazon Bedrock.
///
/// Honours `MODEL_ID` and the standard AWS environment (`AWS_REGION`,
/// `AWS_PROFILE`, ...). A high temperature is the right default for a
/// storyteller; a low one would make it repeat itself.
async fn game_master_model() -> strands_agents::Result<BedrockModel> {
    let config = ModelConfig::new(model_id()).with_temperature(0.9);
    BedrockModel::with_config(config).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roll_dice_stays_on_the_die() {
        for _ in 0..1_000 {
            let roll = roll_dice(20).expect("a d20 is a valid die");
            assert!((1..=20).contains(&roll));
        }
    }

    #[test]
    fn roll_dice_refuses_a_die_with_no_faces() {
        assert!(roll_dice(0).is_err());
        assert!(roll_dice(-3).is_err());
    }
}
