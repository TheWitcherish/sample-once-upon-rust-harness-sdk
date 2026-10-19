//! Chapter 4 - Planar Portals: Lady Luck rolls through MCP
//!
//! A side-by-side port of the Python workshop's `4_mcp_integration/gamemaster_mcp_client.py`:
//!   <https://github.com/aws-samples/sample-once-upon-agentic-ai/tree/main/4_mcp_integration>
//!
//! Lady Luck has no dice of her own any more. She connects to the dice roller
//! MCP server over streamable HTTP, discovers `roll_dice` at runtime, and calls
//! it across the wire. Same SDK, same agent; the tool just lives somewhere else.
//!
//! The Python chapter sends one fixed request. Here you type your own, as many
//! as you like: "roll 3d20", "roll a d100 for loot", or the Python chapter's
//! character-creation request. Watch the server terminal light up per die.
//!
//! Start the server first, in another terminal:
//!   cargo run --bin ch4-dice-mcp-server
//! Then:
//!   cargo run --bin ch4-gamemaster-mcp-client
//!
//! Environment overrides: same as Chapter 1, plus
//!   MCP_URL  the server's MCP endpoint (default http://localhost:8002/mcp)

use std::env;
use std::io::{self, BufRead, Write};
use std::sync::Arc;

use strands_agents::models::{BedrockModel, ModelConfig};
// TODO: Step 1 - Import MCPClient and StreamableHttpTransport from strands_agents::tools::mcp
use strands_agents::{AgentBuilder, Result};

const DEFAULT_MCP_URL: &str = "http://localhost:8002/mcp";

/// The Python chapter's prompt, line for line, with TTRPG for the brand name.
const LADY_LUCK_PROMPT: &str = "You are Lady Luck, the mystical keeper of dice and fortune in TTRPG adventures.
    You speak with theatrical flair and always announce dice rolls with appropriate drama.
    You know all about TTRPG mechanics, always use the appropriate tools when applicable - never make up results!";

/// The Python chapter's query, with TTRPG for the brand name. Offered as an example.
const PYTHON_QUERY: &str = "Help me create a new TTRPG character! Roll the strength, wisdom, charisma and intelligence abilities scores using 4d6 drop lowest method.";

/// Read the adventurer's next request. `None` means leave: end of input, an
/// empty line, or `quit` / `exit`.
fn next_request(stdin: &mut impl BufRead) -> io::Result<Option<String>> {
    print!("\nAdventurer> ");
    io::stdout().flush()?;

    let mut line = String::new();
    if stdin.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let request = line.trim();
    if request.is_empty() || ["quit", "exit"].contains(&request.to_lowercase().as_str()) {
        return Ok(None);
    }
    Ok(Some(request.to_string()))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "strands_agents=warn".into()),
        )
        .init();

    let url = env::var("MCP_URL").unwrap_or_else(|_| DEFAULT_MCP_URL.to_string());
    // TODO: Step 1 - Create a streamable http MCPClient connecting to url (default "http://localhost:8002/mcp")

    let model = game_master_model().await?;
    let mut gamemaster = AgentBuilder::new()
        .model(Box::new(model))
        .system_prompt(LADY_LUCK_PROMPT)
        // TODO: Step 2 - Add the MCP tool provider to the gamemaster agent
        .build()
        .await?;

    println!("Lady Luck is listening, her dice on the server at {url}.");
    println!("Ask for any roll, for example:");
    println!("  roll 3d20");
    println!("  {PYTHON_QUERY}");
    println!("Press Enter on an empty line, or type quit, to leave.");

    let mut stdin = io::stdin().lock();
    let mut rolls_so_far = 0;
    while let Some(request) = next_request(&mut stdin)? {
        let result = gamemaster.run(request.as_str()).await?;
        println!("\nLady Luck: {}", result.content);

        // The rolls happened in another process; the server's log shows each one.
        // The agent's metrics count the whole conversation, so subtract the
        // earlier turns to get this one.
        let total = result
            .event_loop_metrics
            .tool_metrics
            .get("roll_dice")
            .map_or(0, |m| m.call_count);
        println!(
            "\n[roll_dice calls over MCP this turn={} | stop_reason={}]",
            total - rolls_so_far,
            result.stop_reason
        );
        rolls_so_far = total;
    }

    println!("\nLady Luck tucks her dice away. Until next time, adventurer.");
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

    fn read(input: &str) -> Option<String> {
        next_request(&mut io::Cursor::new(input)).expect("reading a cursor cannot fail")
    }

    #[test]
    fn a_request_is_trimmed_and_kept() {
        assert_eq!(read("  roll 3d20  \n"), Some("roll 3d20".to_string()));
    }

    #[test]
    fn empty_line_quit_exit_and_eof_all_leave() {
        for input in ["\n", "   \n", "quit\n", "EXIT\n", ""] {
            assert_eq!(read(input), None, "{input:?} should leave");
        }
    }
}
