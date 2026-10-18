//! Chapter 5 - The Game Master orchestrator (HTTP API on port 8009)
//!
//! A side-by-side port of the Python workshop's
//! `5_a2a_integration/agents/gamemaster_orchestrator/gamemaster_orchestrator.py`:
//!   <https://github.com/aws-samples/sample-once-upon-agentic-ai/tree/main/5_a2a_integration>
//!
//! The Game Master keeps no rules and no character sheets of its own. It rolls
//! dice over MCP (the Chapter 4 server), and asks the Rules Agent and the
//! Character Agent over A2A, discovering them at runtime with `make_a2a_client`.
//! Every answer comes back as a typed `StoryOutput`, forced the way Python's
//! `structured_output_model` forces it.
//!
//! Start the three servers first, each in its own terminal:
//!   cargo run --bin ch4-dice-mcp-server
//!   cargo run --bin ch5-rules-agent
//!   cargo run --bin ch5-character-agent
//! Then:
//!   cargo run --bin ch5-gamemaster-orchestrator
//!   curl -s localhost:8009/inquire -H 'content-type: application/json' -d '{"question":"Roll a d20"}'
//!
//! Environment overrides: MODEL_ID, AWS_REGION, RUST_LOG, MCP_URL (default
//! http://localhost:8002/mcp), CHARACTERS_DB, and GAMEMASTER_BIND (default
//! 127.0.0.1:8009).
//!
//! Browsers may call the API from `http://localhost:<port>` and
//! `http://127.0.0.1:<port>` origins only (the Chapter 6 UI's dev server).

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{HeaderValue, Method, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use strands_agents::a2a::{A2AClientOptions, make_a2a_client};
use strands_agents::agent::STRUCTURED_OUTPUT_PROMPT;
use strands_agents::models::{BedrockModel, ModelConfig};
use strands_agents::tools::mcp::{MCPClient, StreamableHttpTransport};
use strands_agents::types::content::ContentBlock;
use strands_agents::types::message::Message;
use strands_agents::{Agent, AgentBuilder};
use tokio::sync::Mutex;
use tower_http::cors::{AllowOrigin, CorsLayer};

const DEFAULT_MCP_URL: &str = "http://localhost:8002/mcp";
const RULES_AGENT_URL: &str = "http://127.0.0.1:8000";
const CHARACTER_AGENT_URL: &str = "http://127.0.0.1:8001";
const DEFAULT_PORT: u16 = 8009;

/// The Python chapter's prompt, with TTRPG for the brand name.
const SYSTEM_PROMPT: &str = "You are a TTRPG Game Master. Discover the agents you can reach and ask them instead of guessing: rules questions, character creation and lookups are their job. Every dice roll goes through roll_dice. Never make up what a tool can tell you, and narrate with flair.";

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DiceOutput {
    /// The dice type. Ex: d4, d6, d20, etc
    pub dice_type: String,
    /// The dice result value alone
    pub result: i64,
    /// The reason the dice was rolled. Ex: attack roll. And the modificators if there was any
    pub reason: String,
}

/// A single Game Master turn: the narration, what the player could do next, and any dice rolled.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct StoryOutput {
    /// Your narative response as Game Master
    pub response: String,
    /// ['Action 1', 'Action 2', 'Action 3']
    pub actions_suggestions: Vec<String>,
    /// Brief summary of tools/agents used
    pub details: String,
    /// List of dice rolls with dice_type, result, and reason
    #[serde(default)]
    pub dice_rolls: Vec<DiceOutput>,
}

#[derive(Debug, Deserialize)]
struct QuestionRequest {
    question: String,
}

#[derive(Clone)]
struct AppState {
    // One agent, one conversation: requests take turns, like the Python
    // module-level `agent`.
    agent: Arc<Mutex<Agent>>,
    characters_db: PathBuf,
}

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(json!({ "error": message.into() }))).into_response()
}

async fn health() -> Json<Value> {
    Json(json!({ "status": "healthy" }))
}

/// One message in the Converse API shape Python's `agent.messages` returns,
/// which the Chapter 6 UI parses: tool blocks are wrapped as
/// `{"toolUse": {...}}` and `{"toolResult": {...}}`. The Rust SDK serializes
/// them unwrapped, so the wrapper is added here.
fn wire_message(message: &Message) -> Value {
    let content: Vec<Value> = message
        .content
        .iter()
        .map(|block| match block {
            ContentBlock::ToolUse(tool_use) => json!({ "toolUse": tool_use }),
            ContentBlock::ToolResult(result) => json!({ "toolResult": result }),
            other => serde_json::to_value(other).unwrap_or(Value::Null),
        })
        .collect();
    json!({ "role": message.role, "content": content })
}

/// The agent's own "format that as structured output" nudge. It is a user
/// message in the history, but the player never typed it, so the UI must not
/// show it as theirs.
fn is_structured_output_nudge(message: &Message) -> bool {
    message.text_content() == STRUCTURED_OUTPUT_PROMPT
}

async fn messages(State(app): State<AppState>) -> Json<Value> {
    let agent = app.agent.lock().await;
    Json(Value::Array(
        agent
            .messages()
            .iter()
            .filter(|m| !is_structured_output_nudge(m))
            .map(wire_message)
            .collect(),
    ))
}

/// Find a character in the Character Agent's TinyDB file, by exact name.
fn find_character(db: &Value, name: &str) -> Option<Value> {
    db.get("_default")?
        .as_object()?
        .values()
        .find(|c| c.get("name").and_then(Value::as_str) == Some(name))
        .cloned()
}

/// Reads the Character Agent's file directly, like the Python endpoint.
async fn user(State(app): State<AppState>, Path(user_name): Path<String>) -> Response {
    let db = match tokio::fs::read(&app.characters_db).await {
        Ok(bytes) => match serde_json::from_slice::<Value>(&bytes) {
            Ok(db) => db,
            Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        },
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    match find_character(&db, &user_name) {
        Some(character) => {
            tracing::info!(name = %user_name, "found character");
            Json(character).into_response()
        }
        None => error(
            StatusCode::NOT_FOUND,
            format!("Character with name '{user_name}' not found"),
        ),
    }
}

async fn ask(agent: &mut Agent, question: &str) -> strands_agents::Result<StoryOutput> {
    agent
        .run(question)
        .await?
        .structured_output::<StoryOutput>()
}

async fn inquire(State(app): State<AppState>, Json(request): Json<QuestionRequest>) -> Response {
    tracing::info!(question = %request.question, "processing request");
    let mut agent = app.agent.lock().await;
    match ask(&mut agent, &request.question).await {
        Ok(story) => Json(json!({ "response": story })).into_response(),
        Err(e) => {
            tracing::error!(error = %e, "inquire failed");
            error(StatusCode::INTERNAL_SERVER_ERROR, "Internal server error")
        }
    }
}

fn is_local_origin(origin: &HeaderValue) -> bool {
    let Ok(origin) = origin.to_str() else {
        return false;
    };
    ["http://localhost:", "http://127.0.0.1:"]
        .iter()
        .any(|prefix| {
            origin
                .strip_prefix(prefix)
                .is_some_and(|port| port.parse::<u16>().is_ok())
        })
}

/// Python allows every origin (`*`) with credentials. The UI needs neither:
/// it runs on a local dev server and sends no cookies, so only local origins
/// get in.
fn cors() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin, _| is_local_origin(origin)))
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([header::CONTENT_TYPE])
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                "ch5_gamemaster_orchestrator=info,strands_agents::a2a=info".into()
            }),
        )
        .init();

    let mcp_url = std::env::var("MCP_URL").unwrap_or_else(|_| DEFAULT_MCP_URL.to_string());
    // TODO: Step 1 - Create a streamable http MCPClient connecting to mcp_url (default "http://localhost:8002/mcp")
    let dice = MCPClient::new(Box::new(StreamableHttpTransport::new(mcp_url.as_str())));
    // TODO: Step 2 - Create the A2A client tool with make_a2a_client and the allowed agent endpoints
    let a2a = make_a2a_client(A2AClientOptions::new([
        RULES_AGENT_URL,
        CHARACTER_AGENT_URL,
    ]))?;

    let agent = AgentBuilder::new()
        .model(Box::new(game_master_model().await?))
        .system_prompt(SYSTEM_PROMPT)
        // TODO: Step 3 - Add the dice MCP client and the a2a tool to the agent
        .tool_provider(Arc::new(dice))
        .tool(Arc::new(a2a))
        // TODO: Step 4 - Force the response to use the StoryOutput model
        .structured_output::<StoryOutput>()
        .build()
        .await?;

    let characters_db = std::env::var("CHARACTERS_DB").map_or_else(
        |_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("5_a2a_integration/agents/character_agent/characters.json")
        },
        PathBuf::from,
    );
    let state = AppState {
        agent: Arc::new(Mutex::new(agent)),
        characters_db,
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/messages", get(messages))
        .route("/user/{user_name}", get(user))
        .route("/inquire", post(inquire))
        .with_state(state)
        .layer(cors());

    let addr = bind_addr("GAMEMASTER_BIND", DEFAULT_PORT)?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, %mcp_url, "Game Master listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            if tokio::signal::ctrl_c().await.is_err() {
                std::future::pending::<()>().await;
            }
            tracing::info!("shutting down");
        })
        .await?;
    Ok(())
}

/// Default Bedrock model for the whole adventure.
///
/// Verified live against the configured Bedrock account (us-east-1, 2026-10-18
/// via `aws bedrock list-inference-profiles`): `ACTIVE`. Override it with
/// `MODEL_ID` for a cheaper run, e.g.
/// `export MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0`.
const DEFAULT_MODEL_ID: &str = "us.anthropic.claude-sonnet-5-5";

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

/// Resolve where this server listens: `env_var` if set, else loopback on
/// `default_port`. Loopback by default because these servers have no
/// authentication, the same as the Python chapter.
///
/// # Errors
///
/// Returns the parse error when `env_var` holds something that is not
/// `host:port`.
fn bind_addr(
    env_var: &str,
    default_port: u16,
) -> std::result::Result<SocketAddr, std::net::AddrParseError> {
    match std::env::var(env_var) {
        Ok(addr) => addr.parse(),
        Err(_) => Ok(SocketAddr::from(([127, 0, 0, 1], default_port))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/5_a2a_integration/agents/character_agent/characters.json"
    );

    #[test]
    fn user_lookup_reads_the_tinydb_file_by_exact_name() {
        let db: Value = serde_json::from_slice(&std::fs::read(SEED).unwrap()).unwrap();
        let arnaud = find_character(&db, "Arnaud").unwrap();
        assert_eq!(arnaud["stats"]["constitution"], 14);
        assert!(find_character(&db, "arnaud").is_none());
    }

    #[test]
    fn story_output_accepts_the_raw_model_shape_without_dice() {
        let raw = json!({
            "response": "The tavern falls silent.",
            "actions_suggestions": ["Approach the bar", "Leave"],
            "details": "No tools used"
        });
        let story: StoryOutput = serde_json::from_value(raw).unwrap();
        assert!(story.dice_rolls.is_empty());
    }

    #[test]
    fn story_output_rejects_a_text_dice_result() {
        let raw = json!({
            "response": "", "actions_suggestions": [], "details": "",
            "dice_rolls": [{ "dice_type": "d20", "result": "seventeen", "reason": "attack" }]
        });
        assert!(serde_json::from_value::<StoryOutput>(raw).is_err());
    }

    #[test]
    fn the_schema_carries_the_field_descriptions() {
        let schema = serde_json::to_string(&schemars::schema_for!(StoryOutput)).unwrap();
        assert!(schema.contains("\"title\":\"StoryOutput\""));
        assert!(schema.contains("Brief summary of tools/agents used"));
    }

    #[test]
    fn messages_use_the_python_wire_shape_the_ui_parses() {
        use strands_agents::types::content::ToolUseContent;
        let message = Message::assistant_with_content(vec![
            ContentBlock::text("Rolling."),
            ContentBlock::ToolUse(ToolUseContent {
                tool_use_id: "t1".to_string(),
                name: "StoryOutput".to_string(),
                input: json!({ "response": "Hi" }),
            }),
        ]);
        assert_eq!(
            wire_message(&message),
            json!({
                "role": "assistant",
                "content": [
                    { "text": "Rolling." },
                    { "toolUse": { "toolUseId": "t1", "name": "StoryOutput",
                                   "input": { "response": "Hi" } } }
                ]
            })
        );
    }

    #[test]
    fn the_structured_output_nudge_is_not_a_player_message() {
        assert!(is_structured_output_nudge(&Message::user(
            STRUCTURED_OUTPUT_PROMPT
        )));
        assert!(!is_structured_output_nudge(&Message::user(
            "I open the door"
        )));
    }

    #[test]
    fn only_local_origins_pass_cors() {
        let ok = |o: &str| is_local_origin(&HeaderValue::from_str(o).unwrap());
        assert!(ok("http://localhost:5173"));
        assert!(ok("http://127.0.0.1:4173"));
        assert!(!ok("http://localhost"));
        assert!(!ok("http://localhost:5173.evil.example"));
        assert!(!ok("https://evil.example"));
        assert!(!ok("http://localhost.evil.example:80"));
    }

    #[test]
    fn bind_addr_defaults_to_loopback() {
        let addr = bind_addr("ONCE_UPON_TEST_UNSET_BIND", 8000);
        assert_eq!(addr, Ok(SocketAddr::from(([127, 0, 0, 1], 8000))));
    }

    #[test]
    fn bind_addr_rejects_garbage() {
        // SAFETY: this test is the only reader and writer of this variable.
        unsafe { std::env::set_var("ONCE_UPON_TEST_BAD_BIND", "not-an-address") };
        assert!(bind_addr("ONCE_UPON_TEST_BAD_BIND", 8000).is_err());
    }
}
