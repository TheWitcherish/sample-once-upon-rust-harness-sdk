//! Chapter 5 - The Rules Agent (A2A server on port 8000)
//!
//! One tool, `query_rules`: a semantic search over the Basic Rules PDF that
//! returns the three closest passages with their page numbers. Served over A2A
//! so the Game Master can ask this agent instead of guessing the rules.
//!
//! Start ExtendDB, build the knowledge base once, then run the agent:
//!   extenddb serve --foreground      (its own terminal, from the repo root)
//!   cargo run --release --bin ch5-create-knowledge-base
//!   cargo run --bin ch5-rules-agent
//!
//! Environment overrides: MODEL_ID, EMBEDDING_MODEL_ID, AWS_REGION, RUST_LOG,
//! AWS_ENDPOINT_URL_DYNAMODB (default http://127.0.0.1:18443), RULES_TABLE
//! (default RulesPassages), plus RULES_AGENT_BIND (default 127.0.0.1:8000).

// The knowledge base code this agent shares with the builder sits next to the
// builder, in 5_a2a_integration/utils/.
#[path = "../../utils/knowledge_base.rs"]
mod knowledge_base;

use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use aws_sdk_dynamodb::types::AttributeValue;
use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};
use strands_agents::a2a::server::rpc_dispatch;
use strands_agents::a2a::{AgentSkill, StrandsA2AExecutor};
use strands_agents::models::{BedrockModel, ModelConfig};
use strands_agents::tools::ToolResult;
use strands_agents::{AgentBuilder, Tool, ToolContext, ToolSpec};

use knowledge_base::{
    BUILD_KNOWLEDGE_BASE, Chunk, Embedder, KbResult, KnowledgeBase, METADATA_ID, explain,
    extenddb_client, extenddb_endpoint, number_attr, rules_table, string_attr,
};

const NAME: &str = "Rules Agent";
const DESCRIPTION: &str =
    "5e rules lookup: fast, page-referenced answers from the Basic Rules knowledge base.";
const SYSTEM_PROMPT: &str = "You are a 5e rules expert. For each rules question, call query_rules once, then answer briefly with the page reference.";

const DEFAULT_PORT: u16 = 8000;
const TOP_K: usize = 3;

/// The same layout as the Python tool: `[Page N] passage`, blank-line separated.
fn format_passages(hits: &[(f64, Chunk)]) -> String {
    hits.iter()
        .map(|(_, c)| format!("[Page {}] {}", c.page, c.text))
        .collect::<Vec<_>>()
        .join("\n\n")
}

struct QueryRules {
    kb: KnowledgeBase,
    embedder: Embedder,
}

impl QueryRules {
    async fn lookup(&self, query: &str) -> KbResult<String> {
        let vector = self.embedder.embed(query).await?;
        let hits = self.kb.search(&vector, TOP_K).await?;
        for (score, c) in &hits {
            tracing::info!(page = c.page, score = format!("{score:.3}"), "match");
        }
        Ok(format_passages(&hits))
    }
}

#[async_trait]
impl Tool for QueryRules {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "query_rules",
            "Look up a 5e rule in the Basic Rules knowledge base.\n\n\
             Use it for any question about game mechanics: ability checks, combat, \
             spellcasting, conditions, resting. Ask in plain English, as a player would. \
             The lookup is a semantic search over the Basic Rules PDF: the three \
             passages closest to the question come back, each with its page number.\n\n\
             Example response:\n    \
             \"[Page 74] When a hostile creature that you can see moves out of your reach, ...\n\n    \
             [Page 73] ...\"\n\n\
             Notes:\n    \
             - Passages are about 1,000 characters long; the answer is usually in the first one.\n    \
             - Fails if the knowledge base has not been built yet (cargo run --bin ch5-create-knowledge-base).\n\n\
             Returns the matching passages, each prefixed with its page reference, separated by blank lines.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The rules question or topic, in plain English, e.g. \"opportunity attack\" or \"what are the rules for dexterity checks\"."
                    }
                },
                "required": ["query"]
            }),
        )
    }

    async fn execute(
        &self,
        input: Value,
        _ctx: &mut ToolContext,
    ) -> strands_agents::Result<ToolResult> {
        let Some(query) = input.get("query").and_then(Value::as_str) else {
            return Ok(ToolResult::error("query must be a string"));
        };
        tracing::info!(query, "query_rules");
        Ok(match self.lookup(query).await {
            Ok(text) => ToolResult::success(text),
            Err(e) => ToolResult::error(e.to_string()),
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ch5_rules_agent=info".into()),
        )
        .init();

    // Python opens ChromaDB on the first query; opening the table at startup
    // fails before the server is advertised, not in the middle of a game.
    let kb = KnowledgeBase::open(&extenddb_endpoint(), &rules_table())
        .await
        .map_err(|e| e.to_string())?;
    let embedder = Embedder::from_env().await;
    // Vectors from two embedding models live in different spaces: comparing
    // them returns confident nonsense, so refuse instead.
    if kb.model_id != embedder.model_id() {
        return Err(format!(
            "knowledge base was embedded with {}, but EMBEDDING_MODEL_ID is {}; rebuild it",
            kb.model_id,
            embedder.model_id()
        )
        .into());
    }
    tracing::info!(passages = kb.passages, table = %kb.table, model = %kb.model_id, "knowledge base opened");

    let agent = AgentBuilder::new()
        .model(Box::new(game_master_model().await?))
        // TODO: Step 1 - Add the query_rules tool to the agent
        // TODO: Step 2 - Add the name NAME ("Rules Agent") to the agent
        // TODO: Step 3 - Add the description DESCRIPTION ("5e rules lookup: fast, page-referenced answers from the Basic Rules knowledge base.") to the agent
        .system_prompt(SYSTEM_PROMPT)
        .build()
        .await?;

    // TODO: Step 4 - Create a StrandsA2AExecutor for the agent, with query_rules as a skill

    // TODO: Step 5 - Start the A2A server on port 8000 with serve_a2a
    Ok(())
}

// Opening the table is the one thing only this agent does with the knowledge
// base; the rest is in `utils/knowledge_base.rs`, shared with the builder.
impl KnowledgeBase {
    /// Open the table the builder wrote.
    ///
    /// # Errors
    ///
    /// Fails when ExtendDB is not running or the table does not exist yet
    /// (each with the command that fixes it), or when the build that wrote
    /// the table stopped before the end.
    async fn open(endpoint: &str, table: &str) -> KbResult<Self> {
        let client = extenddb_client(endpoint);
        let found = client
            .get_item()
            .table_name(table)
            .key("id", AttributeValue::S(METADATA_ID.to_string()))
            .consistent_read(true)
            .send()
            .await
            .map_err(|e| explain("GetItem", endpoint, table, e))?;
        let metadata = found.item().ok_or_else(|| {
            format!(
                "the build of table {table} did not finish. Build it again: {BUILD_KNOWLEDGE_BASE}"
            )
        })?;
        Ok(Self {
            model_id: string_attr(metadata, "model_id")?,
            passages: number_attr(metadata, "passages")?,
            client,
            endpoint: endpoint.to_string(),
            table: table.to_string(),
        })
    }
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

/// Resolve where this server listens: `env_var` if set, else loopback on
/// `default_port`. Loopback by default because these servers have no
/// authentication.
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

#[derive(Clone)]
struct A2AState {
    executor: Arc<StrandsA2AExecutor>,
    public_url: String,
}

async fn agent_card(State(app): State<A2AState>) -> Json<Value> {
    Json(
        serde_json::to_value(app.executor.agent_card(&app.public_url))
            .unwrap_or_else(|e| json!({ "error": e.to_string() })),
    )
}

async fn rpc(State(app): State<A2AState>, Json(body): Json<Value>) -> Json<Value> {
    let response = rpc_dispatch(&app.executor, body).await;
    Json(serde_json::to_value(response).unwrap_or_else(|e| {
        json!({ "jsonrpc": "2.0", "id": null,
                "error": { "code": -32603, "message": e.to_string() } })
    }))
}

/// Serve an agent over A2A until Ctrl-C.
///
/// Routes:
/// - `GET /.well-known/agent-card.json`: how other agents discover this one.
/// - `POST /`: JSON-RPC 2.0 (`message/send`, `tasks/get`, ...). Root, because
///   that is where `A2AAgent` and `make_a2a_client` post.
///
/// # Errors
///
/// Returns the I/O error when the address cannot be bound (for example, the
/// port is already taken) or the server stops unexpectedly.
async fn serve_a2a(executor: StrandsA2AExecutor, addr: SocketAddr) -> std::io::Result<()> {
    let public_url = format!("http://{addr}");
    let state = A2AState {
        executor: Arc::new(executor),
        public_url: public_url.clone(),
    };
    let app = Router::new()
        .route("/.well-known/agent-card.json", get(agent_card))
        .route("/", post(rpc))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%public_url, "A2A server listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            // If the signal handler cannot be installed, keep serving rather
            // than exit at once; the process can still be killed.
            if tokio::signal::ctrl_c().await.is_err() {
                std::future::pending::<()>().await;
            }
            tracing::info!("shutting down");
        })
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge_base::{
        DEFAULT_EMBEDDING_MODEL_ID, DEFAULT_EXTENDDB_ENDPOINT, DEFAULT_RULES_TABLE, START_EXTENDDB,
    };
    use aws_sdk_dynamodb::types::{
        AttributeDefinition, BillingMode, KeySchemaElement, KeyType, ScalarAttributeType,
        TableStatus,
    };

    fn chunk(page: usize, text: &str) -> Chunk {
        Chunk {
            id: format!("page_{page}_offset_0"),
            text: text.to_string(),
            page,
        }
    }

    #[test]
    fn passages_are_page_referenced_and_blank_line_separated() {
        let a = chunk(74, "Opportunity attacks.");
        let b = chunk(73, "Movement.");
        assert_eq!(
            format_passages(&[(0.9, a), (0.5, b)]),
            "[Page 74] Opportunity attacks.\n\n[Page 73] Movement."
        );
    }

    #[tokio::test]
    async fn a_query_that_is_not_text_is_a_tool_error() {
        let tool = QueryRules {
            // Never called: the input is refused before any search.
            kb: KnowledgeBase {
                client: extenddb_client(DEFAULT_EXTENDDB_ENDPOINT),
                endpoint: DEFAULT_EXTENDDB_ENDPOINT.to_string(),
                table: String::new(),
                model_id: String::new(),
                passages: 0,
            },
            embedder: Embedder::from_env().await,
        };
        let result = tool
            .execute(json!({ "query": 42 }), &mut ToolContext::default())
            .await
            .unwrap();
        assert!(matches!(
            result.status,
            strands_agents::tools::ToolStatus::Error
        ));
    }

    #[tokio::test]
    async fn an_unreachable_extenddb_names_the_start_command() {
        // Bind a free port, then let it go: nothing answers there.
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let err = KnowledgeBase::open(&format!("http://127.0.0.1:{port}"), DEFAULT_RULES_TABLE)
            .await
            .unwrap_err();
        assert!(err.to_string().contains(START_EXTENDDB), "{err}");
    }

    #[tokio::test]
    #[ignore = "needs ExtendDB running: extenddb serve --foreground, then cargo test -- --ignored"]
    async fn extenddb_opens_only_a_finished_table() {
        let endpoint = extenddb_endpoint();
        let table = format!("test-{}", uuid::Uuid::new_v4());
        let err = KnowledgeBase::open(&endpoint, &table).await.unwrap_err();
        assert!(err.to_string().contains(BUILD_KNOWLEDGE_BASE), "{err}");

        // `open` only reads the metadata item, so a table without a vector
        // index stands in for the builder's.
        let client = extenddb_client(&endpoint);
        client
            .create_table()
            .table_name(&table)
            .attribute_definitions(
                AttributeDefinition::builder()
                    .attribute_name("id")
                    .attribute_type(ScalarAttributeType::S)
                    .build()
                    .unwrap(),
            )
            .key_schema(
                KeySchemaElement::builder()
                    .attribute_name("id")
                    .key_type(KeyType::Hash)
                    .build()
                    .unwrap(),
            )
            .billing_mode(BillingMode::PayPerRequest)
            .send()
            .await
            .unwrap();
        // A new table starts CREATING and turns ACTIVE a moment later.
        for _ in 0..600 {
            let described = client
                .describe_table()
                .table_name(&table)
                .send()
                .await
                .unwrap();
            if described.table().and_then(|t| t.table_status()) == Some(&TableStatus::Active) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        // No metadata item: the build stopped halfway.
        let err = KnowledgeBase::open(&endpoint, &table).await.unwrap_err();
        assert!(err.to_string().contains("did not finish"), "{err}");

        client
            .put_item()
            .table_name(&table)
            .item("id", AttributeValue::S(METADATA_ID.to_string()))
            .item(
                "model_id",
                AttributeValue::S(DEFAULT_EMBEDDING_MODEL_ID.to_string()),
            )
            .item("passages", AttributeValue::N("3".to_string()))
            .send()
            .await
            .unwrap();
        let kb = KnowledgeBase::open(&endpoint, &table).await.unwrap();
        assert_eq!(kb.model_id, DEFAULT_EMBEDDING_MODEL_ID);
        assert_eq!(kb.passages, 3);

        client
            .delete_table()
            .table_name(&table)
            .send()
            .await
            .unwrap();
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
