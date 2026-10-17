//! Chapter 5 - The Rules Agent (A2A server on port 8000)
//!
//! A side-by-side port of the Python workshop's
//! `5_a2a_integration/agents/rules_agent/rules_agent.py`:
//!   <https://github.com/aws-samples/sample-once-upon-agentic-ai/tree/main/5_a2a_integration>
//!
//! One tool, `query_rules`: a semantic search over the Basic Rules PDF that
//! returns the three closest passages with their page numbers. Served over A2A
//! so the Game Master can ask this agent instead of guessing the rules.
//!
//! Build the knowledge base once, then run the agent:
//!   cargo run --release --bin ch5-create-knowledge-base
//!   cargo run --bin ch5-rules-agent
//!
//! Environment overrides: MODEL_ID, EMBEDDING_MODEL_ID, AWS_REGION, RUST_LOG,
//! plus RULES_AGENT_BIND (default 127.0.0.1:8000).

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use aws_sdk_bedrockruntime::Client;
use aws_sdk_bedrockruntime::error::DisplayErrorContext;
use aws_sdk_bedrockruntime::primitives::Blob;
use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use strands_agents::a2a::server::rpc_dispatch;
use strands_agents::a2a::{AgentSkill, StrandsA2AExecutor};
use strands_agents::models::{BedrockModel, ModelConfig};
use strands_agents::tools::ToolResult;
use strands_agents::{AgentBuilder, Tool, ToolContext, ToolSpec};

const NAME: &str = "Rules Agent";
const DESCRIPTION: &str =
    "5e rules lookup: fast, page-referenced answers from the Basic Rules knowledge base.";
const SYSTEM_PROMPT: &str = "You are a 5e rules expert. For each rules question, call query_rules once, then answer briefly with the page reference.";

const DEFAULT_PORT: u16 = 8000;
const TOP_K: usize = 3;

/// The same layout as the Python tool: `[Page N] passage`, blank-line separated.
fn format_passages(hits: &[(f32, &Passage)]) -> String {
    hits.iter()
        .map(|(_, p)| format!("[Page {}] {}", p.chunk.page, p.chunk.text))
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
        let hits = self.kb.search(&vector, TOP_K);
        for (score, p) in &hits {
            tracing::info!(page = p.chunk.page, score = format!("{score:.3}"), "match");
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

    // Python opens ChromaDB on the first query; loading at startup fails
    // before the server is advertised, not in the middle of a game.
    let kb = KnowledgeBase::load(&kb_path()).map_err(|e| e.to_string())?;
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
    tracing::info!(passages = kb.passages.len(), model = %kb.model_id, "knowledge base loaded");

    let agent = AgentBuilder::new()
        .model(Box::new(game_master_model().await?))
        // TODO: Step 1 - Add the query_rules tool to the agent
        .tool(Arc::new(QueryRules { kb, embedder }))
        // TODO: Step 2 - Add the name NAME ("Rules Agent") to the agent
        .name(NAME)
        // TODO: Step 3 - Add the description DESCRIPTION ("5e rules lookup: fast, page-referenced answers from the Basic Rules knowledge base.") to the agent
        .description(DESCRIPTION)
        .system_prompt(SYSTEM_PROMPT)
        .build()
        .await?;

    // TODO: Step 4 - Create a StrandsA2AExecutor for the agent, with query_rules as a skill
    let executor = StrandsA2AExecutor::for_agent(agent, NAME, DESCRIPTION)
        .with_version(env!("CARGO_PKG_VERSION"))
        .with_skill(AgentSkill {
            id: "query_rules".to_string(),
            name: "query_rules".to_string(),
            description: "Look up a 5e rule in the Basic Rules knowledge base.".to_string(),
            tags: Vec::new(),
        });

    // TODO: Step 5 - Start the A2A server on port 8000 with serve_a2a
    serve_a2a(executor, bind_addr("RULES_AGENT_BIND", DEFAULT_PORT)?).await?;
    Ok(())
}

// The knowledge base on disk, the Rust side of the Python chapter's ChromaDB
// collection. The builder writes it and this agent reads it, so both files
// carry the same types and the same `Embedder`: keep the two copies identical.
type KbResult<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Verified live (us-east-1, 2026-10-08, `aws bedrock get-foundation-model`):
/// `ACTIVE`, output modality `EMBEDDING`. Override with `EMBEDDING_MODEL_ID`.
const DEFAULT_EMBEDDING_MODEL_ID: &str = "amazon.titan-embed-text-v2:0";
/// Titan V2 offers 256, 512 or 1024. 512 keeps the file small and search sharp.
const EMBEDDING_DIMENSIONS: usize = 512;

fn embedding_model_id() -> String {
    std::env::var("EMBEDDING_MODEL_ID").unwrap_or_else(|_| DEFAULT_EMBEDDING_MODEL_ID.to_string())
}

fn utils_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("5_a2a_integration/utils")
}

fn kb_path() -> PathBuf {
    utils_dir().join("rules_knowledge_base.json")
}

/// A passage cut from the PDF, before it is embedded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Chunk {
    id: String,
    text: String,
    page: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Passage {
    #[serde(flatten)]
    chunk: Chunk,
    embedding: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct KnowledgeBase {
    source: String,
    model_id: String,
    dimensions: usize,
    passages: Vec<Passage>,
}

impl KnowledgeBase {
    /// # Errors
    ///
    /// Fails when the file is missing (with the command that builds it) or is
    /// not a knowledge base.
    fn load(path: &Path) -> KbResult<Self> {
        let bytes = std::fs::read(path).map_err(|e| {
            format!(
                "cannot read {}: {e}. Build it first: cargo run --bin ch5-create-knowledge-base",
                path.display()
            )
        })?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    /// The `k` passages closest to `query`, best first. Titan returns unit
    /// vectors (`normalize: true`), so the dot product is the cosine similarity.
    fn search(&self, query: &[f32], k: usize) -> Vec<(f32, &Passage)> {
        let mut scored: Vec<(f32, &Passage)> = self
            .passages
            .iter()
            .map(|p| (dot(&p.embedding, query), p))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.truncate(k);
        scored
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Titan Text Embeddings V2 on Bedrock. `Clone` is cheap (the SDK client is
/// reference-counted), so the builder hands one copy to each concurrent task.
#[derive(Debug, Clone)]
struct Embedder {
    client: Client,
    model_id: String,
}

impl Embedder {
    /// Honours `EMBEDDING_MODEL_ID` and the standard AWS environment
    /// (`AWS_REGION`, `AWS_PROFILE`, ...).
    async fn from_env() -> Self {
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        Self {
            client: Client::new(&config),
            model_id: embedding_model_id(),
        }
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    /// # Errors
    ///
    /// Fails when Bedrock rejects the call or answers without an embedding.
    async fn embed(&self, text: &str) -> KbResult<Vec<f32>> {
        let body = json!({
            "inputText": text,
            "dimensions": EMBEDDING_DIMENSIONS,
            "normalize": true,
        });
        let response = self
            .client
            .invoke_model()
            .model_id(&self.model_id)
            .content_type("application/json")
            .accept("application/json")
            .body(Blob::new(serde_json::to_vec(&body)?))
            .send()
            .await
            // DisplayErrorContext walks the source chain; plain Display on a
            // credential or connection failure only says "unhandled error".
            .map_err(|e| format!("Bedrock embedding call failed: {}", DisplayErrorContext(&e)))?;
        let answer: Value = serde_json::from_slice(response.body().as_ref())?;
        let embedding: Vec<f32> = serde_json::from_value(
            answer
                .get("embedding")
                .cloned()
                .ok_or("Titan answered without an embedding")?,
        )?;
        Ok(embedding)
    }
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

/// Serve an agent over A2A until Ctrl-C: the Rust side of Python's
/// `A2AServer(agent=..., port=...).serve()`.
///
/// Routes:
/// - `GET /.well-known/agent-card.json`: how other agents discover this one.
/// - `POST /`: JSON-RPC 2.0 (`message/send`, `tasks/get`, ...). Root, like the
///   Python server, because that is where `A2AAgent` and `make_a2a_client` post.
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

    fn passage(page: usize, text: &str) -> Passage {
        Passage {
            chunk: Chunk {
                id: format!("page_{page}_offset_0"),
                text: text.to_string(),
                page,
            },
            embedding: Vec::new(),
        }
    }

    #[test]
    fn passages_are_page_referenced_and_blank_line_separated() {
        let a = passage(74, "Opportunity attacks.");
        let b = passage(73, "Movement.");
        assert_eq!(
            format_passages(&[(0.9, &a), (0.5, &b)]),
            "[Page 74] Opportunity attacks.\n\n[Page 73] Movement."
        );
    }

    #[tokio::test]
    async fn a_query_that_is_not_text_is_a_tool_error() {
        let tool = QueryRules {
            kb: KnowledgeBase {
                source: String::new(),
                model_id: String::new(),
                dimensions: 0,
                passages: Vec::new(),
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

    fn embedded(id: &str, embedding: Vec<f32>) -> Passage {
        Passage {
            chunk: Chunk {
                id: id.to_string(),
                text: id.to_string(),
                page: 1,
            },
            embedding,
        }
    }

    #[test]
    fn search_returns_the_closest_first() {
        let kb = KnowledgeBase {
            source: String::new(),
            model_id: DEFAULT_EMBEDDING_MODEL_ID.to_string(),
            dimensions: 2,
            passages: vec![
                embedded("far", vec![0.0, 1.0]),
                embedded("near", vec![1.0, 0.0]),
                embedded("middle", vec![0.7, 0.7]),
            ],
        };
        let ids: Vec<&str> = kb
            .search(&[1.0, 0.0], 2)
            .iter()
            .map(|(_, p)| p.chunk.id.as_str())
            .collect();
        assert_eq!(ids, ["near", "middle"]);
    }

    #[test]
    fn a_written_knowledge_base_loads_back() {
        let path = std::env::temp_dir().join(format!("kb-{}.json", uuid::Uuid::new_v4()));
        let kb = KnowledgeBase {
            source: String::new(),
            model_id: DEFAULT_EMBEDDING_MODEL_ID.to_string(),
            dimensions: 1,
            passages: vec![embedded("only", vec![1.0])],
        };
        std::fs::write(&path, serde_json::to_vec(&kb).unwrap()).unwrap();
        assert_eq!(KnowledgeBase::load(&path).unwrap(), kb);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn a_missing_file_names_the_build_command() {
        let err = KnowledgeBase::load(Path::new("/nonexistent/kb.json")).unwrap_err();
        assert!(err.to_string().contains("ch5-create-knowledge-base"));
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
