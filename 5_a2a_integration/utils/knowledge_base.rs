//! Chapter 5 - The knowledge base code the builder and the Rules Agent share
//!
//! The Rust side of the Python chapter's ChromaDB collection: Basic Rules
//! passages and their Titan embeddings, in a local ExtendDB table with a vector
//! index. ExtendDB answers the DynamoDB API, so this is plain `aws-sdk-dynamodb`
//! code.
//!
//! Not a program of its own: `create_knowledge_base.rs` and `rules_agent.rs`
//! each pull this file in with a `mod knowledge_base;` declaration, so both
//! reach ExtendDB, read passages and embed text with the same code. What only
//! one of them does stays in its own file: the builder creates and fills the
//! table, the Rules Agent opens it.

use std::collections::HashMap;

use aws_sdk_bedrockruntime::Client;
use aws_sdk_bedrockruntime::error::DisplayErrorContext;
use aws_sdk_bedrockruntime::primitives::Blob;
use aws_sdk_dynamodb as dynamodb;
use dynamodb::config::{BehaviorVersion, Credentials, Region};
use dynamodb::error::{ProvideErrorMetadata, SdkError};
use dynamodb::types::AttributeValue;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub type KbResult<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Verified live (us-west-2, 2026-10-19, `aws bedrock get-foundation-model`):
/// `ACTIVE`, output modality `EMBEDDING`. Override with `EMBEDDING_MODEL_ID`.
pub const DEFAULT_EMBEDDING_MODEL_ID: &str = "amazon.titan-embed-text-v2:0";
/// Titan V2 offers 256, 512 or 1024. 512 keeps the table small and search sharp.
pub const EMBEDDING_DIMENSIONS: usize = 512;

/// ExtendDB's developer mode: plain HTTP, loopback only. Override with the AWS
/// SDKs' standard `AWS_ENDPOINT_URL_DYNAMODB`.
pub const DEFAULT_EXTENDDB_ENDPOINT: &str = "http://127.0.0.1:18443";
/// Override with `RULES_TABLE`.
pub const DEFAULT_RULES_TABLE: &str = "RulesPassages";
pub const VECTOR_INDEX: &str = "rules-embedding";
/// Not a passage: the item that records which model embedded the table. The
/// builder writes it last, so a build that stopped halfway has none.
pub const METADATA_ID: &str = "knowledge_base";
/// Developer mode seeds AWS's documented example key pair, prints it at startup
/// and checks every request's signature against it. It opens nothing but a
/// local developer-mode server, so it is not a secret.
const DEV_ACCESS_KEY_ID: &str = "AKIAIOSFODNN7EXAMPLE";
const DEV_SECRET_ACCESS_KEY: &str = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
/// ExtendDB's own region, which every request signature names.
const EXTENDDB_REGION: &str = "us-east-1";
pub const START_EXTENDDB: &str = "extenddb serve --foreground";
pub const BUILD_KNOWLEDGE_BASE: &str = "cargo run --release --bin ch5-create-knowledge-base";

fn embedding_model_id() -> String {
    std::env::var("EMBEDDING_MODEL_ID").unwrap_or_else(|_| DEFAULT_EMBEDDING_MODEL_ID.to_string())
}

pub fn extenddb_endpoint() -> String {
    std::env::var("AWS_ENDPOINT_URL_DYNAMODB")
        .unwrap_or_else(|_| DEFAULT_EXTENDDB_ENDPOINT.to_string())
}

pub fn rules_table() -> String {
    std::env::var("RULES_TABLE").unwrap_or_else(|_| DEFAULT_RULES_TABLE.to_string())
}

/// A passage cut from the PDF, before it is embedded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub text: String,
    pub page: usize,
}

/// A DynamoDB client for ExtendDB's developer mode. Nothing is sent before the
/// first call.
pub fn extenddb_client(endpoint: &str) -> dynamodb::Client {
    let config = dynamodb::Config::builder()
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new(EXTENDDB_REGION))
        .endpoint_url(endpoint)
        .credentials_provider(Credentials::new(
            DEV_ACCESS_KEY_ID,
            DEV_SECRET_ACCESS_KEY,
            None,
            None,
            "extenddb-dev-mode",
        ))
        .build();
    dynamodb::Client::from_conf(config)
}

/// The table the builder writes and the Rules Agent searches.
#[derive(Debug, Clone)]
pub struct KnowledgeBase {
    pub client: dynamodb::Client,
    pub endpoint: String,
    pub table: String,
    /// The embedding model the passages were embedded with.
    pub model_id: String,
    pub passages: usize,
}

impl KnowledgeBase {
    /// The `k` passages closest to `query`, best first, each with its cosine
    /// similarity (1.0 means the same direction).
    ///
    /// # Errors
    ///
    /// Fails when ExtendDB is unreachable or refuses the query, for example a
    /// vector whose length is not the index's dimensions.
    pub async fn search(&self, query: &[f32], k: usize) -> KbResult<Vec<(f64, Chunk)>> {
        let found = self
            .client
            .search_vectors()
            .table_name(&self.table)
            .index_name(VECTOR_INDEX)
            .set_search_vector(Some(vector(query)))
            .top_k(i32::try_from(k)?)
            .send()
            .await
            .map_err(|e| explain("SearchVectors", &self.endpoint, &self.table, e))?;
        found
            .search_results()
            .iter()
            .map(|hit| {
                let item = hit
                    .item()
                    .ok_or("a search result came back without its passage")?;
                // A COSINE score is a distance (0 for the same direction).
                // Turn it back into a similarity, so higher still means closer.
                Ok((1.0 - hit.score(), chunk_from_item(item)?))
            })
            .collect()
    }
}

/// Name the fix for the two failures a reader meets first: ExtendDB is not
/// running, or the knowledge base is not built yet.
pub fn explain<E>(call: &str, endpoint: &str, table: &str, err: SdkError<E>) -> String
where
    E: ProvideErrorMetadata + std::error::Error + 'static,
{
    match &err {
        SdkError::DispatchFailure(_) | SdkError::TimeoutError(_) => format!(
            "{call} failed: cannot reach ExtendDB at {endpoint} ({}). Start it first: {START_EXTENDDB}",
            causes(&err)
        ),
        SdkError::ServiceError(_) if err.code() == Some("ResourceNotFoundException") => {
            format!(
                "{call} failed: there is no table {table} yet. Build it first: {BUILD_KNOWLEDGE_BASE}"
            )
        }
        // ExtendDB refused and said why: its code and message are the cause.
        SdkError::ServiceError(_) => format!(
            "{call} failed: {}: {}",
            err.code().unwrap_or("unknown error"),
            err.message().unwrap_or("no message")
        ),
        _ => format!("{call} failed: {}", DisplayErrorContext(&err)),
    }
}

/// Plain Display on a connection failure only says "dispatch failure": the
/// reason sits at the end of its chain of sources.
fn causes(err: &dyn std::error::Error) -> String {
    let mut text = err.to_string();
    let mut source = err.source();
    while let Some(cause) = source {
        text.push_str(": ");
        text.push_str(&cause.to_string());
        source = cause.source();
    }
    text
}

/// DynamoDB stores a vector as a list (`L`) of numbers (`N`).
pub fn vector(values: &[f32]) -> Vec<AttributeValue> {
    values
        .iter()
        .map(|v| AttributeValue::N(v.to_string()))
        .collect()
}

pub fn chunk_from_item(item: &HashMap<String, AttributeValue>) -> KbResult<Chunk> {
    Ok(Chunk {
        id: string_attr(item, "id")?,
        text: string_attr(item, "text")?,
        page: number_attr(item, "page")?,
    })
}

pub fn string_attr(item: &HashMap<String, AttributeValue>, name: &str) -> KbResult<String> {
    let value = item
        .get(name)
        .and_then(|v| v.as_s().ok())
        .ok_or_else(|| format!("item without a text attribute {name}"))?;
    Ok(value.clone())
}

pub fn number_attr(item: &HashMap<String, AttributeValue>, name: &str) -> KbResult<usize> {
    let value = item
        .get(name)
        .and_then(|v| v.as_n().ok())
        .ok_or_else(|| format!("item without a number attribute {name}"))?;
    Ok(value.parse()?)
}

/// Titan Text Embeddings V2 on Bedrock. `Clone` is cheap (the SDK client is
/// reference-counted), so the builder hands one copy to each concurrent task.
#[derive(Debug, Clone)]
pub struct Embedder {
    client: Client,
    model_id: String,
}

impl Embedder {
    /// Honours `EMBEDDING_MODEL_ID` and the standard AWS environment
    /// (`AWS_REGION`, `AWS_PROFILE`, ...).
    pub async fn from_env() -> Self {
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        Self {
            client: Client::new(&config),
            model_id: embedding_model_id(),
        }
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// # Errors
    ///
    /// Fails when Bedrock rejects the call or answers without an embedding.
    pub async fn embed(&self, text: &str) -> KbResult<Vec<f32>> {
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
