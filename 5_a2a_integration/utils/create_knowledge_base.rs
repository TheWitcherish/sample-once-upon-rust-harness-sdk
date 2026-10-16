//! Chapter 5 - Build the Rules Agent knowledge base
//!
//! Reads `DnD_BasicRules_2018.pdf` page by page, cuts each page into
//! overlapping ~1000-character passages, embeds every passage with Titan Text
//! Embeddings V2 on Bedrock, and stores them in the `RulesPassages` table of a
//! local ExtendDB, behind a vector index. Run it once, with ExtendDB running
//! (`extenddb serve --foreground`) and the PDF in `5_a2a_integration/utils/`:
//!   cargo run --release --bin ch5-create-knowledge-base
//!
//! Environment overrides: EMBEDDING_MODEL_ID, AWS_REGION, AWS_PROFILE,
//! AWS_ENDPOINT_URL_DYNAMODB (default http://127.0.0.1:18443), RULES_TABLE.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use aws_sdk_bedrockruntime::Client;
use aws_sdk_bedrockruntime::error::DisplayErrorContext;
use aws_sdk_bedrockruntime::primitives::Blob;
use aws_sdk_dynamodb as dynamodb;
use dynamodb::config::{BehaviorVersion, Credentials, Region};
use dynamodb::error::{ProvideErrorMetadata, SdkError};
use dynamodb::types::{
    AttributeDefinition, AttributeValue, BillingMode, IndexStatus, KeySchemaElement, KeyType,
    Projection, ProjectionType, ScalarAttributeType, TableStatus, VectorAttributeDefinition,
    VectorDistanceFunction, VectorIndex,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::task::JoinSet;

/// Embedding calls in flight at once. Enough to finish in about a minute,
/// few enough to stay well under the Bedrock request quota.
const CONCURRENCY: usize = 10;
const PROGRESS_EVERY: usize = 100;

async fn create_knowledge_base() -> KbResult<()> {
    let pdf = pdf_path();
    if !pdf.exists() {
        return Err(format!(
            "'{}' not found. Put {PDF_FILE} in 5_a2a_integration/utils/ first (see the workshop instructions).",
            pdf.display()
        )
        .into());
    }

    let embedder = Embedder::from_env().await;
    // The table comes first: if ExtendDB is not running, this fails now, not
    // after a minute of embedding calls.
    let (endpoint, table) = (extenddb_endpoint(), rules_table());
    println!("Creating table {table} on ExtendDB at {endpoint}...");
    let mut kb = KnowledgeBase::create(&endpoint, &table, embedder.model_id()).await?;

    println!("Extracting text from the PDF...");
    let pages = pdf_extract::extract_text_by_pages(&pdf)?;
    let chunks = extract_chunks(&pages);
    println!(
        "{} passages extracted from {} pages",
        chunks.len(),
        pages.len()
    );

    println!(
        "Embedding passages with {} ({EMBEDDING_DIMENSIONS} dimensions)...",
        embedder.model_id()
    );
    let mut passages = Vec::with_capacity(chunks.len());
    let mut next_report = PROGRESS_EVERY;
    for group in chunks.chunks(CONCURRENCY) {
        let mut tasks = JoinSet::new();
        for (slot, chunk) in group.iter().cloned().enumerate() {
            let embedder = embedder.clone();
            tasks.spawn(async move {
                let embedding = embedder.embed(&chunk.text).await;
                (slot, chunk, embedding)
            });
        }
        // Tasks finish in any order; put them back in PDF order.
        let mut done = Vec::with_capacity(group.len());
        while let Some(joined) = tasks.join_next().await {
            let (slot, chunk, embedding) = joined?;
            done.push((
                slot,
                Passage {
                    chunk,
                    embedding: embedding?,
                },
            ));
        }
        done.sort_by_key(|(slot, _)| *slot);
        passages.extend(done.into_iter().map(|(_, p)| p));

        if passages.len() >= next_report || passages.len() == chunks.len() {
            println!("  {}/{}", passages.len(), chunks.len());
            next_report += PROGRESS_EVERY;
        }
    }

    println!("Storing the passages in {table}...");
    kb.store(&passages).await?;
    println!(
        "Knowledge base created in table {table} ({} passages)",
        kb.passages
    );
    Ok(())
}

#[tokio::main]
async fn main() -> KbResult<()> {
    create_knowledge_base().await?;
    println!("Knowledge base creation complete!");
    Ok(())
}

// The knowledge base, the Rust side of the Python chapter's ChromaDB
// collection: Basic Rules passages and their Titan embeddings, in a local
// ExtendDB table with a vector index. ExtendDB answers the DynamoDB API, so
// this is plain `aws-sdk-dynamodb` code. This builder writes the table and the
// Rules Agent searches it, so what both files need (the types, the ExtendDB
// client, the search and the `Embedder`) is the same code in each: keep the
// two copies identical.
type KbResult<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Verified live (us-west-2, 2026-10-19, `aws bedrock get-foundation-model`):
/// `ACTIVE`, output modality `EMBEDDING`. Override with `EMBEDDING_MODEL_ID`.
const DEFAULT_EMBEDDING_MODEL_ID: &str = "amazon.titan-embed-text-v2:0";
/// Titan V2 offers 256, 512 or 1024. 512 keeps the table small and search sharp.
const EMBEDDING_DIMENSIONS: usize = 512;

/// ExtendDB's developer mode: plain HTTP, loopback only. Override with the AWS
/// SDKs' standard `AWS_ENDPOINT_URL_DYNAMODB`.
const DEFAULT_EXTENDDB_ENDPOINT: &str = "http://127.0.0.1:18443";
/// Override with `RULES_TABLE`.
const DEFAULT_RULES_TABLE: &str = "RulesPassages";
const VECTOR_INDEX: &str = "rules-embedding";
/// Not a passage: the item that records which model embedded the table. The
/// builder writes it last, so a build that stopped halfway has none.
const METADATA_ID: &str = "knowledge_base";
/// Developer mode seeds AWS's documented example key pair, prints it at startup
/// and checks every request's signature against it. It opens nothing but a
/// local developer-mode server, so it is not a secret.
const DEV_ACCESS_KEY_ID: &str = "AKIAIOSFODNN7EXAMPLE";
const DEV_SECRET_ACCESS_KEY: &str = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
/// ExtendDB's own region, which every request signature names.
const EXTENDDB_REGION: &str = "us-east-1";
const START_EXTENDDB: &str = "extenddb serve --foreground";
const BUILD_KNOWLEDGE_BASE: &str = "cargo run --release --bin ch5-create-knowledge-base";
const POLL_EVERY: Duration = Duration::from_millis(200);
const WAIT_LIMIT: Duration = Duration::from_secs(60);

const PDF_FILE: &str = "DnD_BasicRules_2018.pdf";
/// Printed on almost every page; useless for search, so it is stripped.
const FOOTER: &str = "D&D Basic Rules (Version 1.0). Not for resale. Permission granted to print and photocopy this document for personal use only.";
const CHUNK_SIZE: usize = 1000; // characters per passage
const CHUNK_OVERLAP: usize = 200; // shared by two consecutive passages, so a rule cut in two stays findable
const MIN_PASSAGE: usize = 50;

fn embedding_model_id() -> String {
    std::env::var("EMBEDDING_MODEL_ID").unwrap_or_else(|_| DEFAULT_EMBEDDING_MODEL_ID.to_string())
}

fn extenddb_endpoint() -> String {
    std::env::var("AWS_ENDPOINT_URL_DYNAMODB")
        .unwrap_or_else(|_| DEFAULT_EXTENDDB_ENDPOINT.to_string())
}

fn rules_table() -> String {
    std::env::var("RULES_TABLE").unwrap_or_else(|_| DEFAULT_RULES_TABLE.to_string())
}

fn utils_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("5_a2a_integration/utils")
}

fn pdf_path() -> PathBuf {
    utils_dir().join(PDF_FILE)
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

fn squash_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Cut each page into overlapping passages, counted in characters (not bytes)
/// so a cut never splits a multi-byte character.
fn extract_chunks(pages: &[String]) -> Vec<Chunk> {
    let footer = squash_whitespace(FOOTER);
    let mut chunks = Vec::new();
    for (index, page) in pages.iter().enumerate() {
        let page_number = index + 1;
        let text: Vec<char> = squash_whitespace(&squash_whitespace(page).replace(&footer, " "))
            .chars()
            .collect();
        let mut start = 0;
        while start < text.len() {
            let mut end = (start + CHUNK_SIZE).min(text.len());
            if end < text.len() {
                // Cut on a space, not in the middle of a word.
                if let Some(cut) = text[start + CHUNK_SIZE / 2..end]
                    .iter()
                    .rposition(|c| *c == ' ')
                {
                    end = start + CHUNK_SIZE / 2 + cut;
                }
            }
            let passage: String = text[start..end]
                .iter()
                .collect::<String>()
                .trim()
                .to_string();
            if passage.chars().count() > MIN_PASSAGE {
                chunks.push(Chunk {
                    id: format!("page_{page_number}_offset_{start}"),
                    text: passage,
                    page: page_number,
                });
            }
            if end == text.len() {
                break;
            }
            start = end - CHUNK_OVERLAP;
        }
    }
    chunks
}

/// A DynamoDB client for ExtendDB's developer mode. Nothing is sent before the
/// first call.
fn extenddb_client(endpoint: &str) -> dynamodb::Client {
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
struct KnowledgeBase {
    client: dynamodb::Client,
    endpoint: String,
    table: String,
    /// The embedding model the passages were embedded with.
    model_id: String,
    passages: usize,
}

impl KnowledgeBase {
    /// Drop `table` if it exists, then create it empty, with its vector index.
    ///
    /// # Errors
    ///
    /// Fails when ExtendDB is not running (with the command that starts it),
    /// or when the table is not ready within a minute.
    async fn create(endpoint: &str, table: &str, model_id: &str) -> KbResult<Self> {
        let client = extenddb_client(endpoint);
        match client.delete_table().table_name(table).send().await {
            Ok(_) => wait_for(&client, endpoint, table, TableState::Missing).await?,
            Err(e) if e.code() == Some("ResourceNotFoundException") => {}
            Err(e) => return Err(explain("DeleteTable", endpoint, table, e).into()),
        }
        client
            .create_table()
            .table_name(table)
            .attribute_definitions(
                AttributeDefinition::builder()
                    .attribute_name("id")
                    .attribute_type(ScalarAttributeType::S)
                    .build()?,
            )
            .key_schema(
                KeySchemaElement::builder()
                    .attribute_name("id")
                    .key_type(KeyType::Hash)
                    .build()?,
            )
            .billing_mode(BillingMode::PayPerRequest)
            .vector_indexes(vector_index()?)
            .send()
            .await
            .map_err(|e| explain("CreateTable", endpoint, table, e))?;
        wait_for(&client, endpoint, table, TableState::Ready).await?;
        Ok(Self {
            client,
            endpoint: endpoint.to_string(),
            table: table.to_string(),
            model_id: model_id.to_string(),
            passages: 0,
        })
    }

    /// Write every passage, wait until the vector index has caught up, then
    /// write the metadata item that marks the build as finished.
    ///
    /// # Errors
    ///
    /// Fails when ExtendDB rejects a write, for example an embedding whose
    /// length is not the index's dimensions, or when the index has not caught
    /// up within a minute.
    async fn store(&mut self, passages: &[Passage]) -> KbResult<()> {
        for passage in passages {
            self.put(passage_item(passage)).await?;
        }
        if let Some(last) = passages.last() {
            self.wait_until_searchable(last).await?;
        }
        self.put(metadata_item(&self.model_id, passages.len()))
            .await?;
        self.passages = passages.len();
        Ok(())
    }

    /// A vector index is eventually consistent: ExtendDB queues each write and
    /// applies the queue in order, so once a search finds the last passage, it
    /// finds them all.
    async fn wait_until_searchable(&self, passage: &Passage) -> KbResult<()> {
        let started = Instant::now();
        loop {
            let hits = self.search(&passage.embedding, 3).await?;
            if hits.iter().any(|(_, c)| c.id == passage.chunk.id) {
                return Ok(());
            }
            if started.elapsed() > WAIT_LIMIT {
                return Err(format!(
                    "the vector index of {} still misses {} after a minute",
                    self.table, passage.chunk.id
                )
                .into());
            }
            tokio::time::sleep(POLL_EVERY).await;
        }
    }

    async fn put(&self, item: HashMap<String, AttributeValue>) -> KbResult<()> {
        self.client
            .put_item()
            .table_name(&self.table)
            .set_item(Some(item))
            .send()
            .await
            .map_err(|e| explain("PutItem", &self.endpoint, &self.table, e))?;
        Ok(())
    }

    /// The `k` passages closest to `query`, best first, each with its cosine
    /// similarity (1.0 means the same direction).
    ///
    /// # Errors
    ///
    /// Fails when ExtendDB is unreachable or refuses the query, for example a
    /// vector whose length is not the index's dimensions.
    async fn search(&self, query: &[f32], k: usize) -> KbResult<Vec<(f64, Chunk)>> {
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

/// Cosine is what the DynamoDB documentation recommends for Titan text
/// embeddings. The index keeps `page` and `text` next to each vector, so a
/// search returns whole passages in one call.
fn vector_index() -> KbResult<VectorIndex> {
    Ok(VectorIndex::builder()
        .index_name(VECTOR_INDEX)
        .vector_attribute(
            VectorAttributeDefinition::builder()
                .attribute_name("embedding")
                .build()?,
        )
        .dimensions(i64::try_from(EMBEDDING_DIMENSIONS)?)
        .distance_function(VectorDistanceFunction::Cosine)
        .projection(
            Projection::builder()
                .projection_type(ProjectionType::Include)
                .non_key_attributes("page")
                .non_key_attributes("text")
                .build(),
        )
        .build()?)
}

/// Where a table stands, as far as a search is concerned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TableState {
    Missing,
    NotReady,
    Ready,
}

async fn table_state(
    client: &dynamodb::Client,
    endpoint: &str,
    table: &str,
) -> KbResult<TableState> {
    let described = match client.describe_table().table_name(table).send().await {
        Ok(described) => described,
        Err(e) if e.code() == Some("ResourceNotFoundException") => return Ok(TableState::Missing),
        Err(e) => return Err(explain("DescribeTable", endpoint, table, e).into()),
    };
    let ready = described.table().is_some_and(|t| {
        t.table_status() == Some(&TableStatus::Active)
            && t.vector_indexes().iter().any(|index| {
                index.index_name() == Some(VECTOR_INDEX)
                    && index.index_status() == Some(&IndexStatus::Active)
                    && index.backfilling() != Some(true)
            })
    });
    Ok(if ready {
        TableState::Ready
    } else {
        TableState::NotReady
    })
}

/// Tables are created and deleted in the background: poll until `table`
/// reaches `wanted`, for at most a minute.
async fn wait_for(
    client: &dynamodb::Client,
    endpoint: &str,
    table: &str,
    wanted: TableState,
) -> KbResult<()> {
    let started = Instant::now();
    loop {
        let state = table_state(client, endpoint, table).await?;
        if state == wanted {
            return Ok(());
        }
        if started.elapsed() > WAIT_LIMIT {
            return Err(format!("table {table} is still {state:?} after a minute").into());
        }
        tokio::time::sleep(POLL_EVERY).await;
    }
}

/// Name the fix for the two failures a reader meets first: ExtendDB is not
/// running, or the knowledge base is not built yet.
fn explain<E>(call: &str, endpoint: &str, table: &str, err: SdkError<E>) -> String
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

fn passage_item(passage: &Passage) -> HashMap<String, AttributeValue> {
    HashMap::from([
        (
            "id".to_string(),
            AttributeValue::S(passage.chunk.id.clone()),
        ),
        (
            "page".to_string(),
            AttributeValue::N(passage.chunk.page.to_string()),
        ),
        (
            "text".to_string(),
            AttributeValue::S(passage.chunk.text.clone()),
        ),
        (
            "embedding".to_string(),
            AttributeValue::L(vector(&passage.embedding)),
        ),
    ])
}

fn metadata_item(model_id: &str, passages: usize) -> HashMap<String, AttributeValue> {
    HashMap::from([
        ("id".to_string(), AttributeValue::S(METADATA_ID.to_string())),
        (
            "model_id".to_string(),
            AttributeValue::S(model_id.to_string()),
        ),
        (
            "passages".to_string(),
            AttributeValue::N(passages.to_string()),
        ),
    ])
}

/// DynamoDB stores a vector as a list (`L`) of numbers (`N`).
fn vector(values: &[f32]) -> Vec<AttributeValue> {
    values
        .iter()
        .map(|v| AttributeValue::N(v.to_string()))
        .collect()
}

fn chunk_from_item(item: &HashMap<String, AttributeValue>) -> KbResult<Chunk> {
    Ok(Chunk {
        id: string_attr(item, "id")?,
        text: string_attr(item, "text")?,
        page: number_attr(item, "page")?,
    })
}

fn string_attr(item: &HashMap<String, AttributeValue>, name: &str) -> KbResult<String> {
    let value = item
        .get(name)
        .and_then(|v| v.as_s().ok())
        .ok_or_else(|| format!("item without a text attribute {name}"))?;
    Ok(value.clone())
}

fn number_attr(item: &HashMap<String, AttributeValue>, name: &str) -> KbResult<usize> {
    let value = item
        .get(name)
        .and_then(|v| v.as_n().ok())
        .ok_or_else(|| format!("item without a number attribute {name}"))?;
    Ok(value.parse()?)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn words(n: usize) -> String {
        (0..n)
            .map(|i| format!("w{i:03}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn short_pages_are_skipped_and_the_footer_is_stripped() {
        let pages = vec![
            "Too short.".to_string(),
            format!(
                "{FOOTER}\n\n  An opportunity attack happens when a hostile creature moves out of reach."
            ),
        ];
        let chunks = extract_chunks(&pages);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].page, 2);
        assert_eq!(chunks[0].id, "page_2_offset_0");
        assert!(chunks[0].text.starts_with("An opportunity attack"));
    }

    #[test]
    fn long_pages_overlap_and_cut_on_spaces() {
        // 600 four-char words + spaces = 2999 chars: three passages.
        let chunks = extract_chunks(&[words(600)]);
        assert!(chunks.len() >= 3);
        for c in &chunks {
            assert!(c.text.chars().count() <= CHUNK_SIZE);
            assert!(c.text.split(' ').all(|w| w.len() == 4), "cut inside a word");
        }
        // The last word of a passage reappears in the next one: the overlap.
        let last = chunks[0].text.rsplit(' ').next().unwrap();
        assert!(chunks[1].text.contains(last));
    }

    #[test]
    fn multi_byte_text_never_panics() {
        let page = "é".repeat(2500);
        assert!(!extract_chunks(&[page]).is_empty());
    }

    fn passage(id: &str, embedding: Vec<f32>) -> Passage {
        Passage {
            chunk: Chunk {
                id: id.to_string(),
                text: format!("{id} passage"),
                page: 7,
            },
            embedding,
        }
    }

    #[test]
    fn a_passage_survives_the_trip_through_an_item() {
        let p = passage("page_7_offset_0", vec![0.25, -0.5]);
        let item = passage_item(&p);
        assert_eq!(chunk_from_item(&item).unwrap(), p.chunk);
        assert_eq!(
            item["embedding"],
            AttributeValue::L(vec![
                AttributeValue::N("0.25".to_string()),
                AttributeValue::N("-0.5".to_string()),
            ])
        );
    }

    #[test]
    fn the_metadata_item_records_the_model_and_the_count() {
        let item = metadata_item(DEFAULT_EMBEDDING_MODEL_ID, 979);
        assert_eq!(
            string_attr(&item, "model_id").unwrap(),
            DEFAULT_EMBEDDING_MODEL_ID
        );
        assert_eq!(number_attr(&item, "passages").unwrap(), 979);
        assert!(
            !item.contains_key("embedding"),
            "it must stay out of the index"
        );
    }

    #[tokio::test]
    async fn an_unreachable_extenddb_names_the_start_command() {
        // Bind a free port, then let it go: nothing answers there.
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let err = KnowledgeBase::create(
            &format!("http://127.0.0.1:{port}"),
            DEFAULT_RULES_TABLE,
            DEFAULT_EMBEDDING_MODEL_ID,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains(START_EXTENDDB), "{err}");
    }

    /// A unit vector along axis `i`, at the index's dimensions.
    fn axis(i: usize) -> Vec<f32> {
        let mut v = vec![0.0; EMBEDDING_DIMENSIONS];
        v[i] = 1.0;
        v
    }

    /// The item the Rules Agent opens the table with, if the build wrote it.
    async fn metadata(kb: &KnowledgeBase) -> Option<HashMap<String, AttributeValue>> {
        kb.client
            .get_item()
            .table_name(&kb.table)
            .key("id", AttributeValue::S(METADATA_ID.to_string()))
            .consistent_read(true)
            .send()
            .await
            .unwrap()
            .item()
            .cloned()
    }

    #[tokio::test]
    #[ignore = "needs ExtendDB running: extenddb serve --foreground, then cargo test -- --ignored"]
    async fn extenddb_marks_the_build_finished_last_and_returns_the_closest_first() {
        let endpoint = extenddb_endpoint();
        let table = format!("test-{}", uuid::Uuid::new_v4());
        let mut kb = KnowledgeBase::create(&endpoint, &table, DEFAULT_EMBEDDING_MODEL_ID)
            .await
            .unwrap();

        // No metadata item yet: the Rules Agent must not start on this table.
        assert!(metadata(&kb).await.is_none());

        let mut middle = axis(0);
        middle[1] = 1.0;
        let norm = 2.0_f32.sqrt();
        middle.iter_mut().for_each(|x| *x /= norm);
        kb.store(&[
            passage("far", axis(1)),
            passage("near", axis(0)),
            passage("middle", middle),
        ])
        .await
        .unwrap();

        let item = metadata(&kb)
            .await
            .expect("store writes the metadata item last");
        assert_eq!(
            string_attr(&item, "model_id").unwrap(),
            DEFAULT_EMBEDDING_MODEL_ID
        );
        assert_eq!(number_attr(&item, "passages").unwrap(), 3);
        let hits = kb.search(&axis(0), 2).await.unwrap();
        let ids: Vec<&str> = hits.iter().map(|(_, c)| c.id.as_str()).collect();
        assert_eq!(ids, ["near", "middle"]);
        assert!((hits[0].0 - 1.0).abs() < 1e-6, "{hits:?}");
        assert!((hits[1].0 - f64::from(1.0 / norm)).abs() < 1e-6, "{hits:?}");
        assert_eq!(hits[0].1.text, "near passage");
        assert_eq!(hits[0].1.page, 7);

        // A rebuild drops the old table: the new one starts empty and unfinished.
        let rebuilt = KnowledgeBase::create(&endpoint, &table, DEFAULT_EMBEDDING_MODEL_ID)
            .await
            .unwrap();
        assert!(metadata(&rebuilt).await.is_none());

        rebuilt
            .client
            .delete_table()
            .table_name(&table)
            .send()
            .await
            .unwrap();
    }
}
