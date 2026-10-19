//! Chapter 5 - Build the Rules Agent knowledge base
//!
//! A side-by-side port of the Python workshop's
//! `5_a2a_integration/utils/create_knowledge_base.py`:
//!   <https://github.com/aws-samples/sample-once-upon-agentic-ai/tree/main/5_a2a_integration>
//!
//! Reads `DnD_BasicRules_2018.pdf` page by page, cuts each page into
//! overlapping ~1000-character passages, embeds every passage with Titan Text
//! Embeddings V2 on Bedrock, and writes `rules_knowledge_base.json` next to the
//! PDF. Run it once, with the PDF in `5_a2a_integration/utils/`:
//!   cargo run --release --bin ch5-create-knowledge-base
//!
//! Environment overrides: EMBEDDING_MODEL_ID, AWS_REGION, AWS_PROFILE.

use std::path::{Path, PathBuf};

use aws_sdk_bedrockruntime::Client;
use aws_sdk_bedrockruntime::error::DisplayErrorContext;
use aws_sdk_bedrockruntime::primitives::Blob;
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

    println!("Extracting text from the PDF...");
    let pages = pdf_extract::extract_text_by_pages(&pdf)?;
    let chunks = extract_chunks(&pages);
    println!(
        "{} passages extracted from {} pages",
        chunks.len(),
        pages.len()
    );

    let embedder = Embedder::from_env().await;
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

    let kb = KnowledgeBase {
        source: PDF_FILE.to_string(),
        model_id: embedder.model_id().to_string(),
        dimensions: EMBEDDING_DIMENSIONS,
        passages,
    };
    let path = kb_path();
    kb.save(&path)?;
    println!(
        "Knowledge base created at {} ({} passages)",
        path.display(),
        kb.passages.len()
    );
    Ok(())
}

#[tokio::main]
async fn main() -> KbResult<()> {
    create_knowledge_base().await?;
    println!("Knowledge base creation complete!");
    Ok(())
}

// The knowledge base on disk, the Rust side of the Python chapter's ChromaDB
// collection. This builder writes it and the Rules Agent reads it, so both files
// carry the same types and the same `Embedder`: keep the two copies identical.
type KbResult<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Verified live (us-west-2, 2026-10-19, `aws bedrock get-foundation-model`):
/// `ACTIVE`, output modality `EMBEDDING`. Override with `EMBEDDING_MODEL_ID`.
const DEFAULT_EMBEDDING_MODEL_ID: &str = "amazon.titan-embed-text-v2:0";
/// Titan V2 offers 256, 512 or 1024. 512 keeps the file small and search sharp.
const EMBEDDING_DIMENSIONS: usize = 512;

const PDF_FILE: &str = "DnD_BasicRules_2018.pdf";
/// Printed on almost every page; useless for search, so it is stripped.
const FOOTER: &str = "D&D Basic Rules (Version 1.0). Not for resale. Permission granted to print and photocopy this document for personal use only.";
const CHUNK_SIZE: usize = 1000; // characters per passage
const CHUNK_OVERLAP: usize = 200; // shared by two consecutive passages, so a rule cut in two stays findable
const MIN_PASSAGE: usize = 50;

fn embedding_model_id() -> String {
    std::env::var("EMBEDDING_MODEL_ID").unwrap_or_else(|_| DEFAULT_EMBEDDING_MODEL_ID.to_string())
}

fn utils_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("5_a2a_integration/utils")
}

fn pdf_path() -> PathBuf {
    utils_dir().join(PDF_FILE)
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

fn squash_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Cut each page into overlapping passages: the same algorithm as the Python
/// `extract_chunks`, counted in characters (not bytes) so the cuts land in the
/// same places and never split a multi-byte character.
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

impl KnowledgeBase {
    /// Write a temporary file, then rename it over the old one, so an
    /// interrupted build never leaves a half-written knowledge base.
    ///
    /// # Errors
    ///
    /// Fails when the file cannot be written.
    fn save(&self, path: &Path) -> KbResult<()> {
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(self)?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }
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
                text: id.to_string(),
                page: 1,
            },
            embedding,
        }
    }

    #[test]
    fn a_saved_knowledge_base_reads_back() {
        let path = std::env::temp_dir().join(format!("kb-{}.json", uuid::Uuid::new_v4()));
        let kb = KnowledgeBase {
            source: PDF_FILE.to_string(),
            model_id: DEFAULT_EMBEDDING_MODEL_ID.to_string(),
            dimensions: 1,
            passages: vec![passage("only", vec![1.0])],
        };
        kb.save(&path).unwrap();
        let read: KnowledgeBase = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(read, kb);
        std::fs::remove_file(&path).unwrap();
    }
}
