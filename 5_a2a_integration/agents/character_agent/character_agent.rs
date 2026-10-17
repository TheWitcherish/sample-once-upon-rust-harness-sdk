//! Chapter 5 - The Character Agent (A2A server on port 8001)
//!
//! A side-by-side port of the Python workshop's
//! `5_a2a_integration/agents/character_agent/character_agent.py`:
//!   <https://github.com/aws-samples/sample-once-upon-agentic-ai/tree/main/5_a2a_integration>
//!
//! Three tools over a JSON file, served over A2A so the Game Master can ask
//! this agent for character sheets instead of keeping them itself.
//!
//! Run with:
//!   cargo run --bin ch5-character-agent
//!
//! Environment overrides: MODEL_ID, AWS_REGION, RUST_LOG, plus
//! CHARACTER_AGENT_BIND (default 127.0.0.1:8001) and CHARACTERS_DB (default:
//! the characters.json next to this file).

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use strands_agents::a2a::server::rpc_dispatch;
use strands_agents::a2a::{AgentSkill, StrandsA2AExecutor};
use strands_agents::models::{BedrockModel, ModelConfig};
use strands_agents::tools::ToolResult;
use strands_agents::{AgentBuilder, Tool, ToolContext, ToolSpec};
use tokio::sync::Mutex;

const NAME: &str = "Character Agent";
const DESCRIPTION: &str = "TTRPG character management: creates characters (ability scores \
rolled 4d6 drop lowest), stores them, finds and lists them.";
const SYSTEM_PROMPT: &str = "You are a TTRPG character manager. Use your tools to create, find or list characters.
When creating a character, roll each ability score with 4d6 drop lowest. If details are missing (gender, some scores), choose or roll them yourself instead of asking back.
Confirm creations and summarize found characters briefly: class, race, key stats.";

const DEFAULT_PORT: u16 = 8001;

fn ten() -> i64 {
    10
}

/// Any score the model leaves out defaults to 10, like the Python
/// `stats_dict.get('strength', 10)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stats {
    #[serde(default = "ten")]
    pub strength: i64,
    #[serde(default = "ten")]
    pub dexterity: i64,
    #[serde(default = "ten")]
    pub constitution: i64,
    #[serde(default = "ten")]
    pub intelligence: i64,
    #[serde(default = "ten")]
    pub wisdom: i64,
    #[serde(default = "ten")]
    pub charisma: i64,
}

impl Default for Stats {
    fn default() -> Self {
        Self {
            strength: 10,
            dexterity: 10,
            constitution: 10,
            intelligence: 10,
            wisdom: 10,
            charisma: 10,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InventoryItem {
    pub item_name: String,
    pub quantity: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Character {
    pub character_id: String,
    pub name: String,
    pub character_class: String,
    pub race: String,
    pub gender: String,
    pub level: i64,
    pub experience: i64,
    pub stats: Stats,
    pub inventory: Vec<InventoryItem>,
    pub created_at: Option<String>,
}

/// What the model sends to `create_character`.
#[derive(Debug, Deserialize)]
struct CreateParams {
    name: String,
    character_class: String,
    race: String,
    gender: String,
    #[serde(default)]
    stats_dict: Stats,
}

/// The same file layout as TinyDB (`{"_default": {"1": {...}}}`), so the
/// Python chapter's `characters.json` loads as-is and both versions can share it.
#[derive(Debug, Default, Serialize, Deserialize)]
struct DbFile {
    #[serde(rename = "_default", default)]
    table: Map<String, Value>,
}

/// Characters in insertion order (TinyDB's numeric document ids). Sorting the
/// keys as text would put "10" before "2".
fn rows(db: &DbFile) -> Result<Vec<(u64, Character)>, String> {
    let mut rows = Vec::new();
    for (key, value) in &db.table {
        let id = key
            .parse::<u64>()
            .map_err(|_| format!("bad document id {key:?}"))?;
        let character = serde_json::from_value(value.clone())
            .map_err(|e| format!("bad character record {key}: {e}"))?;
        rows.push((id, character));
    }
    rows.sort_by_key(|(id, _)| *id);
    Ok(rows)
}

/// The character store. Every read and write goes through one `Mutex`, so two
/// tool calls in the same turn cannot both read the file and then overwrite
/// each other's new character.
#[derive(Debug)]
pub struct CharacterDb {
    path: PathBuf,
    lock: Mutex<()>,
}

impl CharacterDb {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            lock: Mutex::new(()),
        }
    }

    async fn load(&self) -> Result<DbFile, String> {
        match tokio::fs::read(&self.path).await {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|e| format!("{} is not a valid database: {e}", self.path.display())),
            // A missing file is an empty database, as with TinyDB.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(DbFile::default()),
            Err(e) => Err(format!("cannot read {}: {e}", self.path.display())),
        }
    }

    pub async fn all(&self) -> Result<Vec<Character>, String> {
        let _guard = self.lock.lock().await;
        Ok(rows(&self.load().await?)?
            .into_iter()
            .map(|(_, c)| c)
            .collect())
    }

    /// Exact, case-sensitive match, like the Python `Query().name == name`.
    pub async fn find_by_name(&self, name: &str) -> Result<Option<Character>, String> {
        Ok(self.all().await?.into_iter().find(|c| c.name == name))
    }

    pub async fn insert(&self, character: &Character) -> Result<(), String> {
        let _guard = self.lock.lock().await;
        let mut db = self.load().await?;
        let next_id = rows(&db)?.last().map_or(1, |(id, _)| id + 1);
        let value = serde_json::to_value(character).map_err(|e| e.to_string())?;
        db.table.insert(next_id.to_string(), value);

        // Write a temporary file, then rename it over the old one, so a crash
        // halfway through never leaves a half-written database behind.
        let text = serde_json::to_string_pretty(&db).map_err(|e| e.to_string())?;
        let tmp = self.path.with_extension("json.tmp");
        tokio::fs::write(&tmp, text)
            .await
            .map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
        tokio::fs::rename(&tmp, &self.path)
            .await
            .map_err(|e| format!("cannot replace {}: {e}", self.path.display()))
    }
}

fn to_json(value: &impl Serialize) -> ToolResult {
    match serde_json::to_string_pretty(value) {
        Ok(text) => ToolResult::success(text),
        Err(e) => ToolResult::error(e.to_string()),
    }
}

struct FindCharacterByName(Arc<CharacterDb>);

#[async_trait]
impl Tool for FindCharacterByName {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "find_character_by_name",
            "Find a stored TTRPG character by its exact name.\n\n\
             Use it when a player refers to an existing character and you need its sheet: \
             class, race, level, ability scores, inventory. Names are matched exactly and \
             are case-sensitive; use list_all_characters if you are unsure of the spelling.\n\n\
             Fails with an error if no character with that name exists.",
            json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The character's name exactly as it was created, e.g. \"Thorin\"."
                    }
                },
                "required": ["name"]
            }),
        )
    }

    async fn execute(
        &self,
        input: Value,
        _ctx: &mut ToolContext,
    ) -> strands_agents::Result<ToolResult> {
        let Some(name) = input.get("name").and_then(Value::as_str) else {
            return Ok(ToolResult::error("name must be a string"));
        };
        tracing::info!(name, "find_character_by_name");
        Ok(match self.0.find_by_name(name).await {
            Ok(Some(character)) => to_json(&character),
            Ok(None) => ToolResult::error(format!("Character with name {name:?} not found")),
            Err(e) => ToolResult::error(e),
        })
    }
}

struct ListAllCharacters(Arc<CharacterDb>);

#[async_trait]
impl Tool for ListAllCharacters {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "list_all_characters",
            "List every character stored in the database.\n\n\
             Use it to see which characters exist before creating or looking one up, or \
             when a player asks for the whole party. Takes no parameters and returns every \
             character in full; prefer find_character_by_name when you already know the name. \
             Returns an empty list if no character has been created yet.",
            json!({ "type": "object", "properties": {} }),
        )
    }

    async fn execute(
        &self,
        _input: Value,
        _ctx: &mut ToolContext,
    ) -> strands_agents::Result<ToolResult> {
        Ok(match self.0.all().await {
            Ok(characters) => {
                tracing::info!(count = characters.len(), "list_all_characters");
                to_json(&characters)
            }
            Err(e) => ToolResult::error(e),
        })
    }
}

struct CreateCharacter(Arc<CharacterDb>);

impl CreateCharacter {
    fn build(params: CreateParams) -> Character {
        Character {
            character_id: uuid::Uuid::new_v4().to_string(),
            name: params.name,
            character_class: params.character_class,
            race: params.race,
            gender: params.gender,
            level: 1,
            experience: 0,
            stats: params.stats_dict,
            inventory: vec![
                InventoryItem {
                    item_name: "Starting Equipment Pack".to_string(),
                    quantity: 1,
                },
                InventoryItem {
                    item_name: "Gold Pieces".to_string(),
                    quantity: 100,
                },
            ],
            created_at: Some(
                chrono::Local::now()
                    .naive_local()
                    .format("%Y-%m-%dT%H:%M:%S%.6f")
                    .to_string(),
            ),
        }
    }
}

#[async_trait]
impl Tool for CreateCharacter {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "create_character",
            "Create a new TTRPG character and save it to the database.\n\n\
             Use it once per new character, after the ability scores have been decided \
             (roll them with the 4d6-drop-lowest method first). Every new character starts \
             at level 1 with 0 experience, a Starting Equipment Pack and 100 gold pieces.\n\n\
             Names are not checked for uniqueness: creating \"Thorin\" twice stores two \
             characters. Check with find_character_by_name if in doubt. \
             Any ability score missing from stats_dict defaults to 10.",
            json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "The character's name, e.g. \"Thorin\"." },
                    "character_class": { "type": "string", "description": "A class such as \"Fighter\", \"Wizard\" or \"Rogue\"." },
                    "race": { "type": "string", "description": "A race such as \"Dwarf\", \"Elf\" or \"Human\"." },
                    "gender": { "type": "string", "description": "The character's gender, e.g. \"Female\"; pick one if the player did not say." },
                    "stats_dict": {
                        "type": "object",
                        "description": "Ability scores, each an integer (typically 3 to 18), e.g. {\"strength\": 16, \"dexterity\": 12}.",
                        "properties": {
                            "strength": { "type": "integer" },
                            "dexterity": { "type": "integer" },
                            "constitution": { "type": "integer" },
                            "intelligence": { "type": "integer" },
                            "wisdom": { "type": "integer" },
                            "charisma": { "type": "integer" }
                        }
                    }
                },
                "required": ["name", "character_class", "race", "gender", "stats_dict"]
            }),
        )
    }

    async fn execute(
        &self,
        input: Value,
        _ctx: &mut ToolContext,
    ) -> strands_agents::Result<ToolResult> {
        let params: CreateParams = match serde_json::from_value(input) {
            Ok(params) => params,
            Err(e) => return Ok(ToolResult::error(format!("invalid character: {e}"))),
        };
        let character = Self::build(params);
        Ok(match self.0.insert(&character).await {
            Ok(()) => {
                tracing::info!(
                    name = %character.name,
                    id = %character.character_id,
                    "created character"
                );
                to_json(&character)
            }
            Err(e) => ToolResult::error(e),
        })
    }
}

fn skill(id: &str, description: &str) -> AgentSkill {
    AgentSkill {
        id: id.to_string(),
        name: id.to_string(),
        description: description.to_string(),
        tags: Vec::new(),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ch5_character_agent=info".into()),
        )
        .init();

    let db_path = std::env::var("CHARACTERS_DB").map_or_else(
        |_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("5_a2a_integration/agents/character_agent/characters.json")
        },
        PathBuf::from,
    );
    tracing::info!(db = %db_path.display(), "character database");
    let db = Arc::new(CharacterDb::new(db_path));

    let agent = AgentBuilder::new()
        .model(Box::new(game_master_model().await?))
        // TODO: Step 1 - Add the create_character, find_character_by_name and list_all_characters tools to the agent
        .tool(Arc::new(CreateCharacter(Arc::clone(&db))))
        .tool(Arc::new(FindCharacterByName(Arc::clone(&db))))
        .tool(Arc::new(ListAllCharacters(db)))
        // TODO: Step 2 - Add the name NAME ("Character Agent") to the agent
        .name(NAME)
        // TODO: Step 3 - Add the description DESCRIPTION ("TTRPG character management: creates characters (ability scores rolled 4d6 drop lowest), stores them, finds and lists them.") to the agent
        .description(DESCRIPTION)
        .system_prompt(SYSTEM_PROMPT)
        .build()
        .await?;

    // TODO: Step 4 - Create a StrandsA2AExecutor for the agent, with its three tools as skills
    // Python's A2AServer lists the tools as skills on its own; here we say so.
    let executor = StrandsA2AExecutor::for_agent(agent, NAME, DESCRIPTION)
        .with_version(env!("CARGO_PKG_VERSION"))
        .with_skill(skill(
            "create_character",
            "Create a new character and save it.",
        ))
        .with_skill(skill(
            "find_character_by_name",
            "Find a stored character by its exact name.",
        ))
        .with_skill(skill("list_all_characters", "List every stored character."));

    // TODO: Step 5 - Start the A2A server on port 8001 with serve_a2a
    serve_a2a(executor, bind_addr("CHARACTER_AGENT_BIND", DEFAULT_PORT)?).await?;
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
    use strands_agents::tools::ToolStatus;

    const SEED: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/5_a2a_integration/agents/character_agent/characters.json"
    );

    /// A database in a fresh temp file, optionally seeded with the Python party.
    async fn temp_db(seed: bool) -> (Arc<CharacterDb>, PathBuf) {
        let path = std::env::temp_dir().join(format!("characters-{}.json", uuid::Uuid::new_v4()));
        if seed {
            tokio::fs::copy(SEED, &path).await.expect("copy seed");
        }
        (Arc::new(CharacterDb::new(&path)), path)
    }

    async fn run(tool: &dyn Tool, input: Value) -> ToolResult {
        tool.execute(input, &mut ToolContext::default())
            .await
            .expect("execute never returns Err")
    }

    #[tokio::test]
    async fn the_python_party_loads_in_order() {
        let db = CharacterDb::new(SEED);
        let names: Vec<String> = db
            .all()
            .await
            .unwrap()
            .into_iter()
            .map(|c| c.name)
            .collect();
        assert_eq!(names, ["Arnaud", "Thorin Ironbeard", "Gypsy", "Kzertus"]);
    }

    #[tokio::test]
    async fn find_is_exact_and_case_sensitive() {
        let db = CharacterDb::new(SEED);
        assert_eq!(
            db.find_by_name("Arnaud")
                .await
                .unwrap()
                .unwrap()
                .stats
                .wisdom,
            15
        );
        assert!(db.find_by_name("arnaud").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn an_unknown_name_is_a_tool_error() {
        let tool = FindCharacterByName(Arc::new(CharacterDb::new(SEED)));
        let result = run(&tool, json!({ "name": "Nobody" })).await;
        assert!(matches!(result.status, ToolStatus::Error));
    }

    #[tokio::test]
    async fn create_from_raw_model_json_defaults_missing_scores_to_ten() {
        let (db, path) = temp_db(true).await;
        let tool = CreateCharacter(Arc::clone(&db));
        let result = run(
            &tool,
            json!({
                "name": "Alice", "character_class": "Wizard", "race": "Elf",
                "gender": "Female", "stats_dict": { "intelligence": 17 }
            }),
        )
        .await;
        assert!(matches!(result.status, ToolStatus::Success));

        let alice = db.find_by_name("Alice").await.unwrap().unwrap();
        assert_eq!(alice.stats.intelligence, 17);
        assert_eq!(alice.stats.strength, 10);
        assert_eq!(alice.level, 1);

        // Appended after the 4 seed characters, under TinyDB id "5".
        let raw: DbFile = serde_json::from_slice(&tokio::fs::read(&path).await.unwrap()).unwrap();
        assert!(raw.table.contains_key("5"));
        tokio::fs::remove_file(&path).await.unwrap();
    }

    #[tokio::test]
    async fn create_without_a_name_is_a_tool_error() {
        let (db, _path) = temp_db(false).await;
        let result = run(&CreateCharacter(db), json!({ "race": "Elf" })).await;
        assert!(matches!(result.status, ToolStatus::Error));
    }

    #[tokio::test]
    async fn a_missing_file_is_an_empty_database() {
        let (db, _path) = temp_db(false).await;
        assert!(db.all().await.unwrap().is_empty());
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
