//! Chapter 4 - Planar Portals: the Dice Roller as an MCP server
//!
//! Chapter 3's dice roller lived inside the agent's process. Here it moves out,
//! behind MCP (Model Context Protocol - the standard way agents discover and call
//! tools that live somewhere else). Any MCP client, in any language, can now roll.
//!
//! It uses `rmcp`, the official Rust MCP SDK: the `#[tool]` macro turns a method
//! into an MCP tool. We mount MCP at `/mcp` on an axum router and serve it
//! ourselves.
//!
//! Run with:
//!   cargo run --bin ch4-dice-mcp-server
//!
//! Environment overrides:
//!   MCP_BIND           address to listen on (default 127.0.0.1:8002, loopback only)
//!   MCP_ALLOWED_HOSTS  comma-separated Host headers to accept when serving
//!                      beyond loopback, e.g. "dice.example.com"
//!
//! There is no authentication here. Keep it on
//! loopback, or put it behind a proxy that authenticates before exposing it.

use std::env;

use rmcp::handler::server::wrapper::Parameters;
use rmcp::schemars::JsonSchema;
// TODO: Step 1 - Import StreamableHttpService and LocalSessionManager from rmcp
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
// TODO: Step 2 - Import the tool and tool_router attributes from rmcp
use rmcp::{tool, tool_router};
use serde::Deserialize;

const DEFAULT_BIND: &str = "127.0.0.1:8002";

/// What the model sends when it calls `roll_dice`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[schemars(crate = "rmcp::schemars")]
struct RollDiceParams {
    /// Number of faces on the die, 1 or more. Use 20 for a d20, 6 for a d6,
    /// 100 for a percentile die. Defaults to 6.
    #[serde(default = "six")]
    faces: i64,
}

fn six() -> i64 {
    6
}

/// Roll a single die with the given number of faces. Same function as Chapter 3:
/// it knows nothing about MCP, so it can be tested on its own.
fn roll_dice(faces: i64) -> Result<i64, String> {
    if faces < 1 {
        return Err("Dice must have at least 1 face".to_string());
    }
    Ok(rand::random_range(1..=faces))
}

#[derive(Debug, Clone)]
struct DiceServer;

// TODO: Step 2 - Add the attributes to expose roll_dice as an MCP tool
#[tool_router(server_handler)]
impl DiceServer {
    /// Roll one die with a given number of faces and return the result.
    ///
    /// Use it every time the game needs a random roll: ability scores, attack rolls,
    /// saving throws, damage. Call it once per die; for "4d6 drop lowest", call it
    /// four times with faces=6 and discard the lowest result yourself.
    ///
    /// Example response: 14
    ///
    /// Notes:
    ///     - One die per call: there is no count parameter.
    ///     - Each roll is independent and uniformly random.
    #[tool]
    fn roll_dice(
        &self,
        Parameters(RollDiceParams { faces }): Parameters<RollDiceParams>,
    ) -> Result<String, String> {
        let roll = roll_dice(faces)?;
        tracing::info!("DICE ROLL: d{faces} = {roll}");
        Ok(roll.to_string())
    }
}

/// Loopback-only by default (rmcp rejects any other Host header with 403, the
/// fix for RUSTSEC-2026-0189). Extra hosts are opt-in, never a wildcard.
fn server_config(allowed_hosts: Option<&str>) -> StreamableHttpServerConfig {
    let config = StreamableHttpServerConfig::default();
    match allowed_hosts {
        None => config,
        Some(list) => {
            let mut hosts = config.allowed_hosts.clone();
            hosts.extend(
                list.split(',')
                    .map(str::trim)
                    .filter(|h| !h.is_empty())
                    .map(String::from),
            );
            config.with_allowed_hosts(hosts)
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let bind = env::var("MCP_BIND").unwrap_or_else(|_| DEFAULT_BIND.to_string());
    let config = server_config(env::var("MCP_ALLOWED_HOSTS").ok().as_deref());

    // TODO: Step 1 - Create an MCP server mounted at /mcp (it listens on port 8002)
    let mcp: StreamableHttpService<DiceServer, LocalSessionManager> =
        StreamableHttpService::new(|| Ok(DiceServer), Default::default(), config);
    let router = axum::Router::new().nest_service("/mcp", mcp);
    let listener = tokio::net::TcpListener::bind(&bind).await?;

    println!("Starting TTRPG Dice Roll MCP Server...");
    tracing::info!("listening on http://{}/mcp", listener.local_addr()?);

    // TODO: Step 3 - Run the MCP server with axum until Ctrl-C
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await?;
    Ok(())
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
    }

    // The raw JSON shapes a model sends, through the same deserializer rmcp uses.
    #[test]
    fn missing_faces_means_a_d6() -> Result<(), serde_json::Error> {
        let params: RollDiceParams = serde_json::from_value(serde_json::json!({}))?;
        assert_eq!(params.faces, 6);
        Ok(())
    }

    #[test]
    fn non_integer_faces_is_rejected() {
        let parsed =
            serde_json::from_value::<RollDiceParams>(serde_json::json!({"faces": "twenty"}));
        assert!(parsed.is_err());
    }

    #[test]
    fn only_loopback_unless_hosts_are_named() {
        let default = server_config(None).allowed_hosts;
        assert!(
            default
                .iter()
                .all(|h| ["localhost", "127.0.0.1", "::1"].contains(&h.as_str()))
        );

        let opened = server_config(Some("dice.example.com, ")).allowed_hosts;
        assert!(opened.contains(&"dice.example.com".to_string()));
        assert!(!opened.iter().any(String::is_empty));
    }
}
