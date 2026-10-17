# Once Upon Agentic AI - the Rust port

> This is a **sample** repository for experiments, livestreams, and workshops - not production
> code. It is a Rust port of the Python workshop
> [aws-samples/sample-once-upon-agentic-ai](https://github.com/aws-samples/sample-once-upon-agentic-ai),
> built on the [Strands Agents SDK for Rust](https://strandsagents.com/). Clone it, break it, remix it.

*"Roll for Initiative... in Rust!"*

Ever had an agent forget you the moment the demo moved on? We are going to build one that does the
opposite - a Game Master for a Table-Top Role-Playing Game (TTRPG) that remembers the party, reaches for real rules, and
learns new tricks one chapter at a time. One agent first, then a tool it did not have to write, then
a tool it *did*, then the same tool over a wire, then a whole party of agents talking to each other.

Same philosophy and patterns as the Python workshop; the typed boundaries and the function names line
up so you can read the two side by side. Where Rust asks something different of you - naming the model
provider explicitly, an `AgentResult` instead of a print - the code says so in a comment, right where
it happens. Like the Python workshop, each chapter folder holds all of the code it runs: the helpers a
file needs, such as the model setup, sit at the bottom of that file.

## Quick start (30-Second Kickoff)

```sh
# 1. point your shell at an AWS account with Bedrock access
export AWS_PROFILE=YOUR_PROFILE
export AWS_REGION=YOUR_AWS_REGION          # e.g. us-east-1

# 2. summon your first game master
cargo run --bin ch1-summoning
```

That is the whole loop: credentials in, story out. If you see the Game Master answer the adventurer,
Chapter 1 is done. On `main` each chapter is a skeleton: fill in its `// TODO` steps first, or run
it from the `solution` branch to see the finished version. While a skeleton is unfinished, `cargo`
warns about unused imports and items: each warning points at a step you have not wired in yet.

The agents run on **Amazon Bedrock** (Claude Sonnet 4.6 by default). Pick a cheaper model any time:

```sh
export MODEL_ID=YOUR_MODEL   # e.g. us.anthropic.claude-haiku-4-5-20251001-v1:0
```

Model ids are read from `MODEL_ID` with a verified fallback; the region comes from the standard AWS
environment. Nothing is hard-coded alone.

## The adventure map

| Chapter | Folder | What you build | Status |
|---|---|---|---|
| 1. The Art of Agent Summoning | `1_strands_basics/` | Your first agent, a system prompt, debug logs | **ready** |
| 2. The Adventurer's Arsenal - Vended Tools | `2_vended_tools/` | A tool you did not write: `http_fetch` on the live Open5e API | **ready** |
| 2. Bonus Quest - Kiro the Grey Hat | `2_vended_tools/bonus_quest.rs` | `shell` + `editor`, with your y/N approval before every call (`cargo run --bin ch2-bonus-quest`) | **ready** |
| 3. The Art of Magical Forging | `3_custom_tools/` | Your own tool: the dice roller | **ready** |
| 4. Planar Portals - MCP | `4_mcp_integration/` | The dice roller served over MCP with `rmcp`, and Lady Luck consuming it | **ready** |
| 5. The Grand Alliance - A2A | `5_a2a_integration/` | Rules Agent (RAG over the Basic Rules) + Character Agent over A2A, a Game Master orchestrator with typed `StoryOutput` answers | **ready** |
| 6. Web Interface | `ui` branch | Play your Game Master in the browser (Vue 3 + TypeScript): `git switch ui`, `npm ci`, `npm run dev` | **ready** |
| 7. The Enchanted Armour - Harness | `7_strands_harness/` | The Game Master rebuilt with the managed agent loop | planned |

Complete the chapters in order: each reuses what the previous one built. Every chapter is one binary -
`cargo run --bin ch<N>-<name>`.

Rules lookups use the SRD 5.1 (System Reference Document, CC-BY-4.0) served by the
[Open5e API](https://open5e.com).

Chapter 4 runs in two terminals: `cargo run --bin ch4-dice-mcp-server` first, then
`cargo run --bin ch4-gamemaster-mcp-client`, then type a request such as `roll 3d20`. The server listens on `127.0.0.1:8002` and only accepts
loopback `Host` headers. To serve it beyond your machine, set `MCP_BIND` and name the extra hosts in
`MCP_ALLOWED_HOSTS`. It has no authentication, like the Python chapter, so put an authenticating
proxy in front before exposing it.

Chapter 5 needs the Basic Rules PDF, which is licensed for personal use only and so is not in this
repo. Put your copy at `5_a2a_integration/utils/DnD_BasicRules_2018.pdf`, then build the knowledge
base once (about 35 seconds of Titan embedding calls):

```sh
cargo run --release --bin ch5-create-knowledge-base
```

Then start four servers, each in its own terminal, and ask the Game Master:

```sh
cargo run --bin ch4-dice-mcp-server          # dice over MCP, port 8002
cargo run --bin ch5-rules-agent              # A2A, port 8000
cargo run --bin ch5-character-agent          # A2A, port 8001
cargo run --bin ch5-gamemaster-orchestrator  # HTTP, port 8009

curl -s localhost:8009/inquire -H 'content-type: application/json' \
  -d '{"question":"What are the rules for a dexterity check roll?"}'
```

The Game Master also serves `GET /health`, `GET /messages` and `GET /user/{name}`. Every server listens
on loopback only and has no authentication, like the Python chapter. Its answers use forced structured
output like Python's `structured_output_model`: if the model answers in plain text, the agent asks once
more with only the `StoryOutput` tool offered and the model forced to call it. Browsers may call it from
`http://localhost:<port>` origins only (Python allows every origin). One difference from Python: each
A2A agent keeps one conversation shared by all callers.

## Where the SDK comes from

The Rust Strands SDK is not yet on crates.io in the shape this workshop needs (`strands-agents` has an
early `0.1.0` there; `strands-agents-tools`, which Chapter 2 onward depends on, is not published at
all). So `Cargo.toml` points at a **local checkout** of the SDK via a relative path:

```
sample-once-upon-rust-harness-sdk/  <- you are here
../../aws/strands-agents-rust/      <- the SDK checkout the paths expect
```

If your SDK lives elsewhere, edit the two `path = ...` lines in `Cargo.toml`. A true fresh-clone-and-run
story arrives when both crates are published; until then this is the honest setup.

## Branches

- `main`: the skeleton you clone, with `// TODO` markers to fill in, at the same steps as the Python workshop.
- `solution`: the same files with the answers written below each TODO. Switch to it if you get stuck.
- `ui`: the Chapter 6 web interface.
- Live-built work (a stream, a video) lands on its own `feat/chapter-N-*` branch, then a PR into `main`.

## Credits

Ported from [aws-samples/sample-once-upon-agentic-ai](https://github.com/aws-samples/sample-once-upon-agentic-ai)
(MIT-0). Built with the [Strands Agents SDK for Rust](https://strandsagents.com/).

*Happy forging, Agent Master.*
