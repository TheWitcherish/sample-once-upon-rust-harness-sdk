# Once Upon Rust Harness SDK - Chapter 6: Web Interface

This is a **sample** for workshops, livestreams and experiments, not production code. This `ui`
branch holds only the browser UI: a Vue 3 + TypeScript app where you play the Chapter 5 Game Master.
The Rust agents that answer it live on the `main` branch.

## Play it

1. Start the four Chapter 5 servers from `main` (each in its own terminal). The README there covers
   the one-off knowledge-base build:

   ```sh
   cargo run --bin ch4-dice-mcp-server
   cargo run --bin ch5-rules-agent
   cargo run --bin ch5-character-agent
   cargo run --bin ch5-gamemaster-orchestrator
   ```

2. Start the UI on this branch (Node.js 22):

   ```sh
   npm ci
   npm run dev
   ```

3. Open [http://localhost:5173](http://localhost:5173), create a character, and set the
   **Server URL** to `http://localhost:8009`.

The Game Master only accepts browser calls from `http://localhost:<port>` origins, so open the UI on
`localhost`.

## Let your coding agent play

In a browser with WebMCP, the W3C draft that lets a page hand tools to an AI agent (only Chrome 149+
and Edge 150+, as an experimental trial), the game screen registers two tools: `take_action` plays a
turn through the same path as the input box, and `read_character_sheet` reads your sheet. Elsewhere
the page works as before. Chapter 0's "Prepare a Browser for WebMCP" step and Chapter 6, Part 5 of the
workshop connect a coding agent to them through Chrome DevTools MCP.

## Checks

```sh
npm test        # 126 tests (Vitest)
npm run build   # type-check with vue-tsc, then bundle with Vite
```
