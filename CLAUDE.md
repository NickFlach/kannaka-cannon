# Claude Code Configuration -- Kannaka Cannon

## Behavioral Rules (Always Enforced)

- Do what has been asked; nothing more, nothing less
- NEVER create files unless they're absolutely necessary for achieving your goal
- ALWAYS prefer editing an existing file to creating a new one
- NEVER proactively create documentation files (*.md) or README files unless explicitly requested
- NEVER save working files, text/mds, or tests to the root folder
- Never continuously check status after spawning a swarm -- wait for results
- ALWAYS read a file before editing it
- NEVER commit secrets, credentials, or .env files

## Kannaka Constellation Membership

Kannaka Cannon is the **perception layer** of the Kannaka constellation:

- **Role**: Eyes and ears -- transforms raw video into structured, searchable intelligence
- **HRM Integration**: Video analysis results are stored as Holographic Resonance Memory entries
- **NATS Events**: Publishes `cannon.*` events to the Kannaka mesh (ingest, highlight, render, voice)
- **Recall**: Video segments become first-class memories accessible via `kannaka recall`

### Constellation Members

| Member | Role | Repo |
|--------|------|------|
| Memory | Core HRM | `kannaka-memory` |
| Radio | Ghost DJ | `kannaka-radio` |
| Observatory | Visualization | `kannaka-observatory` |
| **Cannon** | **Video Intelligence** | `kannaka-cannon` |

### Key Integration Points

- `kannaka remember` -- Store video analysis summaries as HRM memories
- `kannaka recall` -- Find video segments matching natural-language queries
- `kannaka swarm sync` -- Synchronize video metadata across the constellation
- NATS subjects: `cannon.ingest.*`, `cannon.highlight.*`, `cannon.render.*`, `cannon.voice.*`

## File Organization

- NEVER save to root folder -- use the directories below
- Use `/src` for source code files
- Use `/tests` for test files
- Use `/docs` for documentation and markdown files
- Use `/config` for configuration files
- Use `/scripts` for utility scripts
- Use `/examples` for example code

## Project Architecture

- Python package lives at `src/clipcannon/` (do NOT rename -- breaks imports)
- 22-stage analysis pipeline in `src/clipcannon/pipeline/`
- 51 MCP tool definitions in `src/clipcannon/tools/`
- All tool names prefixed with `clipcannon_` (MCP protocol identifiers)
- Per-project SQLite databases with sqlite-vec vector tables
- Config stored at `~/.clipcannon/config.json`

## Build & Test

```bash
# Install
pip install -e ".[dev]"

# Test (626 tests)
pytest

# Lint
ruff check src/
```

- ALWAYS run tests after making code changes
- ALWAYS verify lint is clean before committing

## Security Rules

- NEVER hardcode API keys, secrets, or credentials in source files
- NEVER commit .env files or any file containing secrets
- Always validate user input at system boundaries
- Always sanitize file paths to prevent directory traversal

## Concurrency

- Batch all independent tool calls in a single message
- Use `run_in_background: true` for long-running agent tasks
- After spawning background agents, STOP -- don't poll or check status
