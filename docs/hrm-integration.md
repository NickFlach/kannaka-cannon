# HRM Integration Architecture

How Kannaka Cannon bridges video perception with the Holographic Resonance Medium.

## Overview

Kannaka Cannon's 22-stage analysis pipeline produces structured intelligence from raw video: transcripts, emotion contours, visual embeddings, speaker identities, narrative structure, and highlight scores. The HRM integration layer stores these results as wave-interference memories, making video intelligence a first-class citizen in the Kannaka constellation.

## Video Analysis Memories

> The concrete, shipped behavior is in **[What's wired today](#whats-wired-today)**
> below. This section describes the fuller design; only transcript, scene, and
> music stems are wired so far — highlights, speakers, narrative, emotion, and
> OCR are candidates for later.

### What Gets Stored

After a `clipcannon_ingest` pipeline run finalizes, the following are candidates
for HRM storage (✅ = wired today):

| Analysis Output | Memory Type | Wired | Example Query |
|-----------------|-------------|-------|---------------|
| Transcript (summary) | Semantic | ✅ | "find where they discuss pricing" |
| Scene descriptions | Visual | ✅ | "outdoor scenes with mountains" |
| Music / beat features | Audio | ✅ | "upbeat tracks around 120 BPM" |
| Highlight moments | Emotional | planned | "most exciting moment in any video" |
| Speaker profiles | Identity | planned | "everything Sarah said across all projects" |
| Narrative summaries | Structural | planned | "videos with a strong call-to-action ending" |
| Emotion contours | Affective | planned | "moments where the audience laughed" |
| OCR text | Factual | planned | "slides mentioning quarterly revenue" |

### Storage Format

A memory is a single `kannaka remember` call: a plain-text body plus flags. The
HRM derives phase/amplitude/frequency from the text; Cannon does not send raw
embeddings or paths. For example:

```bash
kannaka remember \
  "[cannon:proj_abc123] Q4 All-Hands transcript summary (14:32, 2140 words): \
   And that brings us to the announcement we've all been waiting for…" \
  --importance 0.68 --category cannon --modality semantic \
  --tags cannon,transcript,proj_abc123
```

### How it works

```bash
# On finalize, Cannon composes one concise summary per stem type and stores
# each via a short-lived `kannaka remember` (see the table below for importance).

# Later, from any constellation member — or the clipcannon_hrm_recall MCP tool:
kannaka recall "AI healthcare discussion" --top-k 5
```

## Observatory Integration

The Kannaka Observatory could visualize Cannon's activity and outputs:

### Live Pipeline View

- Active ingestion pipelines displayed as animated DAG nodes
- Stage completion shown in real-time (probe -> frames -> transcribe -> ...)
- Failed stages highlighted with error summaries
- Estimated time remaining based on video length and GPU capacity

### Embedding Space Explorer

- Interactive 2D/3D projection of embedding spaces (t-SNE or UMAP)
- Visual embedding clusters showing scene similarity
- Semantic clusters showing topic groupings
- Cross-project embedding comparison ("how similar are these two videos?")

### Constellation Map

- Cannon appears as a node in the constellation visualization
- Active connections to Memory (storing results) and NATS (broadcasting events)
- Throughput metrics: videos ingested, clips rendered, voices cloned
- Health status: GPU utilization, queue depth, error rate

## Recall Integration

`kannaka recall` becomes the universal interface for finding video segments:

### Query Examples

```bash
# Natural language search across all video projects
kannaka recall "the part where they announce the new product" --top-k 5

# Emotion-filtered search
kannaka recall "excited reactions" --filter emotion:high_energy --top-k 10

# Speaker-specific search
kannaka recall "what did the CEO say about revenue" --top-k 3

# Cross-modal search (finds visual scenes matching a text query)
kannaka recall "whiteboard diagram of the architecture" --top-k 5
```

### Return Format

```json
{
  "results": [
    {
      "project_id": "proj_abc123",
      "video_title": "Q4 All-Hands Meeting",
      "timestamp_start": 145.2,
      "timestamp_end": 162.8,
      "transcript": "And that brings us to the announcement we've all been waiting for...",
      "highlight_score": 0.89,
      "emotion": "excited",
      "speaker": "CEO - Sarah Chen",
      "similarity": 0.94
    }
  ]
}
```

## NATS Event Broadcasting

Cannon publishes events to the Kannaka NATS mesh for real-time constellation coordination:

### Event Subjects

| Subject | Payload | Subscribers |
|---------|---------|-------------|
| `cannon.ingest.started` | `{project_id, video_path, estimated_duration}` | Observatory (live view) |
| `cannon.ingest.stage_complete` | `{project_id, stage_name, duration_ms, row_count}` | Observatory (DAG progress) |
| `cannon.ingest.completed` | `{project_id, stages_completed, total_duration}` | Memory (store summaries), Observatory |
| `cannon.highlight.found` | `{project_id, timestamp, score, reason}` | Memory (store highlight), Radio (play clip?) |
| `cannon.render.completed` | `{project_id, edit_id, output_path, profile, duration}` | Observatory (output gallery) |
| `cannon.voice.cloned` | `{project_id, profile_name, secs_score, sample_count}` | Memory (store voice identity) |
| `cannon.error` | `{project_id, stage, error_type, message}` | Observatory (alerts) |

### Event Flow Example

```
1. User: "Analyze this video"
2. Cannon -> NATS: cannon.ingest.started {project_id: "proj_123", video: "keynote.mp4"}
3. Observatory: Shows live DAG progress
4. Cannon -> NATS: cannon.ingest.stage_complete {stage: "transcribe", rows: 847}
5. Cannon -> NATS: cannon.highlight.found {timestamp: 145.2, score: 0.89}
6. Memory: Stores highlight as HRM memory with importance 0.8
7. Cannon -> NATS: cannon.ingest.completed {stages: 22, duration: 342s}
8. Memory: Stores full analysis summary
9. Observatory: Updates constellation map with new project
```

## Implementation Status

Current status:

- [x] Cannon analysis pipeline (22 stages, production-ready)
- [x] Cannon MCP tools (production-ready)
- [x] Cannon SQLite + sqlite-vec storage (per-project)
- [x] HRM memory bridge (`clipcannon/hrm_bridge.py`)
- [x] Cross-project recall via HRM (`clipcannon_hrm_recall` MCP tool)
- [ ] NATS event publisher (planned)
- [ ] Observatory visualization hooks (planned)

### What's wired today

`clipcannon/hrm_bridge.py` shells out to the `kannaka` binary (located via
`KANNAKA_BIN`, else PATH, with `KANNAKA_QUIET=1`). Each `kannaka remember` /
`kannaka recall` is a short-lived CLI invocation — no long-lived process is
spawned.

**When:** on the pipeline `finalize` stage, after a project is marked ready.
The ingest is fire-and-forget — a no-op when the binary is absent, and any
error is logged and swallowed so it can never fail the (required) finalize
stage.

**What flows in:** one concise memory per stem type (a summary, not full
text), read from the project's `analysis.db`:

| Stem | Memory | `--modality` | Importance | Tags |
|------|--------|--------------|------------|------|
| Transcript | capped summary of segment text (word count in the header) | `semantic` | `0.50 + words/5000`, ≤ 0.85 | `cannon,transcript,<project_id>` |
| Scenes | shot-type distribution + sample descriptions | `visual` | `0.40 + scenes/100`, ≤ 0.70 | `cannon,scene,<project_id>` |
| Music / beats | tempo, beat count, section types | `audio` | 0.50 | `cannon,music,<project_id>` |

Each memory text is prefixed `[cannon:<project_id>] <title> <stem> …` and the
salient content is capped (default 600 chars) so the HRM isn't flooded.

**Envs:**

- `KANNAKA_BIN` — path to the kannaka binary (else looked up on PATH).
- `CLIPCANNON_HRM_INGEST` — automatic post-finalize ingest toggle. Defaults
  **on** (runs whenever the binary is present); set to `0`/`false`/`off` to
  disable. The explicit `clipcannon_hrm_ingest` tool ignores this gate.

**MCP tools:**

- `clipcannon_hrm_recall {query, top_k}` — cross-project recall over the HRM
  via `kannaka recall`. (Distinct from `clipcannon_search_content`, which does
  per-project sqlite-vec / text search within one project's DB.)
- `clipcannon_hrm_ingest {project_id}` — backfill an already-analyzed project.

Both return `hrm_available: false` instead of erroring when the binary is
absent — Cannon works standalone, and the HRM/NATS/Observatory integration
layers on top without breaking existing functionality.
