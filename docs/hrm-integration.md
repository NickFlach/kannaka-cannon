# HRM Integration Architecture

How Kannaka Cannon bridges video perception with the Holographic Resonance Medium.

## Overview

Kannaka Cannon's 22-stage analysis pipeline produces structured intelligence from raw video: transcripts, emotion contours, visual embeddings, speaker identities, narrative structure, and highlight scores. The HRM integration layer stores these results as wave-interference memories, making video intelligence a first-class citizen in the Kannaka constellation.

## Video Analysis Memories

### What Gets Stored

After each successful `clipcannon_ingest` pipeline run, the following are candidates for HRM storage:

| Analysis Output | Memory Type | Importance | Example Query |
|-----------------|-------------|------------|---------------|
| Transcript segments | Semantic | 0.6 | "find where they discuss pricing" |
| Highlight moments | Emotional | 0.8 | "most exciting moment in any video" |
| Speaker profiles | Identity | 0.7 | "everything Sarah said across all projects" |
| Narrative summaries | Structural | 0.7 | "videos with a strong call-to-action ending" |
| Emotion contours | Affective | 0.5 | "moments where the audience laughed" |
| OCR text | Factual | 0.4 | "slides mentioning quarterly revenue" |
| Scene descriptions | Visual | 0.5 | "outdoor scenes with mountains" |

### Storage Format

Each memory entry includes:

```json
{
  "source": "cannon",
  "project_id": "proj_abc123",
  "video_path": "/path/to/video.mp4",
  "timestamp_range": [12.5, 18.3],
  "content": "Speaker discusses the impact of AI on healthcare outcomes",
  "embedding": [0.12, -0.34, ...],
  "importance": 0.75,
  "tags": ["transcript", "ai", "healthcare"],
  "provenance_hash": "sha256:abc123..."
}
```

### How It Would Work

```bash
# After Cannon completes an ingest, store key memories
kannaka remember "Conference keynote: speaker passionate about AI healthcare, highlight at 2:15-2:45" --importance 0.8

# Later, from any constellation member
kannaka recall "AI healthcare discussion" --top-k 5
# Returns: project ID, timestamp range, transcript snippet, highlight score
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
`KANNAKA_BIN`, else PATH). On `finalize`, every analyzed project's stem
outputs are stored as HRM memories, best-effort and non-blocking:

| Stem output | `--modality` | Importance | Tags |
|-------------|--------------|------------|------|
| Transcript segments | `semantic` | 0.6 | `cannon,transcript,<project_id>` |
| Scene descriptions | `visual` | 0.5 | `cannon,scene,<project_id>` |
| Music / beat features | `audio` | 0.5 | `cannon,music,<project_id>` |

Two MCP tools expose the bridge:

- `clipcannon_hrm_recall {query, top_k}` — cross-project recall over the HRM.
- `clipcannon_hrm_ingest {project_id}` — backfill an already-analyzed project.

If the `kannaka` binary is absent, ingest is a no-op and recall returns
`hrm_available: false` — Cannon works standalone, and the HRM/NATS/Observatory
integration layers on top without breaking existing functionality.
