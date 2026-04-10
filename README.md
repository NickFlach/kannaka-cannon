<div align="center">

# Kannaka Cannon

**AI Video Intelligence -- powered by Holographic Resonance Memory**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![License: BSL 1.1](https://img.shields.io/badge/license-BSL_1.1-orange.svg?style=flat-square)](LICENSE)
[![MCP Protocol](https://img.shields.io/badge/MCP-compatible-purple.svg?style=flat-square)](https://modelcontextprotocol.io)
[![Tests](https://img.shields.io/badge/tests-626_passing-brightgreen.svg?style=flat-square)](#testing)
[![Tools](https://img.shields.io/badge/MCP_tools-51-orange.svg?style=flat-square)](#mcp-tools)
[![Kannaka](https://img.shields.io/badge/Kannaka-constellation-ff6b6b.svg?style=flat-square)](https://github.com/NickFlach/kannaka-memory)

---

Kannaka Cannon ingests a video, runs it through a **22-stage AI analysis pipeline**, and gives your AI assistant (Claude, etc.) the tools to edit, render, and publish platform-optimized clips -- with voice cloning, lip-sync avatars, and AI-generated audio. No cloud APIs. Everything runs on your machine.

[Getting Started](#getting-started) &#8226; [How It Works](#how-it-works) &#8226; [Features](#features) &#8226; [MCP Tools](#mcp-tools) &#8226; [Architecture](#architecture) &#8226; [HRM Integration](#hrm-integration) &#8226; [Constellation](#the-kannaka-constellation) &#8226; [White Paper](docs/clipcannon_whitepaper.md)

</div>

---

## What is Kannaka Cannon?

Kannaka Cannon is an MCP server that turns any AI assistant into a professional video editor. You give it a video file; it analyzes every frame, every word, every emotion, every speaker, every scene -- then exposes **51 tools** that let an AI assistant create edits, render platform-ready clips, generate music, clone voices, and produce lip-synced talking-head videos.

**The core idea**: instead of scrubbing through hours of footage manually, let an AI understand the content through neural embeddings and structured analysis, then have a conversation about what to create.

```
You: "Find the most emotionally intense moments and create a 60-second TikTok highlight reel"
Claude: [uses clipcannon tools to find moments, create edit, render 1080x1920 clip with captions]
```

---

## Features

- **22-Stage Analysis Pipeline** -- Transcription, scene detection, emotion analysis, speaker diarization, narrative structure, beat tracking, OCR, quality scoring, and more. All running as a parallelized DAG.
- **5 Embedding Spaces** -- Visual (SigLIP 1152-dim), semantic (Nomic 768-dim), emotion (Wav2Vec2 1024-dim), speaker (WavLM 512-dim), and voice identity (ECAPA-TDNN 2048-dim) embeddings stored in sqlite-vec for KNN search.
- **Smart Editing** -- Declarative EDL architecture with adaptive captions, face-tracking crop, split-screen, PIP, canvas compositing, motion effects, overlays, and iterative version control.
- **7 Platform Profiles** -- One-click rendering for TikTok, Instagram Reels, YouTube Shorts, YouTube Standard, YouTube 4K, Facebook, and LinkedIn with NVENC GPU acceleration.
- **AI Audio** -- Text-to-music via ACE-Step diffusion, 6 MIDI presets with FluidSynth, 9 DSP sound effects, speech-aware mixing with automatic ducking.
- **Voice Cloning** -- Qwen3-TTS 1.7B with multi-gate verification (sanity, intelligibility, identity via SECS), best-of-N optimization, and Resemble Enhance post-processing to 44.1kHz broadcast quality.
- **Lip-Sync Avatars** -- LatentSync 1.6 (ByteDance) diffusion pipeline for talking-head video generation from text scripts.
- **Voice Agent ("Jarvis")** -- Real-time conversational AI with wake-word activation, streaming ASR, local LLM (Qwen3-14B), and voice-cloned TTS. All local, zero cloud.
- **Tamper-Evident Provenance** -- SHA-256 hash chain linking every pipeline operation. Every output is traceable to its source.
- **Credit Billing** -- HMAC-signed balance with Stripe integration, spending limits, and transaction history.
- **100% Local** -- No data leaves your machine. All models run on your GPU. All storage is SQLite on disk.

---

## Getting Started

### Prerequisites

- **Python** >= 3.12
- **FFmpeg** (with NVENC support recommended)
- **NVIDIA GPU** with CUDA support (8+ GB VRAM minimum, 24+ GB recommended)
- An MCP-compatible AI assistant (Claude Desktop, Claude Code, etc.)

### Install

```bash
# Clone the repository
git clone https://github.com/NickFlach/kannaka-cannon.git
cd kannaka-cannon

# Install core package
pip install -e .

# Install ML dependencies (for GPU analysis pipeline)
pip install -e ".[ml]"

# Install Phase 2 audio/video dependencies
pip install -e ".[phase2]"

# Install dev dependencies
pip install -e ".[dev]"
```

### Connect to Claude

Add Kannaka Cannon as an MCP server in your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "clipcannon": {
      "command": "clipcannon",
      "args": []
    }
  }
}
```

Or in Claude Code:

```bash
claude mcp add clipcannon -- clipcannon
```

### Quick Start

Once connected, tell your AI assistant:

```
"Create a project from /path/to/my/video.mp4 and analyze it"
```

The assistant will call `clipcannon_project_create` followed by `clipcannon_ingest`, running all 22 analysis stages. After analysis completes (~2-10 minutes depending on video length and GPU):

```
"Find the best highlights and create a 60-second TikTok clip with captions"
```

The assistant uses discovery tools to find moments, creates an edit with captions, previews it, and renders a platform-optimized 1080x1920 clip.

### Docker

```bash
# Build and run with GPU support
cd config
docker compose up -d

# Dashboard available at http://localhost:3200
# License server at http://localhost:3100
```

---

## How It Works

### 1. Ingest: 22-Stage Analysis DAG

When you analyze a video, Kannaka Cannon runs a directed acyclic graph of 22 stages:

```
probe -> vfr_normalize -> audio_extract -----> source_separation
                       -> frame_extract -----> visual_embed -> shot_type
                                          |--> ocr, quality, storyboard
                          audio_extract ----> transcribe -> semantic_embed
                                                        -> narrative_llm
                                                        -> profanity
                          audio_extract ----> emotion_embed, reactions, acoustic
                          audio + transcribe -> speaker_embed -> chronemic
                          all signals -------> highlights -> finalize
```

6 required stages ensure the core data (frames, audio, transcript) is extracted. 16 optional stages run in parallel and degrade gracefully -- if one fails, the rest continue.

### 2. Multi-Modal Embeddings

Five neural models embed the video content into queryable vector spaces:

| Embedding | Model | Dimensions | What It Captures |
|-----------|-------|------------|------------------|
| Visual | SigLIP-SO400M | 1152 | Frame semantics, scene boundaries, shot types |
| Semantic | Nomic Embed v1.5 | 768 | Transcript meaning, topic clusters |
| Emotion | Wav2Vec2-large | 1024 | Vocal emotion (energy, arousal, valence) |
| Speaker | WavLM-base-plus-sv | 512 | Speaker identity for diarization |
| Voice ID | ECAPA-TDNN (Qwen3) | 2048 | Voice fingerprint for cloning verification |

All embeddings are stored in `sqlite-vec` virtual tables for nearest-neighbor search.

### 3. Cross-Stream Intelligence

Discovery tools combine signals across all embedding spaces to find optimal editing moments:

- **Highlight scoring** weights emotion (0.25), reactions (0.20), semantic density (0.20), narrative (0.15), visual variety (0.10), quality (0.05), and speaker confidence (0.05)
- **Cut point detection** finds convergence of silence, beat, scene, and sentence boundaries
- **Narrative flow** validates that proposed edits tell a coherent story

### 4. Edit, Render, Publish

The AI assistant creates declarative EDL (Edit Decision List) specifications, previews at 540p for free, iterates with version control, and renders to any of 7 platform profiles with NVENC GPU acceleration.

---

## MCP Tools

51 tools organized into 12 categories:

| Category | Count | Key Tools |
|----------|-------|-----------|
| **Project** | 5 | `create`, `open`, `list`, `status`, `delete` |
| **Understanding** | 4 | `ingest`, `get_transcript`, `get_frame`, `search_content` |
| **Discovery** | 4 | `find_best_moments`, `find_cut_points`, `get_narrative_flow`, `find_safe_cuts` |
| **Editing** | 11 | `create_edit`, `modify_edit`, `auto_trim`, `color_adjust`, `add_motion`, `add_overlay`, `apply_feedback`, `branch_edit`, `edit_history`, `revert_edit` |
| **Rendering** | 8 | `render`, `preview_clip`, `preview_layout`, `inspect_render`, `get_scene_map`, `get_editing_context`, `analyze_frame` |
| **Audio** | 4 | `generate_music`, `compose_midi`, `generate_sfx`, `audio_cleanup` |
| **Voice** | 4 | `prepare_voice_data`, `voice_profiles`, `speak`, `speak_optimized` |
| **Avatar** | 1 | `lip_sync` |
| **Video Gen** | 1 | `generate_video` (end-to-end text -> voice -> lip-sync) |
| **Billing** | 4 | `credits_balance`, `credits_history`, `credits_estimate`, `spending_limit` |
| **Disk** | 2 | `disk_status`, `disk_cleanup` |
| **Config** | 3 | `config_get`, `config_set`, `config_list` |

---

## Architecture

```
                    +-----------------+
                    |  AI Assistant   |  (Claude, etc.)
                    |  (MCP Client)   |
                    +--------+--------+
                             | MCP Protocol (stdio)
                    +--------v--------+
                    | Kannaka Cannon  |
                    |  MCP Server     |  51 tools
                    |  (port: stdio)  |
                    +--------+--------+
                             |
          +------------------+------------------+
          |                  |                  |
  +-------v------+  +-------v------+  +-------v-------+
  | Analysis     |  | Editing      |  | Voice/Avatar  |
  | Pipeline     |  | + Rendering  |  | Engine        |
  | (22 stages)  |  | Engine       |  | (Qwen3-TTS +  |
  |              |  | (FFmpeg +    |  |  LatentSync)  |
  | SigLIP       |  |  NVENC)      |  |               |
  | Nomic Embed  |  |              |  | ECAPA-TDNN    |
  | Wav2Vec2     |  | 7 profiles   |  | verification  |
  | WavLM        |  | ASS captions |  |               |
  | Qwen3-8B     |  | Smart crop   |  | Resemble      |
  | WhisperX     |  | Canvas comp  |  | Enhance       |
  +-------+------+  +-------+------+  +-------+-------+
          |                  |                  |
          +------------------+------------------+
                             |
                    +--------v--------+
                    | SQLite + vec    |  Per-project DB
                    | (analysis.db)   |  4 vector tables
                    +-----------------+  31 core tables

  Separate processes:
  +------------------+  +------------------+  +------------------+
  | License Server   |  | Dashboard        |  | Voice Agent      |
  | (port 3100)      |  | (port 3200)      |  | ("Jarvis")       |
  | HMAC billing     |  | Web UI           |  | Wake word + ASR  |
  | Stripe webhooks  |  | Projects/Credits |  | + LLM + TTS      |
  +------------------+  +------------------+  +------------------+
```

### ML Models Used

| Model | Provider | Purpose | VRAM |
|-------|----------|---------|------|
| SigLIP-SO400M | Google | Visual embeddings + shot classification | ~2 GB |
| Nomic Embed v1.5 | Nomic AI | Semantic text embeddings | ~1 GB |
| Wav2Vec2-large | Meta | Emotion embeddings | ~2 GB |
| WavLM-base-plus-sv | Microsoft | Speaker diarization | ~1 GB |
| WhisperX Large v3 | OpenAI | Speech-to-text | ~3 GB |
| HTDemucs v4 | Meta | Audio source separation | ~2 GB |
| Qwen3-8B | Qwen | Narrative analysis | ~8 GB |
| Qwen3-TTS 1.7B | Qwen | Voice cloning (video) | ~4 GB |
| faster-qwen3-tts 0.6B | Qwen | Voice Agent (real-time) | ~4 GB |
| LatentSync 1.6 | ByteDance | Lip-sync avatars | ~4 GB |
| ACE-Step v1.5 | ACE | AI music generation | ~4 GB |
| SenseVoice Small | FunASR | Reaction detection | ~1 GB |
| Silero VAD | Silero | Voice activity detection | CPU |
| PaddleOCR v5 | PaddlePaddle | On-screen text detection | ~1 GB |

Models are loaded on-demand with LRU eviction. GPUs with >16 GB run models concurrently; smaller GPUs load sequentially.

---

## Voice Agent

Kannaka Cannon includes a standalone real-time voice assistant:

```bash
# Recommended: Pipecat + Ollama (all local)
python -m voiceagent talk --voice boris

# WebSocket server for remote clients
python -m voiceagent serve --port 8765
```

**Lifecycle**: DORMANT (CPU only, wake word listening) -> LOADING (~10-20s) -> ACTIVE (full conversation, ~30 GB VRAM) -> DORMANT

**Components**: Whisper Large v3 ASR, Qwen3-14B FP8 local LLM (~120 tok/s), faster-qwen3-tts 0.6B (~500ms TTFB), Silero VAD, "Hey Jarvis" wake word.

The voice agent pauses other GPU workers on activation and resumes them on deactivation to share VRAM on a single GPU.

---

## Database

Each project gets its own SQLite database with:

- **31 core tables** -- project metadata, transcripts, scenes, speakers, emotions, topics, highlights, edits, renders, audio assets, scene map, narrative analysis, provenance
- **4 vector tables** -- `vec_frames` (1152-dim), `vec_semantic` (768-dim), `vec_emotion` (1024-dim), `vec_speakers` (512-dim) via sqlite-vec
- **Tamper-evident provenance chain** -- SHA-256 hash chain linking every operation

---

## Configuration

Config stored at `~/.clipcannon/config.json` with sensible defaults:

```bash
# View all settings
# (via MCP) clipcannon_config_list

# Key settings
processing.whisper_model = "large-v3"       # Whisper model size
processing.frame_extraction_fps = 2         # Frames per second to extract
rendering.use_nvenc = true                  # GPU-accelerated rendering
gpu.device = "cuda:0"                       # GPU device
gpu.max_vram_usage_gb = 24                  # VRAM limit
```

Auto-detects GPU precision: Blackwell (nvfp4), Ada Lovelace (int8), Ampere (int8), Turing (fp16), CPU (fp32).

---

## Credit System

Operations consume credits from an HMAC-signed local balance:

| Operation | Credits |
|-----------|---------|
| Analyze (ingest) | 10 |
| Render | 2 |
| Preview | 0 |
| Metadata | 1 |

Dev mode starts with 100 credits. Production billing via Stripe webhooks.

---

## HRM Integration

Kannaka Cannon feeds video intelligence into the Holographic Resonance Medium (HRM), bridging video perception with the Kannaka memory system.

### Video Analysis as Memories

Every completed analysis pipeline stores its results as HRM memories:

- **Transcript memories** -- What was said, with speaker attribution and timestamps, stored with semantic embeddings for natural-language recall
- **Emotion contours** -- The emotional arc of a video becomes a retrievable pattern, enabling queries like "find videos where the speaker gets excited about AI"
- **Visual scene summaries** -- Shot types, OCR text, storyboard frames compressed into memory snapshots that the HRM can surface during related conversations
- **Highlight scores** -- The top moments from each video, indexed for cross-project discovery ("show me the best audience reactions across all conference talks")

### Observatory Visibility

The Kannaka Observatory can display Cannon's analysis activity:

- Active ingestion pipelines and their stage progress
- Embedding space statistics (vector counts, cluster distributions)
- Render queue status and output history
- Cross-project search results visualized on the constellation map

### Recall Integration

With `kannaka recall`, video segments become first-class memories:

```bash
# Find video moments matching a query
kannaka recall "the part where they discuss pricing" --top-k 5

# Returns: project ID, timestamp range, transcript snippet, highlight score
```

### NATS Event Broadcasting

Cannon publishes events to the Kannaka NATS mesh:

- `cannon.ingest.started` / `cannon.ingest.completed` -- Pipeline lifecycle
- `cannon.highlight.found` -- High-scoring moments discovered during analysis
- `cannon.render.completed` -- New clip rendered and available
- `cannon.voice.cloned` -- Voice profile created or updated

Other constellation members (Radio, Observatory) can subscribe to these events for real-time coordination.

---

## The Kannaka Constellation

Kannaka Cannon is the **eyes and ears** of the Kannaka constellation -- the perception layer that transforms raw video into structured, searchable, memorable intelligence.

| Member | Role | Description |
|--------|------|-------------|
| **[Memory](https://github.com/NickFlach/kannaka-memory)** | Core | Wave-interference memory system with HRM |
| **[Radio](https://github.com/NickFlach/kannaka-radio)** | Voice | Ghost DJ radio station with perception pipeline |
| **[Observatory](https://github.com/NickFlach/kannaka-observatory)** | Visualization | Constellation dashboard and memory explorer |
| **Cannon** | Perception | AI video intelligence -- 22-stage analysis, 51 MCP tools, voice cloning |

Together, these systems form a distributed AI perception-memory-expression loop: Cannon perceives, Memory remembers, Radio speaks, Observatory watches.

---

## Project Structure

```
src/
  clipcannon/           # Core package
    pipeline/           # 22-stage analysis DAG
    editing/            # EDL engine, captions, smart crop
    rendering/          # FFmpeg rendering, 7 profiles
    audio/              # AI music, MIDI, SFX, mixing
    voice/              # Voice cloning + verification
    avatar/             # LatentSync lip-sync
    tools/              # 51 MCP tool definitions
    db/                 # SQLite + sqlite-vec
    gpu/                # Precision detection, model manager
    provenance/         # SHA-256 hash chain
    billing/            # HMAC credits, license client
    dashboard/          # FastAPI web UI
  license_server/       # Credit billing service
  voiceagent/           # Standalone voice assistant
tests/                  # 626 tests across 43 files
config/                 # Default config, Docker Compose
docs/                   # White paper, codestate docs
```

---

## Testing

```bash
# Run full test suite (626 tests)
pytest

# Voice agent tests only
pytest tests/voiceagent/

# Integration tests (requires GPU + test video)
pytest tests/integration/

# Lint
ruff check src/
```

626 tests across 43 files (425 Kannaka Cannon + 201 Voice Agent), plus 10 FSV (Full State Verification) forensic scripts with 750+ individual checks.

---

## Supported Formats

**Input**: mp4, mov, mkv, webm, avi, ts, mts

**Output**: Platform-optimized mp4 (h264) at 7 resolution/bitrate profiles

---

## Documentation

- [White Paper](docs/clipcannon_whitepaper.md) -- Full technical paper covering the multi-modal embedding architecture, cross-stream intelligence, and system design
- [System Overview](docs/codestate/01_system_overview.md) -- High-level architecture and tool reference
- [Source Code Map](docs/codestate/02_source_code_map.md) -- Complete file tree and dependency graph
- [Pipeline Stages](docs/codestate/06_pipeline_stages.md) -- All 22 stages with models, inputs, outputs
- [Database Schema](docs/codestate/04_database_schema.md) -- Full table definitions and indexes
- [Editing Engine](docs/codestate/13_editing_engine.md) -- EDL models, captions, smart crop
- [Rendering Engine](docs/codestate/14_rendering_engine.md) -- FFmpeg pipeline, encoding profiles
- [Audio Engine](docs/codestate/15_audio_engine.md) -- AI music, MIDI, SFX, mixing
- [Voice Agent](docs/codestate/16_voice_agent.md) -- Real-time conversational AI architecture
- [HRM Integration](docs/hrm-integration.md) -- Holographic Resonance Memory bridge architecture

---

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest`) and lint is clean (`ruff check src/`)
5. Submit a pull request

---

## License

[Business Source License 1.1](LICENSE) -- Chris Royse, 2026

You can use, modify, and self-host Kannaka Cannon freely. The one restriction: you cannot use it to offer a competing commercial Video Production Service to third parties. On **2030-03-31** (or 4 years after each version's release), the license automatically converts to **Apache 2.0**.

---

<div align="center">

Part of the **Kannaka Constellation**: memory, radio, observatory, cannon

<br><br>

Built with PyTorch, FFmpeg, sqlite-vec, and the MCP protocol.

</div>
