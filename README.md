```
 ██████╗ █████╗ ███╗   ██╗███╗   ██╗ ██████╗ ███╗   ██╗
██╔════╝██╔══██╗████╗  ██║████╗  ██║██╔═══██╗████╗  ██║
██║     ███████║██╔██╗ ██║██╔██╗ ██║██║   ██║██╔██╗ ██║
██║     ██╔══██║██║╚██╗██║██║╚██╗██║██║   ██║██║╚██╗██║
╚██████╗██║  ██║██║ ╚████║██║ ╚████║╚██████╔╝██║ ╚████║
 ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═══╝
   V I D E O · I N T E L L I G E N C E
```

**AI video intelligence — powered by Holographic Resonance Memory.**

`kannaka-cannon` is the constellation's eyes-on-pixels: a 22-stage video analysis pipeline that ingests any clip, decomposes it into stems (scene, motion, speech, music, faces, text, emotion), and lights up specialized MCP tools to operate on each. Voice cloning, lip-sync avatars, captioning, scene transition detection. The substrate underneath is the same HRM every kannaka node uses.

[![License](https://img.shields.io/badge/license-BUSL--1.1-blueviolet)]() [![Python](https://img.shields.io/badge/python-3.12+-yellow)]() [![MCP](https://img.shields.io/badge/MCP-51%20tools-purple)]()

---

## What's Inside

```
┌─────────────────────────────────────────────────────────────┐
│                     kannaka-cannon                          │
├─────────────────────┬───────────────────────────────────────┤
│  22-stage pipeline  │  MCP server (51 tools)                │
│  · scene split      │  · clip · transcribe · caption        │
│  · motion vectors   │  · voice-clone · lip-sync · avatar    │
│  · speech ASR       │  · scene-detect · cut · concat        │
│  · music separation │  · upload · publish · post            │
│  · faces / emotion  │  · vector-search (HRM-backed)         │
│  · OCR text         │  · billing · usage · quota            │
├─────────────────────┴───────────────────────────────────────┤
│  Stem outputs                                               │
│  · audio (vocals / inst / sfx)   · video (scenes / frames)  │
│  · transcript    · captions       · faces.json              │
└─────────────────────────────────────────────────────────────┘
```

Heavy ML deps (torch, faster-whisper, transformers, demucs, mediapipe, etc.) live behind optional `ml` / `phase2` / `phase3` dependency groups so the base install stays light.

---

## Install

```bash
git clone https://github.com/NickFlach/kannaka-cannon.git
cd kannaka-cannon
uv sync                    # base — MCP server + lightweight ops
uv sync --extra dev        # adds pytest + ruff for development
uv sync --group ml         # adds torch + whisper + transformers
uv sync --group phase2     # audio processing + mediapipe + ace-step
uv sync --group phase3     # lip-sync + voice (LatentSync, Qwen3-TTS)
```

Python 3.12+ required. Uses `uv` for fast deterministic installs.

---

## Run the dashboard

```bash
# Local development — auto-login, no secret required
CLIPCANNON_DEV_MODE=1 uv run python -m clipcannon.dashboard.app

# Production — dev mode is OFF by default and a real JWT secret is mandatory
export CLIPCANNON_JWT_SECRET="<a strong, unique value>"
uv run python -m clipcannon.dashboard.app
```

`CLIPCANNON_DEV_MODE` defaults to **off** (production). In dev mode the
dashboard auto-authenticates and signs sessions with a well-known dev
secret. In production the server **fails hard at startup** unless
`CLIPCANNON_JWT_SECRET` is set to a non-default value, so forged session
cookies are impossible. Set `CLIPCANNON_DEV_MODE=1` for local work.

---

## Architecture

```
              video file
                  ↓
         ┌────────────────┐
         │ 22-stage scan  │
         └───────┬────────┘
                 │  stems
       ┌─────────┼─────────────────┐
       │         │                 │
       ▼         ▼                 ▼
    audio    video frames     transcript
       │         │                 │
       └─────────┼─────────────────┘
                 │
         ┌───────▼────────┐
         │   HRM ingest   │ ─ → wavefronts in kannaka-memory
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │   MCP tools    │ ─ → caller invokes via stdio/HTTP
         └────────────────┘
```

Output stems become first-class memories: a 5-minute clip becomes one cluster in the HRM, with member memories for each scene, each speaker turn, each face appearance. Later `kannaka recall "the scene where the bridge crosses"` resonates against that cluster.

---

## License

BUSL-1.1.
