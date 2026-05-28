---
name: skill-kannaka-cannon
version: 0.1.0
description: "Kannaka Cannon — AI video intelligence via the `clipcannon` MCP server (51 tools). Use when: user wants to analyze/understand a video, transcribe or caption it, find the best moments / cut points, build and render an edit, generate music/SFX, clone a voice / speak, or drive a lip-sync avatar. Covers installing + registering the MCP server and the ingest → understand → edit → render → voice workflow. The stems it extracts become HRM memories (see skill-kannaka-memory)."
---

# Kannaka Cannon — AI video intelligence (MCP)

## What this is

`kannaka-cannon` is the constellation's eyes-on-pixels. It is an **MCP server** (not a CLI)
exposing **51 `clipcannon_*` tools** backed by a 22-stage pipeline that ingests a clip,
decomposes it into stems (scene, motion, speech/ASR, music separation, faces/emotion, OCR
text), and folds those stems into the same HRM every kannaka node uses. A 5-minute clip
becomes one HRM cluster with member memories per scene / speaker turn / face — later
recallable with `kannaka recall "the scene where the bridge crosses"`.

- **Transport**: stdio MCP (`clipcannon.server:main`)
- **Console script**: `clipcannon`
- **Config**: `~/.clipcannon/config.json`
- **Projects**: per-project SQLite + `sqlite-vec` under `~/.clipcannon/projects/`
- **Tool prefix**: every tool is named `clipcannon_*` (MCP protocol identifiers)

## When to use this skill

- "analyze / understand this video", "what happens in this clip?"
- "transcribe / caption it", "get the transcript", "scene map"
- "find the best moments / cut points / safe cuts"
- "make an edit", "trim", "add an overlay / motion / color", "render it"
- "generate music / SFX / a video", "compose music"
- "clone this voice", "speak this line", "lip-sync avatar"
- "what's this costing?" (credits / billing tools)

Do NOT use for:
- Plain memory recall/store of the resulting clusters → `skill-kannaka-memory`
  (`kannaka recall`, `kannaka observe`)
- Radio station, markets, constellation health → the respective kannaka skills

---

## Install + register the MCP server

```bash
git clone https://github.com/NickFlach/kannaka-cannon.git
cd kannaka-cannon
uv sync                 # base — MCP server + lightweight ops
uv sync --group ml      # torch + faster-whisper + transformers (real understanding)
uv sync --group phase2  # audio processing + mediapipe + ace-step (music/faces)
uv sync --group phase3  # lip-sync + voice (LatentSync, Qwen3-TTS)
```

Python 3.12+. Heavy ML deps live behind the optional `ml` / `phase2` / `phase3` groups, so
base install stays light — but understanding/voice/avatar tools need their group installed
or they degrade / error.

Register the stdio server with Claude Code (run from the repo so `uv` resolves the venv):

```bash
claude mcp add kannaka-cannon -- uv run --directory /ABS/PATH/kannaka-cannon clipcannon
# or, if `clipcannon` is already on PATH (e.g. `uv tool install .`):
claude mcp add kannaka-cannon -- clipcannon
```

Then the `clipcannon_*` tools appear as callable MCP tools. There is no CLI surface to
shell out to — drive everything through the MCP tools.

### Environment

| Var | For |
|-----|-----|
| `ANTHROPIC_API_KEY` / `KANNAKA_LLM_API_KEY` | LLM-backed understanding / narrative |
| `HUGGING_FACE_HUB_TOKEN` | pulling ML model weights |
| `CLIPCANNON_LICENSE_URL` | license check (billing/credits tools) |
| `STRIPE_PUBLISHABLE_KEY` | paid credit purchases (billing) |

---

## The workflow (tool order matters)

```
project_create  →  ingest  →  (understand / search)  →  create_edit / modify_edit
                →  preview  →  render  →  speak / lip_sync (voice + avatar)
```

1. **Open a workspace**: `clipcannon_project_create` (then `project_open` / `project_status`).
2. **Ingest** the source: `clipcannon_ingest` — runs the 22-stage scan, writes stems, folds
   them into the HRM. Everything downstream reads from the ingested project.
3. **Understand** before editing: `clipcannon_get_transcript`, `clipcannon_get_scene_map`,
   `clipcannon_get_narrative_flow`, `clipcannon_search_content`, `clipcannon_find_best_moments`,
   `clipcannon_find_cut_points` / `clipcannon_find_safe_cuts`, `clipcannon_analyze_frame`.
4. **Edit** declaratively: `clipcannon_create_edit`, `clipcannon_modify_edit`,
   `clipcannon_add_motion` / `add_overlay` / `color_adjust` / `auto_trim` / `auto_music`,
   with `branch_edit` / `list_branches` / `revert_edit` / `edit_history` for versioning.
5. **Preview** cheaply before paying for a full render: `clipcannon_preview_clip` /
   `preview_segment` / `preview_layout` (and the `preview_540p` fast path).
6. **Render**: `clipcannon_render`, then `clipcannon_inspect_render`.
7. **Voice + avatar**: `clipcannon_prepare_voice_data` → `voice_profiles` → `speak` /
   `speak_optimized`; `clipcannon_lip_sync` for the avatar.

Always `understand` before you `edit`, and `preview` before you `render` — render and the
generate/voice tools consume credits.

---

## Tool catalog (51 `clipcannon_*` tools, by group)

- **project**: `project_create`, `project_open`, `project_list`, `project_status`, `project_delete`
- **ingest / understanding**: `ingest`, `get_transcript`, `get_scene_map`, `get_narrative_flow`,
  `get_editing_context`, `get_frame`, `analyze_frame`, `extract_webcam`
- **discovery**: `find_best_moments`, `find_cut_points`, `find_safe_cuts`
- **search** (HRM-backed): `search_content`
- **editing**: `create_edit`, `modify_edit`, `add_motion`, `add_overlay`, `color_adjust`,
  `auto_trim`, `auto_music`, `apply_feedback`, `branch_edit`, `list_branches`, `revert_edit`,
  `edit_history`
- **rendering / preview**: `render`, `inspect_render`, `preview_clip`, `preview_segment`,
  `preview_layout`, `preview_540p`
- **audio**: `audio_cleanup`, `compose_music`, `compose_midi`, `generate_music`, `generate_sfx`
- **generate**: `generate_video`
- **voice**: `prepare_voice_data`, `voice_profiles`, `speak`, `speak_optimized`
- **avatar**: `lip_sync`
- **config**: `config_get`, `config_set`, `config_list`
- **disk**: `disk_status`, `disk_cleanup`
- **billing / credits**: `credits_balance`, `credits_estimate`, `credits_history`,
  `spending_limit`

> Tool names are exact MCP identifiers — call them verbatim (with the `clipcannon_` prefix).
> `render`, `generate_*`, and the `speak`/`lip_sync` tools cost credits; use
> `credits_estimate` first and respect `spending_limit`.

## Version

Skill 0.1.0 covers kannaka-cannon 0.1.0 (51 MCP tools, stdio transport, `~/.clipcannon`
config + per-project sqlite-vec).
