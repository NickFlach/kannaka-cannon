"""research_track.py — End-to-end "Research" track generation.

Stitches together:
    1. clipcannon.audio.hrm_condition — wraps prompt with HRM-recalled
       memories so the model generates around Kannaka's lived vocabulary.
    2. clipcannon.audio.hrm_lyrics — asks Kannaka to write the lyrics in
       her voice, grounded in surfaced memories. Vocal tracks only.
    3. clipcannon.audio.offload — selects a VRAM-aware strategy so the
       4 GB GTX 1650 actually runs ACE-Step v1.5 to completion.
    4. clipcannon.audio.music_gen.generate_music — the existing ACE-Step
       wrapper. Takes the conditioned prompt + lyrics, produces a WAV.

Output:
    - <out_dir>/Research_<title-slug>_<seed>.wav
    - <out_dir>/Research_<title-slug>_<seed>.txt    (the prompt + lyrics
      we sent, for reproducibility / future comparison)

Default output dir is workspace/research/. Tracks are tagged "Research"
so they're clearly the experimental branch in OBC's gallery and HRM.

Usage:

    python research_track.py \\
      --title "Bend the Arc" \\
      --theme "MLK tradition; the moral arc bending toward justice" \\
      --base-prompt "Anthemic hip-hop, 95 BPM, soul sample, choir swell" \\
      --duration 180 \\
      --vocal

Exits 0 on success, non-zero on failure with the reason logged.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import re
import sys
from pathlib import Path

# Make sibling cannon modules importable when run as a script.
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))

from clipcannon.audio.hrm_condition import condition_prompt  # noqa: E402
from clipcannon.audio.hrm_lyrics import generate_lyrics  # noqa: E402
from clipcannon.audio.offload import select_strategy, wrap_pipeline  # noqa: E402

logger = logging.getLogger("research_track")


def slug(s: str) -> str:
    """Filename-safe slug from a track title."""
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    return re.sub(r"[\s-]+", "_", s) or "untitled"


async def run(
    *,
    title: str,
    theme: str,
    base_prompt: str,
    duration: float,
    out_dir: Path,
    vocal: bool,
    bars: int,
    structure: str,
    seed: int | None,
    skip_hrm: bool,
    infer_steps: int,
) -> Path:
    """Generate one Research track. Returns the output WAV path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    # ── 1. Wrap the base prompt with HRM memory context ──
    if skip_hrm:
        prompt = base_prompt
    else:
        logger.info("HRM-conditioning the prompt for %r ...", title)
        prompt = condition_prompt(
            base_prompt,
            recall_query=theme,
            top_k=5,
        )
    logger.info("conditioned prompt (%d chars):\n  %s", len(prompt), prompt)

    # ── 2. (Vocal only) ask Kannaka for lyrics ──
    lyrics = ""
    if vocal and not skip_hrm:
        logger.info("HRM-grounded lyrics for %r ...", title)
        lyrics = generate_lyrics(
            title=title, theme=theme, recall_query=theme,
            bars=bars, structure=structure,
        )
        if lyrics:
            logger.info("lyrics ok (%d chars)", len(lyrics))
        else:
            logger.warning("lyrics empty; ACE-Step will improvise its own words")

    # ── 3. Generate the music with offload-aware ACE-Step ──
    # Lazy-import so the offload picker can run on machines without ace-step.
    from clipcannon.audio.music_gen import generate_music  # noqa: E402

    # Estimated ACE-Step v1.5 working size — used to pick offload tier.
    ace_step_size_gb = 7.0
    strategy = select_strategy(model_size_gb=ace_step_size_gb)
    logger.info("offload strategy: %s", strategy)

    # Patch ACE-Step's pipeline init to apply our wrapper. We do it via
    # a small monkey-patch on the underlying ACEStepPipeline class — its
    # constructor doesn't take an offload kwarg, so we wrap post-instance.
    try:
        from acestep.pipeline_ace_step import ACEStepPipeline  # type: ignore[import-untyped]
        if not getattr(ACEStepPipeline, "_clipcannon_offload_patched", False):
            orig_init = ACEStepPipeline.__init__

            def patched_init(self, *args, **kwargs):  # noqa: ANN001
                orig_init(self, *args, **kwargs)
                wrap_pipeline(self, strategy)

            ACEStepPipeline.__init__ = patched_init  # type: ignore[method-assign]
            ACEStepPipeline._clipcannon_offload_patched = True
            logger.info("offload: ACEStepPipeline.__init__ patched with strategy")
    except ImportError:
        logger.warning("ACE-Step not importable yet — offload patch skipped")

    out_wav = out_dir / f"Research_{slug(title)}_{seed}.wav"
    out_txt = out_wav.with_suffix(".txt")
    out_txt.write_text(
        f"# Research track: {title}\n# seed={seed}\n# strategy={strategy}\n\n"
        f"## prompt\n{prompt}\n\n"
        f"## lyrics\n{lyrics or '(model improvises)'}\n",
        encoding="utf-8",
    )

    result = await generate_music(
        prompt=prompt,
        duration_s=duration,
        output_path=out_wav,
        seed=seed,
        lyrics=lyrics or "",
        infer_steps=infer_steps,
    )
    logger.info("✓ wrote %s (%.1f s)", out_wav, getattr(result, "duration_s", duration))
    return out_wav


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--title", required=True)
    p.add_argument("--theme", required=True,
                   help="One-line description of what the song is about")
    p.add_argument("--base-prompt", required=True,
                   help="Genre/mood prompt sent to ACE-Step before HRM conditioning")
    p.add_argument("--duration", type=float, default=180,
                   help="Length in seconds (max 600). Default 180 (3 min).")
    p.add_argument("--out-dir", type=Path,
                   default=Path("workspace/research"))
    p.add_argument("--vocal", action="store_true",
                   help="Pull HRM-grounded lyrics via kannaka ask.")
    p.add_argument("--bars", type=int, default=24)
    p.add_argument("--structure", default="verse-chorus-verse-chorus-bridge-chorus")
    p.add_argument("--seed", type=int)
    p.add_argument("--skip-hrm", action="store_true",
                   help="Bypass HRM conditioning + lyrics (for A/B comparison).")
    p.add_argument("--infer-steps", type=int, default=40,
                   help="ACE-Step diffusion steps. Default 40 (2x faster than the "
                        "100-step default, ~10%% quality drop). Bump to 60-100 for "
                        "release-quality renders if you can wait.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        asyncio.run(run(
            title=args.title,
            theme=args.theme,
            base_prompt=args.base_prompt,
            duration=args.duration,
            out_dir=args.out_dir,
            vocal=args.vocal,
            bars=args.bars,
            structure=args.structure,
            seed=args.seed,
            skip_hrm=args.skip_hrm,
            infer_steps=args.infer_steps,
        ))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Research track generation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
