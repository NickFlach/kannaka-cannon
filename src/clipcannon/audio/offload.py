"""offload.py — VRAM-aware execution strategy for ACE-Step (and friends).

Detects available GPU memory and, when it's below the model's comfortable
ceiling, layers progressively more aggressive memory-saving techniques so
generation completes on smaller cards (e.g. GTX 1650 4 GB) instead of OOM.

Strategies, in order of preference:

    full_gpu          Default. Model fully resident on GPU. Fastest.
    sequential_offload  HF diffusers' `enable_sequential_cpu_offload()` —
                      each layer moves to GPU only for its forward pass,
                      then back to CPU. ~3-5x slower per step. No quality
                      change. Adds essentially the entire system RAM to
                      effective VRAM.
    bnb_4bit          Loads weights in 4-bit precision via bitsandbytes.
                      ~4x VRAM reduction. ~5% quality loss, ~10-20% slower.
    bnb_4bit_offload  Both stacked — best for very small VRAM. The slowest
                      but the most flexible; runs ~10-15 GB models on 4 GB.
    cpu_only          No GPU. Last resort. 10-30x slower than full GPU.

The selector is conservative — when in doubt, picks the heavier strategy.
A user can also force one with the OFFLOAD_STRATEGY env var.

Usage from music_gen.py:

    from clipcannon.audio.offload import select_strategy, wrap_pipeline

    strategy = select_strategy(model_size_gb=7.0)
    pipeline = ACEStepPipeline(...)
    pipeline = wrap_pipeline(pipeline, strategy)

The wrapper is best-effort. If a particular pipeline class doesn't support
the chosen offload hooks, we fall through to the next strategy and log a
warning. Music gen still completes; just at the next-tier speed/quality.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def select_strategy(
    model_size_gb: float,
    *,
    safety_factor: float = 1.5,
) -> str:
    """Pick an execution strategy based on available VRAM vs model size.

    `safety_factor` is the headroom multiplier: a model needs roughly
    1.5x its weight size in VRAM for inference (activations, KV cache,
    workspace). When VRAM is below that ceiling we shift strategies.

    Override with env: OFFLOAD_STRATEGY=full_gpu|sequential_offload|
                                       bnb_4bit|bnb_4bit_offload|cpu_only.
    """
    forced = os.environ.get("OFFLOAD_STRATEGY", "").strip().lower()
    if forced in {
        "full_gpu", "sequential_offload", "bnb_4bit",
        "bnb_4bit_offload", "cpu_only",
    }:
        logger.info("offload: strategy forced via env -> %s", forced)
        return forced

    try:
        import torch  # type: ignore[import-untyped]
    except ImportError:
        return "cpu_only"

    if not torch.cuda.is_available():
        return "cpu_only"

    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / 1e9
    needed_full = model_size_gb * safety_factor
    needed_offload = model_size_gb * 0.5
    needed_4bit_offload = model_size_gb * 0.15

    logger.info(
        "offload: vram=%.1f GB, model=%.1f GB, needed_full=%.1f, "
        "needed_offload=%.1f, needed_4bit_offload=%.1f",
        vram_gb, model_size_gb, needed_full, needed_offload, needed_4bit_offload,
    )

    if vram_gb >= needed_full:
        return "full_gpu"
    if vram_gb >= needed_offload:
        return "sequential_offload"
    if vram_gb >= needed_4bit_offload:
        return "bnb_4bit_offload"
    if vram_gb >= model_size_gb * 0.10:
        return "bnb_4bit"
    return "cpu_only"


def wrap_pipeline(pipeline: Any, strategy: str) -> Any:
    """Apply the chosen strategy to an ACE-Step / diffusers pipeline.

    Returns the same pipeline object (most strategies mutate in place)
    or a wrapper. Falls through to a less aggressive strategy if a hook
    isn't supported by the pipeline class.

    The pipeline objects vary — ACE-Step v1.5 wraps a diffusers DiT plus
    a small LM. We probe for known hooks and apply what's available.
    """
    if strategy == "full_gpu":
        return pipeline

    if strategy == "cpu_only":
        return _to_cpu(pipeline)

    if strategy == "sequential_offload":
        return _try_sequential_offload(pipeline) or _to_cpu(pipeline)

    if strategy == "bnb_4bit":
        # Pure 4-bit without offload — model resident on GPU but quantized.
        # ACE-Step's pipeline is usually loaded with float16; converting
        # post-hoc requires bitsandbytes' replace_with_4bit on submodules.
        return _try_bnb_4bit(pipeline) or _try_sequential_offload(pipeline) or _to_cpu(pipeline)

    if strategy == "bnb_4bit_offload":
        # Both stacked.
        result = _try_bnb_4bit(pipeline)
        if result is not None:
            return _try_sequential_offload(result) or result
        return _try_sequential_offload(pipeline) or _to_cpu(pipeline)

    logger.warning("offload: unknown strategy %r, falling back to full_gpu", strategy)
    return pipeline


# ── strategy implementations ─────────────────────────────────────────


def _try_sequential_offload(pipeline: Any) -> Any | None:
    """Try diffusers' standard offload hook. Returns None if unsupported."""
    fn = getattr(pipeline, "enable_sequential_cpu_offload", None)
    if callable(fn):
        try:
            fn()
            logger.info("offload: enabled sequential CPU offload")
            return pipeline
        except Exception as exc:  # noqa: BLE001
            logger.warning("offload: sequential_cpu_offload raised: %s", exc)
            return None

    # ACE-Step's pipeline may not expose the standard hook. Try walking
    # known submodule names and wrapping each diffusers component.
    sub_names = ["unet", "transformer", "vae", "text_encoder", "music_decoder"]
    applied_any = False
    for name in sub_names:
        sub = getattr(pipeline, name, None)
        if sub is None:
            continue
        sub_fn = getattr(sub, "enable_sequential_cpu_offload", None)
        if callable(sub_fn):
            try:
                sub_fn()
                applied_any = True
                logger.info("offload: applied sequential offload to %s", name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("offload: %s.enable_sequential_cpu_offload: %s", name, exc)

    return pipeline if applied_any else None


def _try_bnb_4bit(pipeline: Any) -> Any | None:
    """Walk the pipeline submodules and convert linear layers to 4-bit.

    Uses bitsandbytes' `replace_with_bnb_linear` (via transformers) on
    each torch.nn.Module child. Returns None if bitsandbytes isn't
    installed or no conversion succeeds.
    """
    try:
        from transformers.utils.bitsandbytes import (  # type: ignore[import-untyped]
            replace_with_bnb_linear,
        )
        from transformers import BitsAndBytesConfig  # type: ignore[import-untyped]
    except ImportError:
        logger.info("offload: bitsandbytes/transformers not installed; skipping 4-bit")
        return None

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=_pick_compute_dtype(),
        bnb_4bit_use_double_quant=True,
    )

    import torch  # type: ignore[import-untyped]
    converted_any = False
    for name, module in vars(pipeline).items():
        if not isinstance(module, torch.nn.Module):
            continue
        try:
            replace_with_bnb_linear(module, quantization_config=bnb_cfg)
            converted_any = True
            logger.info("offload: 4-bit quantized submodule %s", name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("offload: 4-bit quant failed on %s: %s", name, exc)
    return pipeline if converted_any else None


def _to_cpu(pipeline: Any) -> Any:
    """Move every torch submodule to CPU. Last-resort fallback."""
    try:
        import torch  # type: ignore[import-untyped]
    except ImportError:
        return pipeline
    for _, module in vars(pipeline).items():
        if isinstance(module, torch.nn.Module):
            module.to("cpu")
    logger.info("offload: pipeline moved to CPU")
    return pipeline


def _pick_compute_dtype():
    """bf16 if the GPU supports it, else fp16. CPU path uses fp32."""
    try:
        import torch  # type: ignore[import-untyped]
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            return torch.bfloat16 if major >= 8 else torch.float16
        return torch.float32
    except Exception:  # noqa: BLE001
        import torch  # type: ignore[import-untyped]
        return torch.float16
