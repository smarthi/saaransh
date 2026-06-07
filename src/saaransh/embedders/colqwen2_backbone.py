"""Shared ColQwen2 backbone loader — transformers-native.

Uses transformers' native ColQwen2ForRetrieval + ColQwen2Processor on the *merged*
HF checkpoint (vidore/colqwen2-v1.0-hf). This avoids colpali-engine entirely:
  - merged weights -> no LoRA adapter step (the peft path that breaks on transformers 5.x)
  - native classes track transformers -> no version-lag key-mapping errors
  - same transformers 5.x that Gemma 4 needs -> one environment for all three arms

Keeps the Apple-Silicon env flags from the user's proven setup.
"""

from __future__ import annotations

import os


def configure_mac_env() -> None:
    """MPS/tokenizer flags for Apple Silicon (setdefault so a shell value wins)."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


def pick_device(prefer: str | None = None):
    import torch

    if prefer:
        return prefer, (torch.bfloat16 if prefer in ("mps", "cuda") else torch.float32)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def _resolve_local_model(model_name: str, cache_dir: str | None) -> tuple[str, str | None]:
    """Forgiving path handling: if cache_dir is actually a downloaded model folder
    (contains config.json), use it as the model path. Catches the common mix-up of
    passing an `hf download --local-dir` folder to --cache-dir instead of --colqwen-model.
    """
    if cache_dir:
        cd = os.path.expanduser(cache_dir)
        if os.path.isfile(os.path.join(cd, "config.json")):
            return cd, None
    return model_name, cache_dir


def load_colqwen2(
    model_name: str = "vidore/colqwen2-v1.0-hf",
    *,
    device: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
):
    """Return (model.eval(), processor, device, dtype_str) via transformers-native classes."""
    configure_mac_env()
    import torch  # noqa: F401
    from transformers import ColQwen2ForRetrieval, ColQwen2Processor

    resolved, cache_dir = _resolve_local_model(model_name, cache_dir)
    if resolved != model_name:
        print(f"[saaransh] using model folder from --cache-dir: {resolved}")
        model_name = resolved

    dev, dtype = pick_device(device)
    kw: dict = {}
    if cache_dir:
        kw["cache_dir"] = cache_dir
    if local_files_only:
        kw["local_files_only"] = True

    model = ColQwen2ForRetrieval.from_pretrained(
        model_name, dtype=dtype, device_map=dev, attn_implementation="sdpa", **kw
    ).eval()
    processor = ColQwen2Processor.from_pretrained(model_name, **kw)
    return model, processor, dev, str(dtype).replace("torch.", "")
