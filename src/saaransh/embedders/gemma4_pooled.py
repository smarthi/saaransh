"""Pipeline 2 — Gemma 4 12B as a zero-shot single-vector embedder.

Gemma 4 12B (google/gemma-4-12b-it, released 2026-06-03) is a *generative*,
encoder-free, decoder-only VLM. No vision encoder to tap — the lightweight module
projects raw 48x48 patches straight into the decoder stream. So "embedding" means a
forward pass + pooling the last hidden states into one vector, scored by cosine.

API per the official model card, CONFIRMED on device (transformers 5.10.2, probe_gemma.py):
  - class:   AutoModelForMultimodalLM  (config: Gemma4UnifiedConfig)
  - hidden_size = 3840 at config.text_config.hidden_size
  - forward(**inputs, output_hidden_states=True) -> 49 hidden_states, last = (1, T, 3840)
  - inputs carry pixel_values (1, N_visual, 6912=48*48*3), image_position_ids, mm_token_type_ids
  - input:   processor.apply_chat_template(msgs, tokenize=True, return_dict=True,
             return_tensors="pt", add_generation_prompt=False, enable_thinking=False)
  - image BEFORE text in the message content.

The transformers bf16/fp16 path is verified end-to-end. The MLX (q8/q4) hidden-state
path remains `# VERIFY`. Two known levers, not yet wired (next steps, not bugs):
  - visual_tokens (70/140/280/560/1120): currently only labels the run; the processor
    uses its default (280). Wiring the budget is a one-line probe on the processor.
  - mm_token_type_ids: marks image vs text positions -> enables image-only pooling and
    the ColGemma-lite late-interaction probe.
"""

from __future__ import annotations

import numpy as np

from saaransh.embedders.base import l2_normalize

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _load_multimodal_lm(model_name: str, dtype, device: str):
    """Card's class is AutoModelForMultimodalLM; fall back to ImageTextToText."""
    import transformers

    last_err = None
    for cls_name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            return cls.from_pretrained(model_name, dtype=dtype, device_map=device).eval(), cls_name
        except Exception as e:  # try the next class
            last_err = e
    raise RuntimeError(f"Could not load {model_name} with a known multimodal class: {last_err}")


def _hidden_size(config) -> int:
    for attr in ("hidden_size",):
        if getattr(config, attr, None):
            return int(getattr(config, attr))
    tc = getattr(config, "text_config", None)
    if tc is not None and getattr(tc, "hidden_size", None):
        return int(tc.hidden_size)
    raise AttributeError("could not find hidden_size on the Gemma 4 config; see probe output")


_IMG_PROMPT = "Represent this document image for retrieval."
_QRY_PROMPT = "Represent this query for retrieving a matching document image: {q}"


class Gemma4PooledEmbedder:
    metric = "cosine"

    def __init__(
        self,
        model_name: str = "google/gemma-4-12b-it",
        *,
        backend: str = "transformers",
        precision: str = "bf16",
        pooling: str = "mean",            # "mean" | "last" | "image"
        visual_tokens: int = 280,         # 70|140|280|560|1120
        image_token_type: int = 1,        # mm_token_type_ids value marking image tokens
        device: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.name = f"gemma4-12b[{backend}/{precision}/{pooling}]"
        self.backend = backend
        self.precision = precision
        self.pooling = pooling
        self.visual_tokens = visual_tokens
        self.image_token_type = image_token_type

        if backend == "transformers":
            if torch is None:
                raise ImportError("Install the gemma extra:  uv pip install -e '.[gemma]'")
            if precision not in {"bf16", "fp16"}:
                raise ValueError("transformers backend supports bf16/fp16 only; use backend='mlx' for q8/q4.")
            from transformers import AutoProcessor

            self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
            dtype = torch.bfloat16 if precision == "bf16" else torch.float16
            self.model, self.loaded_class = _load_multimodal_lm(model_name, dtype, self.device)
            self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
            self.dim = _hidden_size(self.model.config)
        elif backend == "mlx":
            self._init_mlx(model_name, precision)
        else:
            raise ValueError(f"unknown backend {backend!r}")

    # ── transformers: per-item forward + pool (batched multimodal chat templates
    #     are fragile, so we process one conversation at a time) ───────────────
    def _embed_one(self, content: list[dict]) -> np.ndarray:
        msgs = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=False, enable_thinking=False,
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[-1][0]          # (T, H)
        return self._pool(hidden, inputs).float().cpu().numpy()

    def _pool(self, hidden, inputs):
        mask = inputs["attention_mask"][0].bool()
        # image-only pooling: average just the image-token positions (kills shared-prompt
        # contamination). Falls back to masked mean when there are no image tokens (queries).
        if self.pooling == "image" and "mm_token_type_ids" in inputs:
            is_img = (inputs["mm_token_type_ids"][0] == self.image_token_type) & mask
            if bool(is_img.any()):
                return hidden[is_img].mean(dim=0)
        h = hidden[mask]
        return h[-1] if self.pooling == "last" else h.mean(dim=0)

    def embed_images(self, images: list) -> np.ndarray:
        if self.backend == "mlx":
            return l2_normalize(self._mlx_embed_images(images))
        vecs = [self._embed_one([{"type": "image", "image": im}, {"type": "text", "text": _IMG_PROMPT}])
                for im in images]
        return l2_normalize(np.stack(vecs))

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        if self.backend == "mlx":
            return l2_normalize(self._mlx_embed_queries(queries))
        vecs = [self._embed_one([{"type": "text", "text": _QRY_PROMPT.format(q=q)}]) for q in queries]
        return l2_normalize(np.stack(vecs))

    # ── MLX quantized backend (Apple Silicon) ────────────────────────────
    def _init_mlx(self, model_name: str, precision: str) -> None:
        try:
            from mlx_vlm import load as _load  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise ImportError("Install the mlx extra on a Mac:  uv pip install -e '.[mlx]'") from e
        bits = 8 if precision == "q8" else 4
        # mlx-community converts from the base; naming is gemma-4-12B-{bits}bit.
        repo = model_name if "mlx" in model_name else f"mlx-community/gemma-4-12B-{bits}bit"
        from mlx_vlm import load as _load2
        self._mlx_model, self._mlx_processor = _load2(repo)
        self.dim = int(getattr(self._mlx_model.config, "hidden_size", 0)) or 3840

    def _mlx_hidden(self, image, prompt: str) -> np.ndarray:
        # >>> VERIFY ON DEVICE (probe_gemma.py --backend mlx) <<<
        # mlx-vlm's generate path doesn't surface hidden states; call the underlying
        # model forward and capture the last hidden state. Wire against your installed
        # mlx-vlm version once the transformers baseline is working.
        raise NotImplementedError(
            "MLX hidden-state extraction must be wired against your mlx-vlm version. "
            "Use backend='transformers' (bf16/fp16) for the quality baseline first."
        )

    def _mlx_embed_images(self, images: list) -> np.ndarray:
        return np.stack([self._mlx_hidden(im, _IMG_PROMPT) for im in images]).astype("float32")

    def _mlx_embed_queries(self, queries: list[str]) -> np.ndarray:
        return np.stack([self._mlx_hidden(None, _QRY_PROMPT.format(q=q)) for q in queries]).astype("float32")
