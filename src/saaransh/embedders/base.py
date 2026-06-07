"""Single-vector embedder interface shared by both pipelines.

Both pipelines reduce a document (image) or a query (text) to **one** dense
vector, so they drop into an identical flat index and eval harness. They differ
only in (a) how the vector is produced and (b) the similarity metric:

  - ColQwen2 + MUVERA  -> FDE vector, scored by **inner product** (Chamfer proxy)
  - Gemma 4 12B pooled -> pooled hidden state, scored by **cosine**

Keeping the metric on the embedder (not the index) is deliberate: MUVERA FDEs
must NOT be L2-normalized or the SUM/AVG asymmetry that approximates MaxSim is
destroyed. The index reads `embedder.metric` and does the right thing.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np

Metric = Literal["ip", "cosine"]


@runtime_checkable
class SingleVectorEmbedder(Protocol):
    """Anything that turns images/queries into one (N, dim) float32 matrix."""

    name: str
    dim: int
    metric: Metric

    def embed_images(self, images: list) -> np.ndarray:  # (N, dim) float32
        ...

    def embed_queries(self, queries: list[str]) -> np.ndarray:  # (M, dim) float32
        ...


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalize (for cosine-metric embedders only)."""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return (x / np.maximum(norm, eps)).astype("float32")


def unpad_multivectors(
    embeddings: np.ndarray,
    attention_mask: np.ndarray,
) -> list[np.ndarray]:
    """Convert a padded (B, T, d) batch + (B, T) mask into a list of (n_i, d) bags.

    colpali-engine returns right-padded tensors; feeding the pad tokens into
    MUVERA would pollute the FDE, so we slice each item to its true token count.
    """
    out: list[np.ndarray] = []
    for emb, mask in zip(embeddings, attention_mask, strict=True):
        keep = np.asarray(mask).astype(bool)
        out.append(np.asarray(emb)[keep].astype("float32"))
    return out
