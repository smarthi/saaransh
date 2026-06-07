"""Retrieval indexes.

Both pipelines emit one vector per document, so the index is a thin layer over a
similarity search. Two backends share one interface (`add` / `search` / `stats`):

  - FlatIndex   — numpy, zero-dependency, exact. Default; keeps the core install
                  free of faiss. Optional int8 scalar quant of stored vectors.
  - FaissIndex  — faiss-cpu backend: exact Flat, IVF+PQ, or HNSW. The IVF+PQ path
                  mirrors a production ANN store (cf. OpenSearch IVF+PQ:
                  nlist=4096, nprobe=128, m=16 for a ColQwen2 corpus at scale).

Metric handling is uniform: both "ip" and "cosine" reduce to **inner product** at
the index. The embedder owns normalization — cosine embedders pre-normalize, MUVERA
FDEs are left raw so the dot product still approximates Chamfer/MaxSim. So faiss uses
METRIC_INNER_PRODUCT throughout and `metric` is only a label here.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

Metric = Literal["ip", "cosine"]


class IndexStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_docs: int
    dim: int
    bytes_per_doc: float
    metric: str
    backend: str


class IndexConfig(BaseModel):
    """How to build the index. Validated up front so bad ANN params fail fast."""

    model_config = ConfigDict(frozen=True)

    backend: Literal["numpy", "faiss"] = "numpy"
    kind: Literal["flat", "ivfpq", "hnsw"] = "flat"
    quantize_int8: bool = False  # numpy backend only: symmetric int8 scalar quant
    # IVF+PQ
    nlist: int = 256
    nprobe: int = 32
    pq_m: int = 16
    pq_nbits: int = 8
    # HNSW
    hnsw_m: int = 32
    ef_construction: int = 200
    ef_search: int = 64


# ── numpy exact index (zero-dep default) ─────────────────────────────────
class FlatIndex:
    backend = "numpy"

    def __init__(self, metric: Metric = "cosine", quantize_int8: bool = False) -> None:
        assert metric in ("ip", "cosine")
        self.metric = metric
        self.quantize_int8 = quantize_int8
        self._docs: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    def add(self, doc_vectors: np.ndarray) -> None:
        v = np.ascontiguousarray(doc_vectors, dtype="float32")
        if self.quantize_int8:
            self._scale = np.maximum(np.abs(v).max(axis=1, keepdims=True), 1e-8) / 127.0
            self._docs = np.round(v / self._scale).astype("int8")
        else:
            self._docs = v

    def _docs_f32(self) -> np.ndarray:
        return self._docs.astype("float32") * self._scale if self.quantize_int8 else self._docs

    def search(self, query_vectors: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(query_vectors, dtype="float32")
        d = self._docs_f32()
        scores = q @ d.T
        k = min(k, d.shape[0])
        idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        return np.take_along_axis(scores, idx, axis=1), idx

    def stats(self) -> IndexStats:
        d = self._docs
        bpd = d.dtype.itemsize * d.shape[1] + (4 if self.quantize_int8 else 0)
        return IndexStats(
            n_docs=d.shape[0], dim=d.shape[1], bytes_per_doc=float(bpd),
            metric=self.metric, backend=self.backend,
        )


# ── faiss backend (flat / ivfpq / hnsw) ──────────────────────────────────
class FaissIndex:
    backend = "faiss"

    def __init__(self, metric: Metric = "cosine", config: IndexConfig | None = None) -> None:
        import faiss  # noqa: F401  (import-time check)

        self.metric = metric
        self.cfg = config or IndexConfig(backend="faiss")
        self._faiss = faiss
        self._index = None
        self._dim = 0
        self._n = 0
        self._eff_nlist = 0

    def add(self, doc_vectors: np.ndarray) -> None:
        faiss = self._faiss
        v = np.ascontiguousarray(doc_vectors, dtype="float32")
        self._n, self._dim = v.shape
        d = self._dim
        kind = self.cfg.kind

        if kind == "flat":
            index = faiss.IndexFlatIP(d)
        elif kind == "hnsw":
            index = faiss.IndexHNSWFlat(d, self.cfg.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = self.cfg.ef_construction
            index.hnsw.efSearch = self.cfg.ef_search
        elif kind == "ivfpq":
            if d % self.cfg.pq_m != 0:
                raise ValueError(
                    f"pq_m={self.cfg.pq_m} must divide the FDE dimension {d}. "
                    f"Set --fde-compress to a multiple of pq_m (e.g. {self.cfg.pq_m * 64})."
                )
            # FAISS wants >= ~39 training points per centroid; clamp nlist to corpus size.
            self._eff_nlist = max(1, min(self.cfg.nlist, self._n // 39 or 1))
            quant = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFPQ(
                quant, d, self._eff_nlist, self.cfg.pq_m, self.cfg.pq_nbits,
                faiss.METRIC_INNER_PRODUCT,
            )
            index.train(v)
            index.nprobe = min(self.cfg.nprobe, self._eff_nlist)
        else:  # pragma: no cover
            raise ValueError(f"unknown faiss index kind {kind!r}")

        if not index.is_trained:
            index.train(v)
        index.add(v)
        self._index = index

    def search(self, query_vectors: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(query_vectors, dtype="float32")
        scores, idx = self._index.search(q, min(k, self._n))
        return scores, idx

    def stats(self) -> IndexStats:
        if self.cfg.kind == "ivfpq":
            # PQ code size; ignores coarse-quantizer + codebook overhead amortized over N.
            bpd = float(self.cfg.pq_m * (self.cfg.pq_nbits / 8.0))
        else:
            bpd = float(4 * self._dim)
        label = f"{self.backend}:{self.cfg.kind}"
        return IndexStats(
            n_docs=self._n, dim=self._dim, bytes_per_doc=bpd, metric=self.metric, backend=label,
        )


def make_index(metric: Metric, config: IndexConfig | None = None):
    """Factory: numpy FlatIndex or a faiss backend, per `config.backend`."""
    cfg = config or IndexConfig()
    if cfg.backend == "faiss":
        return FaissIndex(metric, cfg)
    return FlatIndex(metric, quantize_int8=cfg.quantize_int8)
