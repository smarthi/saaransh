"""FAISS backend tests. Skipped automatically if faiss isn't installed."""

from __future__ import annotations

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from saaransh.embedders.base import l2_normalize  # noqa: E402
from saaransh.index import FaissIndex, FlatIndex, IndexConfig, make_index  # noqa: E402


def _planted(n=60, d=128, n_q=10, seed=0):
    rng = np.random.default_rng(seed)
    docs = l2_normalize(rng.standard_normal((n, d)).astype("float32"))
    targets = [(i * 7 + 3) % n for i in range(n_q)]
    queries = docs[targets].copy()
    return docs, queries, targets


def test_faiss_flat_matches_numpy_top1():
    docs, queries, targets = _planted()
    npx = FlatIndex(metric="cosine")
    npx.add(docs)
    fx = FaissIndex(metric="cosine", config=IndexConfig(backend="faiss", kind="flat"))
    fx.add(docs)
    _, n_idx = npx.search(queries, k=1)
    _, f_idx = fx.search(queries, k=1)
    assert (n_idx[:, 0] == np.array(targets)).all()
    assert (f_idx[:, 0] == n_idx[:, 0]).all()


def test_make_index_dispatch():
    assert make_index("cosine", IndexConfig(backend="numpy")).backend == "numpy"
    assert make_index("ip", IndexConfig(backend="faiss", kind="flat")).backend == "faiss"


def test_ivfpq_smoke_recall():
    # PQ needs enough training points (>= 2^nbits); use a corpus that supports it.
    rng = np.random.default_rng(1)
    docs = l2_normalize(rng.standard_normal((2000, 256)).astype("float32"))
    q = docs[:50].copy()  # self-query
    fx = FaissIndex(
        metric="cosine",
        config=IndexConfig(backend="faiss", kind="ivfpq", nlist=64, nprobe=16, pq_m=16, pq_nbits=8),
    )
    fx.add(docs)
    _, idx = fx.search(q, k=10)
    hits = sum(i in idx[r].tolist() for r, i in enumerate(range(50)))
    assert hits / 50 >= 0.8  # PQ is lossy; self should still be in top-10 most of the time
    st = fx.stats()
    assert st.bytes_per_doc == 16  # pq_m=16 * 8 bits / 8


def test_ivfpq_rejects_indivisible_dim():
    fx = FaissIndex(metric="ip", config=IndexConfig(backend="faiss", kind="ivfpq", pq_m=15))
    with pytest.raises(ValueError, match="must divide"):
        fx.add(np.random.randn(500, 128).astype("float32"))
