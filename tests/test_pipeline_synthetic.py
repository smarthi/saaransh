"""Synthetic end-to-end test of the model-agnostic glue.

No torch / no model downloads. The ColQwen2-side mock feeds random multi-vector
bags through the REAL pymuvera encoder, so MUVERA scoring, the index, and the
metrics are all exercised for real. The model forward passes themselves are
verified on-device (Mac), not here.
"""

from __future__ import annotations

import numpy as np

from saaransh.embedders.base import l2_normalize
from saaransh.index import FlatIndex
from saaransh.runner import build_and_eval
from pymuvera import MUVERAEncoder


# ── deterministic harness check: planted answers must rank #1 ────────────
def test_index_eval_planted_cosine():
    rng = np.random.default_rng(0)
    docs = l2_normalize(rng.standard_normal((20, 64)).astype("float32"))
    # query i points exactly at doc (i+3) % 20
    targets = [(i + 3) % 20 for i in range(5)]
    queries = docs[targets].copy()
    idx = FlatIndex(metric="cosine")
    idx.add(docs)
    _, ranked = idx.search(queries, k=10)
    qrels = {i: {t} for i, t in enumerate(targets)}
    from saaransh.eval import evaluate

    m = evaluate(ranked, qrels)
    assert m["recall@1"] == 1.0
    assert m["mrr@10"] == 1.0


def test_index_int8_preserves_top1():
    rng = np.random.default_rng(1)
    docs = l2_normalize(rng.standard_normal((50, 128)).astype("float32"))
    q = docs[[7, 13, 41]].copy()
    a = FlatIndex(metric="cosine")
    a.add(docs)
    b = FlatIndex(metric="cosine", quantize_int8=True)
    b.add(docs)
    _, ra = a.search(q, k=1)
    _, rb = b.search(q, k=1)
    assert (ra[:, 0] == rb[:, 0]).all()  # int8 keeps the planted top-1


# ── real pymuvera integration via a mock ColQwen2 ────────────────────────
class MockColQwen2Muvera:
    metric = "ip"
    name = "mock-colqwen2+muvera"

    def __init__(self, n_docs=24, token_dim=128, seed=7):
        self.token_dim = token_dim
        self.enc = MUVERAEncoder(dimension=token_dim, num_simhash_projections=4, num_repetitions=4,
                                 fill_empty_partitions=True, seed=seed)
        self.dim = self.enc.fde_dimension
        rng = np.random.default_rng(seed)
        # each doc = a well-separated cluster of patch vectors
        centers = rng.standard_normal((n_docs, token_dim)).astype("float32") * 5.0
        self._doc_bags = [
            (centers[i] + rng.standard_normal((rng.integers(300, 900), token_dim)).astype("float32"))
            for i in range(n_docs)
        ]

    def embed_images(self, images):
        return np.stack([self.enc.encode_document(b) for b in self._doc_bags]).astype("float32")

    def embed_queries(self, queries):
        # query i = a few patches lifted straight from doc i -> exact-match Chamfer
        bags = [self._doc_bags[i][:8] for i in range(len(queries))]
        return np.stack([self.enc.encode_query(b) for b in bags]).astype("float32")


def test_muvera_pipeline_self_retrieval():
    n = 24
    emb = MockColQwen2Muvera(n_docs=n)
    images = list(range(n))
    queries = [f"q{i}" for i in range(n)]
    qrels = {i: {i} for i in range(n)}
    r = build_and_eval(emb, images, queries, qrels)
    assert r.dim == emb.enc.fde_dimension
    assert set(r.metrics) == {"recall@1", "recall@5", "ndcg@5", "mrr@10"}
    # queries are literal sub-bags of their own doc -> MUVERA should nail top-1
    assert r.metrics["recall@1"] >= 0.9
    assert r.metrics["recall@5"] == 1.0


class MockGemmaPooled:
    metric = "cosine"
    name = "mock-gemma4-pooled"

    def __init__(self, n_docs=24, dim=512, seed=3):
        self.dim = dim
        rng = np.random.default_rng(seed)
        self._docs = l2_normalize(rng.standard_normal((n_docs, dim)).astype("float32"))

    def embed_images(self, images):
        return self._docs

    def embed_queries(self, queries):
        return self._docs[: len(queries)].copy()  # query i aligned to doc i


def test_gemma_pipeline_self_retrieval():
    n = 24
    emb = MockGemmaPooled(n_docs=n)
    r = build_and_eval(emb, list(range(n)), [f"q{i}" for i in range(n)], {i: {i} for i in range(n)})
    assert r.metrics["recall@1"] == 1.0
    assert r.bytes_per_doc == 512 * 4
