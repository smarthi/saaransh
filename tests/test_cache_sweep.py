"""Cache round-trip + MUVERA sweep + exact-MaxSim ceiling, on synthetic bags (no model)."""

from __future__ import annotations

import numpy as np

from saaransh.cache import load_cache, maxsim_score, save_cache
from saaransh.sweep import ceiling_row, grid, run_muvera_sweep


def _synthetic(n=16, token_dim=128, seed=5):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n, token_dim)).astype("float32") * 6.0
    doc_bags = [
        (centers[i] + rng.standard_normal((rng.integers(200, 400), token_dim)).astype("float32"))
        for i in range(n)
    ]
    query_bags = [doc_bags[i][:6] for i in range(n)]  # exact sub-bags -> planted answers
    qrels = {i: {i} for i in range(n)}
    return doc_bags, query_bags, qrels


def test_cache_roundtrip(tmp_path):
    doc_bags, query_bags, qrels = _synthetic()
    path = str(tmp_path / "bags")
    save_cache(path, doc_bags, query_bags, qrels, {"token_dim": 128, "n_docs": len(doc_bags)})
    d2, q2, qr2, meta = load_cache(path)
    assert len(d2) == len(doc_bags) and len(q2) == len(query_bags)
    assert qr2 == qrels and meta["token_dim"] == 128
    assert d2[0].shape[1] == 128


def test_maxsim_ceiling_planted():
    doc_bags, query_bags, qrels = _synthetic()
    row = ceiling_row(query_bags, doc_bags, qrels)
    assert row["name"].startswith("colqwen2-maxsim")
    assert row["recall@1"] == 1.0  # query is an exact sub-bag of its own doc
    s = maxsim_score(query_bags, doc_bags)
    assert s.shape == (len(query_bags), len(doc_bags))


def test_muvera_sweep_rows():
    doc_bags, query_bags, qrels = _synthetic()
    cfgs = grid(["default_identity", "calibrated_eigenbasis"], [4], [4], [None, 1024])
    rows = run_muvera_sweep(doc_bags, query_bags, qrels, cfgs, token_dim=128)
    assert len(rows) == 4
    for r in rows:
        assert 0.0 <= r["ndcg@5"] <= 1.0
        assert r["recall@5"] >= 0.8  # planted self-retrieval should be easy
    # compressed config reports the compressed dim
    compressed = [r for r in rows if "c1024" in r["name"]]
    assert all(r["fde_dim"] == 1024 for r in compressed)


def test_pareto_frontier_envelope():
    from saaransh.sweep import pareto_frontier

    rows = [
        {"name": "a", "bytes_per_doc": 8192, "ndcg@5": 0.35},
        {"name": "b", "bytes_per_doc": 8192, "ndcg@5": 0.10},   # dominated (same bytes, worse)
        {"name": "c", "bytes_per_doc": 32768, "ndcg@5": 0.43},
        {"name": "d", "bytes_per_doc": 65536, "ndcg@5": 0.40},  # dominated (more bytes, worse than c)
        {"name": "e", "bytes_per_doc": 131072, "ndcg@5": 0.48},
    ]
    front = [p["name"] for p in pareto_frontier(rows)]
    assert front == ["a", "c", "e"]
