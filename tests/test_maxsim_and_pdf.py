"""MaxSim baseline glue + PDF rendering. PDF test skips if PyMuPDF is absent."""

from __future__ import annotations

import numpy as np
import pytest

from saaransh.runner import build_and_eval_maxsim


class MockMaxSimRetriever:
    """Implements the retriever interface with a planted score matrix."""

    name = "mock-maxsim"
    token_dim = 128

    def __init__(self, n=12):
        self.n = n
        # doc i is the gold page for query i
        self._scores = np.eye(n, dtype="float32") + 0.01 * np.random.default_rng(0).random((n, n))
        self._patches = [400 + i for i in range(n)]

    def embed_images(self, images):
        return [np.zeros((self._patches[i], self.token_dim), dtype="float32") for i in range(self.n)]

    def embed_queries(self, queries):
        return list(range(len(queries)))

    def rank(self, q_mv, doc_mv, k):
        idx = np.argsort(-self._scores, axis=1)[:, :k]
        return idx

    @staticmethod
    def bytes_per_doc(doc_mv, token_dim):
        avg = float(np.mean([len(d) for d in doc_mv]))
        return avg * token_dim * 4.0


def test_maxsim_runner_glue():
    n = 12
    r = MockMaxSimRetriever(n=n)
    res = build_and_eval_maxsim(r, list(range(n)), [f"q{i}" for i in range(n)], {i: {i} for i in range(n)})
    assert res.backend == "maxsim"
    assert res.metrics["recall@1"] == 1.0
    assert res.bytes_per_doc > 400 * 128 * 4  # fat multi-vector storage


def test_maxsim_rank_argsort():
    from saaransh.embedders.colqwen2_maxsim import ColQwen2MaxSimRetriever

    inst = object.__new__(ColQwen2MaxSimRetriever)  # bypass model load
    inst._score = lambda q, d: np.array([[0.1, 0.9, 0.3], [0.5, 0.2, 0.8]], dtype="float32")
    ranked = inst.rank(query_mv=[0, 1], doc_mv=[0, 1, 2], k=3)
    assert ranked[0].tolist() == [1, 2, 0]
    assert ranked[1].tolist() == [2, 0, 1]


def test_pdf_render():
    fitz = pytest.importorskip("fitz")
    from saaransh.corpus import render_document

    path = "/tmp/_saaransh_test.pdf"
    doc = fitz.open()
    for _ in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), "saaransh test page")
    doc.save(path)
    doc.close()

    pages = render_document(path, dpi=100)
    assert len(pages) == 3
    assert all(hasattr(p, "size") for p in pages)  # PIL images
