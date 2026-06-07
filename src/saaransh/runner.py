"""Glue: take an embedder + corpus, build the index, run queries, time it, score it."""

from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict

from saaransh.embedders.base import SingleVectorEmbedder
from saaransh.eval import evaluate
from saaransh.index import IndexConfig, make_index


class RunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    backend: str
    metrics: dict[str, float]
    dim: int
    bytes_per_doc: float
    index_s: float
    query_s: float


def build_and_eval(
    embedder: SingleVectorEmbedder,
    images: list,
    queries: list[str],
    qrels: dict[int, set[int]],
    *,
    index_config: IndexConfig | None = None,
    k: int = 10,
) -> RunResult:
    t0 = time.perf_counter()
    doc_vecs = embedder.embed_images(images)
    idx = make_index(embedder.metric, index_config)
    idx.add(doc_vecs)
    index_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    q_vecs = embedder.embed_queries(queries)
    _, ranked = idx.search(q_vecs, k=k)
    query_s = time.perf_counter() - t1

    st = idx.stats()
    return RunResult(
        name=embedder.name, backend=st.backend, metrics=evaluate(ranked, qrels),
        dim=st.dim, bytes_per_doc=st.bytes_per_doc, index_s=index_s, query_s=query_s,
    )


def build_and_eval_maxsim(
    retriever,
    images: list,
    queries: list[str],
    qrels: dict[int, set[int]],
    *,
    k: int = 10,
) -> RunResult:
    """Eval the exact-MaxSim baseline (multi-vector, no flat index)."""
    import time as _t

    t0 = _t.perf_counter()
    doc_mv = retriever.embed_images(images)
    index_s = _t.perf_counter() - t0

    t1 = _t.perf_counter()
    q_mv = retriever.embed_queries(queries)
    ranked = retriever.rank(q_mv, doc_mv, k=k)
    query_s = _t.perf_counter() - t1

    return RunResult(
        name=retriever.name,
        backend="maxsim",
        metrics=evaluate(ranked, qrels),
        dim=retriever.token_dim,
        bytes_per_doc=retriever.bytes_per_doc(doc_mv, retriever.token_dim),
        index_s=index_s,
        query_s=query_s,
    )


def format_table(results: list[RunResult]) -> str:
    cols = ["recall@1", "recall@5", "ndcg@5", "mrr@10"]
    head = (
        f"{'pipeline':<34}{'index':>13}{'dim':>7}{'B/doc':>9}"
        + "".join(f"{c:>10}" for c in cols)
        + f"{'idx s':>8}{'qry s':>8}"
    )
    lines = [head, "-" * len(head)]
    for r in results:
        row = f"{r.name:<34}{r.backend:>13}{r.dim:>7}{r.bytes_per_doc:>9.0f}"
        row += "".join(f"{r.metrics[c]:>10.3f}" for c in cols)
        row += f"{r.index_s:>8.1f}{r.query_s:>8.1f}"
        lines.append(row)
    return "\n".join(lines)
