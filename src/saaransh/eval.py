"""Retrieval metrics for image-document retrieval.

`qrels` maps query index -> set of relevant doc indices. For ViDoRe-style page
retrieval each query usually has exactly one relevant page, so Recall@1 and MRR
are the headline numbers; nDCG@k generalizes to multi-relevant cases.
"""

from __future__ import annotations

import numpy as np


def recall_at_k(ranked: np.ndarray, qrels: dict[int, set[int]], k: int) -> float:
    hits = 0
    for qi, row in enumerate(ranked):
        rel = qrels.get(qi, set())
        if rel and rel & set(row[:k].tolist()):
            hits += 1
    return hits / max(len(ranked), 1)


def mrr(ranked: np.ndarray, qrels: dict[int, set[int]], k: int = 10) -> float:
    total = 0.0
    for qi, row in enumerate(ranked):
        rel = qrels.get(qi, set())
        for rank, doc in enumerate(row[:k].tolist(), start=1):
            if doc in rel:
                total += 1.0 / rank
                break
    return total / max(len(ranked), 1)


def ndcg_at_k(ranked: np.ndarray, qrels: dict[int, set[int]], k: int) -> float:
    def dcg(gains: list[float]) -> float:
        return sum(g / np.log2(i + 2) for i, g in enumerate(gains))

    total = 0.0
    for qi, row in enumerate(ranked):
        rel = qrels.get(qi, set())
        gains = [1.0 if d in rel else 0.0 for d in row[:k].tolist()]
        ideal = [1.0] * min(len(rel), k)
        idcg = dcg(ideal)
        total += (dcg(gains) / idcg) if idcg > 0 else 0.0
    return total / max(len(ranked), 1)


def evaluate(ranked: np.ndarray, qrels: dict[int, set[int]]) -> dict[str, float]:
    return {
        "recall@1": recall_at_k(ranked, qrels, 1),
        "recall@5": recall_at_k(ranked, qrels, 5),
        "ndcg@5": ndcg_at_k(ranked, qrels, 5),
        "mrr@10": mrr(ranked, qrels, 10),
    }


def evaluate_per_query(ranked: np.ndarray, qrels: dict[int, set[int]], k: int = 5) -> list[dict]:
    """Same signal as evaluate(), but one row per query instead of an aggregate.

    Adds `gold_rank`: 1-indexed position of the first relevant doc in `ranked`,
    or None if it fell outside the retrieved list entirely. This is the field
    that makes failure analysis possible — you can diff gold_rank across arms
    per query instead of only comparing two floats.
    """
    rows: list[dict] = []
    for qi, row in enumerate(ranked):
        rel = qrels.get(qi, set())
        row_list = row.tolist()
        gains = [1.0 if d in rel else 0.0 for d in row_list[:k]]
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k)))
        gold_rank = next((i + 1 for i, d in enumerate(row_list) if d in rel), None)
        rows.append({
            "qi": qi,
            "recall@1": 1.0 if rel and row_list[0] in rel else 0.0,
            "recall@5": 1.0 if rel and (rel & set(row_list[:5])) else 0.0,
            f"ndcg@{k}": (sum(g / np.log2(i + 2) for i, g in enumerate(gains)) / idcg) if idcg > 0 else 0.0,
            "gold_rank": gold_rank,
        })
    return rows