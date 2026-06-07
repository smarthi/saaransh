"""Cache ColQwen2 multivectors once, then sweep cheap derivations over them.

The expensive part of every ColQwen2 run is the forward pass (~1.3 s/page on MPS).
The raw patch/token bags it produces are config-independent, so we extract them once,
persist them, and then (a) sweep any number of MUVERA configs in seconds each and
(b) recompute the exact-MaxSim ceiling without reloading the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def extract_colqwen2_bags(
    images: list,
    queries: list[str],
    *,
    model_name: str,
    token_dim: int = 128,
    device: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    batch_size: int = 4,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run ColQwen2 once, return (doc_bags, query_bags) as lists of (n_i, token_dim) arrays."""
    import torch

    from saaransh.embedders.base import unpad_multivectors
    from saaransh.embedders.colqwen2_backbone import load_colqwen2

    model, processor, dev, _ = load_colqwen2(
        model_name, device=device, cache_dir=cache_dir, local_files_only=local_files_only
    )

    def run(items: list, kind: str) -> list[np.ndarray]:
        bags: list[np.ndarray] = []
        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            if kind == "image":
                inputs = processor(images=chunk, return_tensors="pt").to(dev)
            else:
                inputs = processor(text=list(chunk), return_tensors="pt").to(dev)
            with torch.no_grad():
                emb = model(**inputs).embeddings
            if emb.shape[-1] != token_dim:
                raise ValueError(f"ColQwen token dim {emb.shape[-1]} != {token_dim}")
            bags += unpad_multivectors(
                emb.float().cpu().numpy(), inputs["attention_mask"].cpu().numpy()
            )
        return bags

    return run(images, "image"), run(queries, "query")


def save_cache(path: str | Path, doc_bags, query_bags, qrels: dict[int, set[int]], meta: dict) -> None:
    path = str(path)
    np.savez(
        path,
        doc=np.array(doc_bags, dtype=object),
        query=np.array(query_bags, dtype=object),
    )
    side = dict(meta)
    side["qrels"] = {str(k): sorted(v) for k, v in qrels.items()}
    Path(path + ".meta.json").write_text(json.dumps(side))


def load_cache(path: str | Path):
    path = str(path)
    npz_path = path if path.endswith(".npz") else path + ".npz"
    data = np.load(npz_path, allow_pickle=True)
    doc_bags = list(data["doc"])
    query_bags = list(data["query"])
    meta = json.loads(Path(npz_path.replace(".npz", "") + ".meta.json").read_text())
    qrels = {int(k): set(v) for k, v in meta.pop("qrels").items()}
    return doc_bags, query_bags, qrels, meta


def maxsim_score(query_bags: list[np.ndarray], doc_bags: list[np.ndarray]) -> np.ndarray:
    """Exact ColBERT MaxSim from cached bags: sum_i max_j (q_i . d_j), per-token L2-normalized.
    Returns (n_queries, n_docs). Should reproduce processor.score_retrieval up to normalization.
    """
    def norm(b):
        return b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)

    qn = [norm(q.astype("float32")) for q in query_bags]
    dn = [norm(d.astype("float32")) for d in doc_bags]
    scores = np.zeros((len(qn), len(dn)), dtype="float32")
    for i, q in enumerate(qn):
        for j, d in enumerate(dn):
            scores[i, j] = (q @ d.T).max(axis=1).sum()
    return scores
