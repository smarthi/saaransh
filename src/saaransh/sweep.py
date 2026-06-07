"""Sweep MUVERA configs over cached ColQwen2 bags — the storage/quality frontier.

Each config is cheap (FDE encode + flat index + eval) since the model forward is cached.
Also computes the exact-MaxSim ceiling from the same bags so the curve has its reference.
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np
from pymuvera import MUVERAEncoder, ProjectionType

from saaransh.cache import maxsim_score
from saaransh.eval import evaluate
from saaransh.index import FlatIndex


def ceiling_row(query_bags, doc_bags, qrels) -> dict:
    t = time.perf_counter()
    scores = maxsim_score(query_bags, doc_bags)
    ranked = np.argsort(-scores, axis=1)[:, :10]
    dt = time.perf_counter() - t
    avg_patches = float(np.mean([len(d) for d in doc_bags]))
    return {
        "name": "colqwen2-maxsim (ceiling)",
        "fde_dim": doc_bags[0].shape[1],
        "bytes_per_doc": avg_patches * doc_bags[0].shape[1] * 4.0,
        "sweep_s": dt,
        **evaluate(ranked, qrels),
    }


def run_muvera_sweep(doc_bags, query_bags, qrels, configs, token_dim: int = 128) -> list[dict]:
    rows: list[dict] = []
    calib = None
    for cfg in configs:
        mode = cfg["mode"].upper()
        enc = MUVERAEncoder(
            dimension=token_dim,
            num_simhash_projections=cfg["k"],
            num_repetitions=cfg["reps"],
            projection_type=ProjectionType[mode],
            final_projection_dimension=cfg.get("fde_compress"),
            fill_empty_partitions=True,
            seed=42,
        )
        if mode == "CALIBRATED_EIGENBASIS":
            if calib is None:
                calib = np.concatenate(doc_bags, axis=0)
            enc.calibrate(calib)

        t = time.perf_counter()
        D = np.stack([enc.encode_document(b) for b in doc_bags]).astype("float32")
        Q = np.stack([enc.encode_query(b) for b in query_bags]).astype("float32")
        idx = FlatIndex(metric="ip")
        idx.add(D)
        _, ranked = idx.search(Q, k=10)
        dt = time.perf_counter() - t

        c = cfg.get("fde_compress")
        rows.append({
            "name": f"muvera[{mode[:2]}|k{cfg['k']}|r{cfg['reps']}|c{c or '-'}]",
            "fde_dim": enc.fde_dimension,
            "bytes_per_doc": float(enc.fde_dimension * 4),
            "sweep_s": dt,
            **evaluate(ranked, qrels),
        })
    return rows


def grid(modes: list[str], ks: list[int], reps: list[int], compress: list[int | None]) -> list[dict]:
    return [
        {"mode": m, "k": k, "reps": r, "fde_compress": c}
        for m, k, r, c in product(modes, ks, reps, compress)
    ]


def format_sweep(rows: list[dict]) -> str:
    cols = ["recall@1", "recall@5", "ndcg@5", "mrr@10"]
    head = f"{'config':<34}{'fde_dim':>9}{'B/doc':>10}" + "".join(f"{c:>10}" for c in cols) + f"{'s':>7}"
    out = [head, "-" * len(head)]
    for r in rows:
        line = f"{r['name']:<34}{r['fde_dim']:>9}{r['bytes_per_doc']:>10.0f}"
        line += "".join(f"{r[c]:>10.3f}" for c in cols) + f"{r['sweep_s']:>7.1f}"
        out.append(line)
    return "\n".join(out)


def pareto_frontier(rows: list[dict]) -> list[dict]:
    """Upper-left staircase: best nDCG@5 achievable at or below each storage budget."""
    pts = sorted(rows, key=lambda r: (r["bytes_per_doc"], -r["ndcg@5"]))
    out: list[dict] = []
    best = -1.0
    for p in pts:
        if p["ndcg@5"] > best:
            out.append(p)
            best = p["ndcg@5"]
    return out


def plot_frontier(rows: list[dict], out_path: str, ceiling: dict | None = None) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))
    # faded scatter of every config, colored by mode prefix
    modes = {
        "muvera[DE": ("#4C78A8", "DEFAULT_IDENTITY"),
        "muvera[CA": ("#F58518", "CALIBRATED_EIGENBASIS"),
    }
    for prefix, (col, label) in modes.items():
        pts = [r for r in rows if r["name"].startswith(prefix)]
        if pts:
            ax.scatter([p["bytes_per_doc"] for p in pts], [p["ndcg@5"] for p in pts],
                       s=22, color=col, alpha=0.35, label=label)
    # bold Pareto envelope
    front = pareto_frontier(rows)
    ax.plot([p["bytes_per_doc"] for p in front], [p["ndcg@5"] for p in front],
            "o-", color="black", lw=2, ms=5, label="Pareto frontier")
    if ceiling:
        ax.axhline(ceiling["ndcg@5"], ls="--", color="gray",
                   label=f"MaxSim ceiling ({ceiling['ndcg@5']:.3f})")
        ax.axvline(ceiling["bytes_per_doc"], ls=":", color="gray", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("bytes / doc (log)")
    ax.set_ylabel("nDCG@5")
    ax.set_title("MUVERA storage vs quality frontier")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
