"""Per-query failure analysis: where does MUVERA actually lose to the MaxSim ceiling?

Requires a cache built AFTER the cache.py/cli.py patch (so meta["query_texts"] exists).
Cheap: reuses the cached ColQwen2 bags, no model forward passes.

Usage:
    python scripts/failure_analysis.py --cache ./cache/docvqa200 \
        --mode calibrated_eigenbasis --k 8 --reps 8 --out failure_report.csv
"""

from __future__ import annotations

import argparse
import re

import numpy as np
from pymuvera import MUVERAEncoder, ProjectionType

from saaransh.cache import load_cache, maxsim_score
from saaransh.eval import evaluate_per_query
from saaransh.index import FlatIndex

QUESTION_WORDS = ("what", "how", "why", "who", "which", "where", "when", "does", "is", "are")


def query_bucket(q: str) -> str:
    first = re.sub(r"[^a-z]", "", q.strip().lower().split(" ")[0])
    return first if first in QUESTION_WORDS else "other"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--mode", default="calibrated_eigenbasis")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--compress", type=int, default=None)
    p.add_argument("--out", default="failure_report.csv")
    p.add_argument("--top-n", type=int, default=20, help="worst regressions to print")
    args = p.parse_args()

    doc_bags, query_bags, qrels, meta = load_cache(args.cache)
    query_texts = meta.get("query_texts")
    if query_texts is None:
        raise SystemExit(
            "This cache has no query_texts. Apply the cache.py/cli.py patch and "
            "re-run saaransh-cache before running this script."
        )
    token_dim = int(meta.get("token_dim", 128))
    print(f"cache: {len(doc_bags)} docs, {len(query_bags)} queries, token_dim={token_dim}")

    # --- ceiling (exact MaxSim) ---
    ceil_scores = maxsim_score(query_bags, doc_bags)
    ceil_ranked = np.argsort(-ceil_scores, axis=1)[:, :10]
    ceil_pq = evaluate_per_query(ceil_ranked, qrels)

    # --- one MUVERA config ---
    enc = MUVERAEncoder(
        dimension=token_dim,
        num_simhash_projections=args.k,
        num_repetitions=args.reps,
        projection_type=ProjectionType[args.mode.upper()],
        final_projection_dimension=args.compress,
        fill_empty_partitions=True,
        seed=42,
    )
    if args.mode.upper() == "CALIBRATED_EIGENBASIS":
        enc.calibrate(np.concatenate(doc_bags, axis=0))
    D = np.stack([enc.encode_document(b) for b in doc_bags]).astype("float32")
    Q = np.stack([enc.encode_query(b) for b in query_bags]).astype("float32")
    idx = FlatIndex(metric="ip")
    idx.add(D)
    _, mu_ranked = idx.search(Q, k=10)
    mu_pq = evaluate_per_query(mu_ranked, qrels)

    # --- join ---
    rows = []
    for qi, q in enumerate(query_texts):
        c, m = ceil_pq[qi], mu_pq[qi]
        c_rank = c["gold_rank"] if c["gold_rank"] is not None else 11  # outside top-10
        m_rank = m["gold_rank"] if m["gold_rank"] is not None else 11
        rows.append({
            "qi": qi,
            "query": q,
            "n_words": len(q.split()),
            "bucket": query_bucket(q),
            "ceiling_rank": c["gold_rank"],
            "muvera_rank": m["gold_rank"],
            "rank_delta": m_rank - c_rank,  # >0 = MUVERA did worse than ceiling
            "ceiling_ok_muvera_lost": c_rank == 1 and m_rank > 1,
        })

    with open(args.out, "w") as f:
        f.write("qi,query,n_words,bucket,ceiling_rank,muvera_rank,rank_delta,ceiling_ok_muvera_lost\n")
        for r in rows:
            q_escaped = r["query"].replace('"', "'")
            f.write(
                f'{r["qi"]},"{q_escaped}",{r["n_words"]},{r["bucket"]},'
                f'{r["ceiling_rank"]},{r["muvera_rank"]},{r["rank_delta"]},{r["ceiling_ok_muvera_lost"]}\n'
            )
    print(f"wrote {args.out}  ({len(rows)} queries)")

    # --- summary: does query length or question-word type predict the loss? ---
    lost = [r for r in rows if r["ceiling_ok_muvera_lost"]]
    print(f"\nceiling got it right but MUVERA lost it: {len(lost)}/{len(rows)} queries")

    by_bucket: dict[str, list[int]] = {}
    for r in rows:
        by_bucket.setdefault(r["bucket"], []).append(r["rank_delta"])
    print("\navg rank_delta by first-word bucket (higher = MUVERA relatively worse):")
    for b, deltas in sorted(by_bucket.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  {b:<8} n={len(deltas):<4} avg_delta={np.mean(deltas):+.2f}")

    lengths = np.array([r["n_words"] for r in rows])
    deltas = np.array([r["rank_delta"] for r in rows])
    if len(set(lengths)) > 1:
        corr = np.corrcoef(lengths, deltas)[0, 1]
        print(f"\ncorr(query length, rank_delta) = {corr:+.3f}")

    print(f"\nworst {args.top_n} regressions (ceiling rank vs MUVERA rank):")
    for r in sorted(rows, key=lambda r: -r["rank_delta"])[: args.top_n]:
        print(f"  delta={r['rank_delta']:>3}  ceiling={r['ceiling_rank']!s:>4}  "
              f"muvera={r['muvera_rank']!s:>4}  \"{r['query']}\"")


if __name__ == "__main__":
    main()