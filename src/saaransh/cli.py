"""Command-line entrypoints.

  saaransh-demo  --corpus ./sample_docs
  saaransh-demo  --vidore vidore/docvqa_test_subsampled --limit 200
  saaransh-demo  --vidore ... --index ivfpq --pq-m 16 --fde-compress 1024
  saaransh-sweep --vidore ... --precisions bf16,q8,q4

Index backends: flat (numpy, default), faiss-flat, ivfpq, hnsw. ivfpq/hnsw need
the faiss extra:  uv pip install -e '.[faiss]'.
"""

from __future__ import annotations

import argparse
import os

from pymuvera import ProjectionType

from saaransh.index import IndexConfig
from saaransh.runner import build_and_eval, build_and_eval_maxsim, format_table


def _load_corpus(args):
    if args.corpus:
        from saaransh.corpus import load_local

        return load_local(args.corpus, dpi=args.dpi)
    from saaransh.corpus import load_vidore

    return load_vidore(args.vidore, limit=args.limit)


def _index_config(args) -> IndexConfig:
    if args.index == "flat":
        return IndexConfig(backend="numpy", kind="flat", quantize_int8=args.int8)
    kind = "flat" if args.index == "faiss-flat" else args.index
    return IndexConfig(
        backend="faiss", kind=kind,
        nlist=args.nlist, nprobe=args.nprobe, pq_m=args.pq_m, pq_nbits=args.pq_nbits,
    )


def _build_colqwen2(args):
    from saaransh.embedders.colqwen2_muvera import ColQwen2MuveraEmbedder

    return ColQwen2MuveraEmbedder(
        model_name=args.colqwen_model,
        token_dim=args.token_dim,
        cache_dir=args.cache_dir,
        local_files_only=args.local_only,
        projection_type=ProjectionType[args.muvera_mode.upper()],
        num_simhash_projections=args.k,
        num_repetitions=args.reps,
        final_projection_dimension=args.fde_compress,
    )


def _build_maxsim(args):
    from saaransh.embedders.colqwen2_maxsim import ColQwen2MaxSimRetriever

    return ColQwen2MaxSimRetriever(
        model_name=args.colqwen_model, token_dim=args.token_dim,
        cache_dir=args.cache_dir, local_files_only=args.local_only,
    )


def _build_gemma(args, precision: str, backend: str):
    from saaransh.embedders.gemma4_pooled import Gemma4PooledEmbedder

    return Gemma4PooledEmbedder(
        model_name=args.gemma_model,
        cache_dir=args.cache_dir,
        backend=backend, precision=precision, pooling=args.pooling,
        visual_tokens=args.visual_tokens, image_token_type=args.image_token_type,
    )


def _common(p: argparse.ArgumentParser) -> None:
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--corpus", help="local folder with images + queries.jsonl")
    src.add_argument("--vidore", help="ViDoRe dataset name")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dpi", type=int, default=175, help="PDF render DPI for --corpus")
    p.add_argument("--cache-dir", default="./model_cache", help="HF cache dir for ColQwen2")
    p.add_argument("--local-only", action="store_true", help="load cached weights, no network")
    # index
    p.add_argument("--index", default="flat", choices=["flat", "faiss-flat", "ivfpq", "hnsw"])
    p.add_argument("--int8", action="store_true", help="numpy flat: int8-quantize stored vectors")
    p.add_argument("--nlist", type=int, default=256)
    p.add_argument("--nprobe", type=int, default=32)
    p.add_argument("--pq-m", type=int, default=16, help="must divide the FDE dimension")
    p.add_argument("--pq-nbits", type=int, default=8)
    # MUVERA / ColQwen2
    p.add_argument("--colqwen-model", default="vidore/colqwen2-v1.0-hf",
                   help="merged HF checkpoint; repo id OR local dir")
    p.add_argument("--token-dim", type=int, default=128, help="128=ColQwen2, 320=ColQwen3.5")
    p.add_argument("--muvera-mode", default="default_identity",
                   help="default_identity | calibrated_eigenbasis | low_rank_gaussian | srht | cross_polytope")
    p.add_argument("--k", type=int, default=4, help="num_simhash_projections")
    p.add_argument("--reps", type=int, default=4, help="num_repetitions")
    p.add_argument("--fde-compress", type=int, default=None, help="final_projection_dimension")
    # Gemma
    p.add_argument("--pooling", default="mean", help="mean | last | image")
    p.add_argument("--image-token-type", type=int, default=1,
                   help="mm_token_type_ids value marking image tokens (confirm via probe)")
    p.add_argument("--visual-tokens", type=int, default=280)


def demo_main() -> None:
    p = argparse.ArgumentParser(description="ColQwen2+MUVERA vs Gemma 4 12B pooled")
    _common(p)
    p.add_argument("--only", choices=["colqwen2", "gemma", "maxsim"], help="run a single pipeline")
    p.add_argument("--maxsim", action="store_true", help="include the exact ColQwen2 MaxSim baseline")
    p.add_argument("--gemma-model", default="google/gemma-4-12b-it",
                   help="repo id OR local dir (e.g. ./model_cache/gemma-4-12b-it)")
    p.add_argument("--gemma-precision", default="bf16")
    p.add_argument("--gemma-backend", default="transformers")
    args = p.parse_args()

    if args.local_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    images, queries, qrels = _load_corpus(args)
    cfg = _index_config(args)
    print(f"corpus: {len(images)} images, {len(queries)} queries | index: {cfg.backend}:{cfg.kind}")

    results = []
    if args.maxsim or args.only == "maxsim":
        results.append(build_and_eval_maxsim(_build_maxsim(args), images, queries, qrels))
    if args.only in (None, "colqwen2"):
        emb = _build_colqwen2(args)
        if args.muvera_mode.upper() == "CALIBRATED_EIGENBASIS":
            emb.calibrate(images[: min(64, len(images))])
        results.append(build_and_eval(emb, images, queries, qrels, index_config=cfg))
    if args.only in (None, "gemma"):
        emb = _build_gemma(args, args.gemma_precision, args.gemma_backend)
        results.append(build_and_eval(emb, images, queries, qrels, index_config=cfg))

    print("\n" + format_table(results))


def sweep_main() -> None:
    p = argparse.ArgumentParser(description="Gemma 4 12B retrieval-quality vs weight precision")
    _common(p)
    p.add_argument("--gemma-model", default="google/gemma-4-12b-it",
                   help="repo id OR local dir")
    p.add_argument("--precisions", default="bf16,q8,q4")
    p.add_argument("--out", default="sweep_results.csv")
    args = p.parse_args()

    images, queries, qrels = _load_corpus(args)
    cfg = _index_config(args)
    rows, results = [], []
    for prec in args.precisions.split(","):
        backend = "transformers" if prec in {"bf16", "fp16"} else "mlx"
        emb = _build_gemma(args, prec, backend)
        r = build_and_eval(emb, images, queries, qrels, index_config=cfg)
        results.append(r)
        rows.append((prec, r.metrics["recall@1"], r.metrics["ndcg@5"], r.bytes_per_doc))

    print("\n" + format_table(results))
    with open(args.out, "w") as f:
        f.write("precision,recall@1,ndcg@5,bytes_per_doc\n")
        for prec, r1, n5, bpd in rows:
            f.write(f"{prec},{r1:.4f},{n5:.4f},{bpd:.0f}\n")
    print(f"\nwrote {args.out}")


# ── cache + MUVERA frontier sweep ────────────────────────────────────────
def cache_main() -> None:
    p = argparse.ArgumentParser(description="Cache ColQwen2 multivectors once for cheap sweeps")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--corpus")
    src.add_argument("--vidore")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dpi", type=int, default=175)
    p.add_argument("--colqwen-model", default="vidore/colqwen2-v1.0-hf")
    p.add_argument("--token-dim", type=int, default=128)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--local-only", action="store_true")
    p.add_argument("--out", default="./cache/colqwen2_bags")
    args = p.parse_args()
    if args.local_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    images, queries, qrels = _load_corpus(args)
    print(f"corpus: {len(images)} images, {len(queries)} queries -> extracting ColQwen2 bags")

    from pathlib import Path

    from saaransh.cache import extract_colqwen2_bags, save_cache

    doc_bags, query_bags = extract_colqwen2_bags(
        images, queries, model_name=args.colqwen_model, token_dim=args.token_dim,
        cache_dir=args.cache_dir, local_files_only=args.local_only,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    meta = {"source": args.corpus or args.vidore, "token_dim": args.token_dim,
            "n_docs": len(doc_bags), "n_queries": len(query_bags)}
    save_cache(args.out, doc_bags, query_bags, qrels, meta)
    print(f"wrote {args.out}.npz  (+ .meta.json)")


def _intlist(s):
    return [int(x) for x in s.split(",") if x.strip()]


def muvera_sweep_main() -> None:
    p = argparse.ArgumentParser(description="Sweep MUVERA configs over a cached corpus")
    p.add_argument("--cache", default="./cache/colqwen2_bags")
    p.add_argument("--modes", default="default_identity,calibrated_eigenbasis")
    p.add_argument("--k", default="4,6")
    p.add_argument("--reps", default="4,8")
    p.add_argument("--compress", default="none,8192", help="final FDE dims; 'none' = uncompressed")
    p.add_argument("--ceiling", action="store_true", help="also compute exact-MaxSim from cache")
    p.add_argument("--out-csv", default="muvera_sweep.csv")
    p.add_argument("--plot", default=None, help="path to save an nDCG-vs-bytes PNG")
    args = p.parse_args()

    from saaransh.cache import load_cache
    from saaransh.sweep import ceiling_row, format_sweep, grid, plot_frontier, run_muvera_sweep

    doc_bags, query_bags, qrels, meta = load_cache(args.cache)
    token_dim = int(meta.get("token_dim", 128))
    print(f"cache: {len(doc_bags)} docs, {len(query_bags)} queries, token_dim={token_dim}")

    compress = [None if c.strip().lower() == "none" else int(c) for c in args.compress.split(",")]
    cfgs = grid(args.modes.split(","), _intlist(args.k), _intlist(args.reps), compress)

    rows = run_muvera_sweep(doc_bags, query_bags, qrels, cfgs, token_dim=token_dim)
    ceil = ceiling_row(query_bags, doc_bags, qrels) if args.ceiling else None
    all_rows = ([ceil] if ceil else []) + rows

    print("\n" + format_sweep(all_rows))
    cols = ["name", "fde_dim", "bytes_per_doc", "recall@1", "recall@5", "ndcg@5", "mrr@10", "sweep_s"]
    with open(args.out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in all_rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nwrote {args.out_csv}")
    if args.plot:
        plot_frontier(rows, args.plot, ceiling=ceil)
        print(f"wrote {args.plot}")
