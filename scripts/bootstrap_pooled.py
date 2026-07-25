"""Pooled, cross-dataset bootstrap CIs on rank_delta gaps.

Per-dataset tests (bootstrap_ci.py) are underpowered for small subgroups
(has_color n=17 in arxivqa, n=2 in docvqa, n=0 in tabfquad, etc.). This pools
each feature across all four datasets to ask: combining everything we have,
is has_color / has_digit / etc. actually harder or easier for MUVERA overall?

Uses a STRATIFIED bootstrap: each dataset's group A and group B are resampled
independently (preserving each dataset's own n and variance), then the
resampled values are pooled across datasets before computing the mean diff.
This avoids one dataset silently dominating the resampling just because it
has more rows with the feature present.

Usage:
    python scripts/bootstrap_pooled.py --root ./results \
        --datasets docvqa,arxivqa,infovqa,tabfquad --out bootstrap_pooled.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

BOOL_FEATURES = ["has_digit", "has_at", "has_color", "has_quoted_or_acronym", "has_identifier_noun"]
MIN_PER_GROUP = 5  # a dataset contributes to the pool only past this many rows in a group


def load_report(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["rank_delta"] = int(r["rank_delta"])
        for col in BOOL_FEATURES:
            if col in r:
                r[col] = r[col] == "True"
    return rows


def stratified_bootstrap_diff_ci(
    per_dataset: dict[str, tuple[np.ndarray, np.ndarray]],
    n_boot: int,
    ci: float,
    rng: np.random.Generator,
):
    """per_dataset: {name: (group_a_values, group_b_values)}. Resamples each
    dataset's own groups independently each iteration, then pools across
    datasets to compute the overall mean diff for that iteration."""
    observed_a = np.concatenate([a for a, _ in per_dataset.values() if len(a)])
    observed_b = np.concatenate([b for _, b in per_dataset.values() if len(b)])
    observed_diff = float(observed_a.mean() - observed_b.mean())

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pooled_a, pooled_b = [], []
        for a, b in per_dataset.values():
            if len(a):
                pooled_a.append(rng.choice(a, size=len(a), replace=True))
            if len(b):
                pooled_b.append(rng.choice(b, size=len(b), replace=True))
        pa = np.concatenate(pooled_a)
        pb = np.concatenate(pooled_b)
        diffs[i] = pa.mean() - pb.mean()

    lo, hi = np.percentile(diffs, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return observed_diff, float(lo), float(hi), len(observed_a), len(observed_b)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./results")
    p.add_argument("--datasets", default="docvqa,arxivqa,infovqa,tabfquad")
    p.add_argument("--n-boot", type=int, default=3000)
    p.add_argument("--ci", type=float, default=95.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="bootstrap_pooled.md")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    dataset_names = args.datasets.split(",")

    loaded: dict[str, list[dict]] = {}
    for name in dataset_names:
        path = Path(args.root) / name / "failure_report.csv"
        if path.exists():
            loaded[name] = load_report(path)
        else:
            print(f"warning: {path} missing, skipping {name}")

    lines = [
        f"# Pooled cross-dataset bootstrap {args.ci:.0f}% CIs "
        f"(stratified, {args.n_boot} resamples, seed={args.seed})",
        "",
        "Each dataset's own group is resampled independently, then pooled across "
        "datasets, before computing the mean diff per bootstrap iteration — a dataset "
        "can't dominate the estimate just by having more rows with the feature present. "
        f"A dataset only contributes to a feature's pool if it has at least {MIN_PER_GROUP} "
        "rows in that group; per-dataset counts are shown so you can see exactly which "
        "datasets are actually driving each pooled result.",
        "",
        "| feature | per-dataset n (present) | pooled n (present/absent) | diff | CI | excludes 0 |",
        "|---|---|---|---|---|---|",
    ]

    all_bucket = {}  # composite bucket handled the same way as a "feature"
    for feat in BOOL_FEATURES + ["bucket"]:
        per_dataset: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        per_dataset_n_present: list[str] = []
        for name, rows in loaded.items():
            if feat == "bucket":
                is_a = np.array([r["bucket"] == "specific_entity_lookup" for r in rows])
            else:
                if feat not in rows[0]:
                    continue
                is_a = np.array([r[feat] for r in rows])
            deltas = np.array([r["rank_delta"] for r in rows], dtype=float)
            a, b = deltas[is_a], deltas[~is_a]
            if len(a) >= MIN_PER_GROUP and len(b) >= MIN_PER_GROUP:
                per_dataset[name] = (a, b)
                per_dataset_n_present.append(f"{name}={len(a)}")
            elif len(a) > 0:
                per_dataset_n_present.append(f"{name}={len(a)}(excluded,<{MIN_PER_GROUP})")

        if len(per_dataset) < 1:
            lines.append(f"| {feat} | {', '.join(per_dataset_n_present) or 'none'} | "
                          f"insufficient data across all datasets | - | - | - |")
            continue

        diff, lo, hi, n_a, n_b = stratified_bootstrap_diff_ci(per_dataset, args.n_boot, args.ci, rng)
        excl = "yes" if (lo > 0 or hi < 0) else "no"
        lines.append(f"| {feat} | {', '.join(per_dataset_n_present)} | {n_a}/{n_b} | "
                      f"{diff:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {excl} |")

    out_text = "\n".join(lines)
    Path(args.out).write_text(out_text + "\n")
    print(out_text)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()