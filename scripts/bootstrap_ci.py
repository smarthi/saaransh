"""Bootstrap CIs on rank_delta gaps from failure_report.csv files.

Answers: is the specific_entity_lookup vs descriptive_general gap (and each
individual feature's gap) actually distinguishable from noise, or is it small
enough to be a sampling artifact? Percentile bootstrap, resampling within each
group independently.

Usage:
    python scripts/bootstrap_ci.py --root ./results \
        --datasets docvqa,arxivqa,infovqa,tabfquad --out bootstrap_summary.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

BOOL_FEATURES = ["has_digit", "has_at", "has_color", "has_quoted_or_acronym", "has_identifier_noun"]


def load_report(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["rank_delta"] = int(r["rank_delta"])
        for col in BOOL_FEATURES:
            if col in r:
                r[col] = r[col] == "True"
    return rows


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int, ci: float, rng: np.random.Generator):
    """Diff = mean(a) - mean(b), resampling each group independently with replacement."""
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        a_s = rng.choice(a, size=len(a), replace=True)
        b_s = rng.choice(b, size=len(b), replace=True)
        diffs[i] = a_s.mean() - b_s.mean()
    lo, hi = np.percentile(diffs, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./results")
    p.add_argument("--datasets", default="docvqa,arxivqa,infovqa,tabfquad")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--ci", type=float, default=95.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="bootstrap_summary.md")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    datasets = args.datasets.split(",")

    lines = [
        f"# Bootstrap {args.ci:.0f}% CIs on rank_delta gaps ({args.n_boot} resamples, seed={args.seed})",
        "",
        "Positive diff = specific/feature-present group lost more ground to ceiling than the "
        "comparison group. \"excludes 0\" = the interval doesn't cross zero at this CI, i.e. the "
        "gap survives resampling noise; it does NOT by itself mean the effect is large or "
        "practically important, only that it's not indistinguishable from zero at this sample size.",
        "",
        "| dataset | split | n (group A / B) | diff | CI | excludes 0 |",
        "|---|---|---|---|---|---|",
    ]

    for name in datasets:
        report_path = Path(args.root) / name / "failure_report.csv"
        if not report_path.exists():
            lines.append(f"| {name} | (missing failure_report.csv) | - | - | - | - |")
            continue
        rows = load_report(report_path)
        deltas = np.array([r["rank_delta"] for r in rows], dtype=float)

        # composite bucket
        is_specific = np.array([r["bucket"] == "specific_entity_lookup" for r in rows])
        a, b = deltas[is_specific], deltas[~is_specific]
        if len(a) and len(b):
            diff, lo, hi = bootstrap_diff_ci(a, b, args.n_boot, args.ci, rng)
            excl = "yes" if (lo > 0 or hi < 0) else "no"
            lines.append(f"| {name} | bucket: specific_entity_lookup vs descriptive_general | "
                          f"{len(a)}/{len(b)} | {diff:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {excl} |")

        # each individual structural feature
        for feat in BOOL_FEATURES:
            if feat not in rows[0]:
                continue
            has = np.array([r[feat] for r in rows])
            a, b = deltas[has], deltas[~has]
            if len(a) < 3 or len(b) < 3:
                lines.append(f"| {name} | {feat} (present vs absent) | {len(a)}/{len(b)} | "
                              f"- | too few samples | - |")
                continue
            diff, lo, hi = bootstrap_diff_ci(a, b, args.n_boot, args.ci, rng)
            excl = "yes" if (lo > 0 or hi < 0) else "no"
            lines.append(f"| {name} | {feat} (present vs absent) | {len(a)}/{len(b)} | "
                          f"{diff:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {excl} |")

    out_text = "\n".join(lines)
    Path(args.out).write_text(out_text + "\n")
    print(out_text)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()