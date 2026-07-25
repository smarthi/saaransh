"""Roll up per-dataset results/<name>/muvera_sweep.csv + failure_report.csv into one table.

Usage: python scripts/tabulate_results.py --root ./results --out summary.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_sweep(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def best_muvera_row(rows: list[dict]) -> dict | None:
    muvera_rows = [r for r in rows if not r["name"].startswith("colqwen2-maxsim")]
    if not muvera_rows:
        return None
    return max(muvera_rows, key=lambda r: float(r["ndcg@5"]))


def ceiling_row(rows: list[dict]) -> dict | None:
    for r in rows:
        if r["name"].startswith("colqwen2-maxsim"):
            return r
    return None


def failure_summary(path: Path) -> str:
    if not path.exists():
        return "-"
    with path.open() as f:
        rows = list(csv.DictReader(f))
    lost = sum(1 for r in rows if r["ceiling_ok_muvera_lost"] == "True")
    return f"{lost}/{len(rows)} lost"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./results")
    p.add_argument("--out", default="summary.md")
    args = p.parse_args()

    root = Path(args.root)
    dataset_dirs = sorted(d for d in root.iterdir() if d.is_dir())

    header = ["dataset", "ceiling nDCG@5", "best MUVERA nDCG@5", "% of ceiling",
              "best config", "storage/page", "ceiling-lost-by-MUVERA"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    for d in dataset_dirs:
        sweep_csv = d / "muvera_sweep.csv"
        if not sweep_csv.exists():
            continue
        rows = load_sweep(sweep_csv)
        ceil = ceiling_row(rows)
        best = best_muvera_row(rows)
        if not ceil or not best:
            continue
        ceil_n, best_n = float(ceil["ndcg@5"]), float(best["ndcg@5"])
        pct = f"{100 * best_n / ceil_n:.0f}%" if ceil_n > 0 else "-"
        storage_kb = f"{float(best['bytes_per_doc']) / 1024:.0f} KB"
        fail = failure_summary(d / "failure_report.csv")
        lines.append(
            f"| {d.name} | {ceil_n:.3f} | {best_n:.3f} | {pct} | "
            f"{best['name']} | {storage_kb} | {fail} |"
        )

    out_text = "\n".join(lines)
    Path(args.out).write_text(out_text + "\n")
    print(out_text)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()