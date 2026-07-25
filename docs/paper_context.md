# Saaransh → paper: context and current state

## What this project is
Saaransh benchmarks whether single-vector retrieval methods recover the
retrieval quality of exact late-interaction (ColBERT-style MaxSim), using
ColQwen2 as the multi-vector model. Three arms:
- **ceiling**: ColQwen2 MaxSim (exact, expensive — the thing everyone wants
  to approximate)
- **MUVERA**: fixed-dimensional-encoding (FDE) projection of the same
  ColQwen2 token embeddings — a principled approximation
- **Gemma-4-12B pooled**: mean-pooled hidden states from a general-purpose
  VLM — the "what if you just pool a normal model instead" baseline

## Headline result (current, real, from the 4-dataset run in results/)
MUVERA recovers 82–94% of ceiling nDCG@5 across 4 domains at 512KB–1MB/page:

| dataset  | ceiling | MUVERA | % of ceiling | best config          | storage/page |
|----------|---------|--------|---------------|-----------------------|--------------|
| arxivqa  | 0.895   | 0.830  | 93%           | default_identity k8/r8 | 1024 KB     |
| docvqa   | 0.655   | 0.534  | 82%           | calibrated_eigenbasis k8/r8 | 1024 KB |
| infovqa  | 0.914   | 0.839  | 92%           | calibrated_eigenbasis k8/r8 | 1024 KB |
| tabfquad | 0.594   | 0.557  | 94%           | default_identity k8/r4 | 512 KB      |

**Note the config isn't uniform** — arxivqa and tabfquad's best frontier
point uses `default_identity`, not `calibrated_eigenbasis`. Don't assume one
config applies across all four datasets.

Gemma pooled: nDCG@5 = 0.025, recall@1 = 0.01 — essentially chance on a
200-doc corpus. **This is the real contrast to lead the paper with**:
structured projection vs. naive pooling. An earlier framing in this project's
history said "MUVERA loses ~35% of ceiling" as if that were a weakness —
that number predates this actual run and is wrong; correct it if you see it
anywhere. 82–94% recovery is a strong result, not a weak one.

## Mechanistic finding (the actual contribution, not just a leaderboard)
Reading the worst-N regressions (results/<dataset>/failure_report.csv)
across all 4 datasets, independent of domain or language, MUVERA's losses
concentrate on queries asking for one small localized detail (a number, an
email/handle, a color, a named/quoted field) vs. broad descriptive or
inferential questions about the page. This held up qualitatively across
scanned forms (docvqa), scientific figures (arxivqa), infographics
(infovqa), and French financial tables (tabfquad).

`scripts/failure_analysis.py` encodes this as a testable feature set —
`has_digit`, `has_at`, `has_color`, `has_quoted_or_acronym`,
`has_identifier_noun` → composite `bucket`: `specific_entity_lookup` vs
`descriptive_general` — replacing an earlier first-word heuristic that only
recognized English question words and produced a single useless bucket on
the French tabfquad data. Validated at 11/12 on a hand-labeled sanity set;
known miss category: bare proper-noun references with no digit/color/
quote/acronym marker (e.g. "who is the author of X") — this is a stated,
accepted limitation, not something to chase further with more regex.

Query length was tested and **ruled out** as the driver: corr(length,
rank_delta) = −0.14, −0.11, +0.02, −0.14 across the four datasets. State
this as a ruled-out hypothesis in the paper — don't just omit it.

### NEXT STEP — not yet done
Re-run `failure_analysis.py` against all 4 cached datasets with the fixed
script, **using each dataset's actual best config from muvera_sweep.csv**
(see table above), and check whether `specific_entity_lookup` shows a
consistently higher avg `rank_delta` than `descriptive_general` across all
four. That's the statistical backing for the mechanism claim, replacing "I
read 20 examples by eye and noticed a pattern."

## Remaining gaps for the paper
1. **Failure-analysis re-run with fixed bucketing + correct per-dataset
   config** — do this first, see above.
2. **Weight-precision sweep is incomplete.** Only `bf16` ran on docvqa
   (results/gemma_precision_sweep.csv). MLX q8/q4 hidden-state extraction is
   an unimplemented stub in `gemma4_pooled.py`. Decision needed: implement
   real quantized extraction, or drop the "precision curve nobody publishes"
   claim from the paper entirely. Do not report bf16-only as if it were a
   completed sweep.
3. **Related work section not yet drafted.** Needs positioning against the
   ColPali/ColQwen2 paper, the original MUVERA paper, and the 2025–2026
   single-vector-vs-multi-vector retrieval literature. Not yet researched.
4. Best MUVERA config isn't uniform across datasets (see table) — report
   the per-dataset frontier curve (results/<name>/frontier.png already
   exist), not a single "recommended" operating point across domains.
5. `idx_s`/`query_s`/`sweep_s` latency numbers are already collected in the
   sweep CSVs — just needs surfacing into the results tables. No new
   experiments required for this one.

## Repo/data pointers
- `results/<dataset>/{cache, muvera_sweep.csv, frontier.png,
  failure_report.csv}` for docvqa, arxivqa, infovqa, tabfquad
- `results/gemma_precision_sweep.csv` — bf16 only, see gap #2
- `summary.md` / `summary_results.md` — rolled-up tables + failure-analysis
  console output from the first full run (pre-bucketing-fix)
- `run_all.sh`, `scripts/tabulate_results.py` — orchestration, already
  working end to end

## Run notes / known environment issues (already solved once, don't re-debug)
- `KMP_DUPLICATE_LIB_OK=TRUE` needed on macOS (faiss+torch OpenMP crash)
- Gemma 4 12B download is flaky over HF (xet stalls, connection resets) —
  point `--gemma-model` at an already-complete local checkpoint if this
  happens again rather than retrying the download blindly
- MLX q8/q4 quantized hidden-state extraction: not implemented (see gap #2)

## Paper target
Working direction: single-vector multimodal document retrieval — when does
FDE-based projection hold up vs. naive pooling, and where specifically does
it lose. Likely path: arXiv technical report first; SIGIR/ECIR resource
track submission if gaps 1–4 get filled out with rigor (multi-dataset +
ablation + mechanism + latency — most of which is already done).