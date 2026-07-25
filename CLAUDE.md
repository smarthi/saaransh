# saaransh — project instructions for Claude Code

Single-vector multimodal retrieval benchmark: ColQwen2 MaxSim (exact ceiling)
vs MUVERA (FDE-based single-vector encoding) vs Gemma-4-12B mean-pooled
hidden states, across 4 ViDoRe datasets (docvqa, arxivqa, infovqa, tabfquad).
Goal: turn this into a paper / arXiv technical report.

Full history, decisions, and current real results: **read docs/paper_context.md
before doing anything on this project.** It has the actual numbers, the
mechanistic finding, and exactly what's left to do — don't re-derive any of
it from scratch or guess at what's already been decided.

## Working agreements
- Run things, don't estimate or narrate what they'd probably show. If a step
  fails, show the real error before retrying or working around it silently.
- Report raw script/test output (console dumps, CSV contents), not a
  paraphrased summary invented on top of it.
- Run `pytest tests/` after any src/ change, before touching real data.
- `KMP_DUPLICATE_LIB_OK=TRUE` is required on macOS (faiss+torch OpenMP
  conflict) — this is a known, already-solved issue, don't re-debug it.
- Different datasets have different best MUVERA configs (see
  docs/paper_context.md) — don't apply one config across all four blindly.