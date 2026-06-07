# saaransh

**Single-vector multimodal retrieval bench — ColQwen2 + MUVERA vs Gemma 4 12B pooled, on a 32 GB Mac.**

A head-to-head harness for one question: if you collapse a late-interaction model
(ColQwen2) to a single vector with MUVERA, how does it compare — on retrieval
quality *and* on storage/latency — to using a generative encoder-free VLM
(Gemma 4 12B) as a zero-shot single-vector embedder?

Both pipelines emit exactly **one** dense vector per page, drop into the same flat
index, and are scored with the same metrics, so the comparison is about
representation, not infrastructure.

## The name

**सारांश (sārāṃśa)** — Sanskrit/Hindi for *summary*, *gist*, or *the distilled essence*.
A fitting name for a bench about reducing a whole document page to a single vector: the
question throughout is how much of a page's retrievable essence survives when you collapse
its ~1,000 late-interaction patch vectors down to one. *saaransh* is exactly that — the
gist that's left after the collapse.

## The three arms

Three retrievers, same corpus and metrics:

**Baseline — ColQwen2 exact MaxSim.** Keeps the full ~1000-vector bag per page and scores
with native late interaction (`processor.score_multi_vector`). The **quality ceiling** and
the fat end of the storage axis (~0.5 MB/page). It answers the question that makes the whole
comparison meaningful: how much retrieval quality does collapsing to a single vector cost?

**Pipeline 1 — ColQwen2 + MUVERA.** The same patch bag collapsed by
[pymuvera](https://github.com/smarthi/pymuvera) into one Fixed Dimensional Encoding whose
**inner product approximates MaxSim** (query SUM / document AVERAGE aggregation). One vector
(~32 KB), scored without ever materializing the full MaxSim.

**Pipeline 2 — Gemma 4 12B pooled.** Gemma 4 12B (released 2026-06-03) is a *generative*,
**encoder-free**, decoder-only VLM — no vision encoder to tap (the 35M module drops raw
48×48 patches straight into the decoder). "Embedding" means a forward pass + pooling the
last hidden states into one vector (~15 KB), scored by **cosine**. Never trained for
retrieval; that is the thing under test.

### One correctness detail that is easy to get wrong
MUVERA FDEs must **not** be L2-normalized — the SUM (query) / AVERAGE (document)
asymmetry is what makes the dot product approximate MaxSim. So the metric lives on
the embedder: ColQwen2+MUVERA → `ip`, Gemma pooled → `cosine`. The index reads
`embedder.metric` and does the right thing.

### Token-dim gotcha
ColQwen2 emits 128-d tokens; ColQwen3.5 emits 320-d. Building MUVERA at 128 against a
v3 backbone silently truncates ~60% of the representation — the embedder asserts the
dims match and tells you to pass `--token-dim 320`.

## Install (uv)

```bash
uv venv && source .venv/bin/activate
uv pip install -e .                 # core only (numpy + pymuvera): runs the tests
uv pip install -e '.[colqwen2]'     # ColQwen2 (transformers-native, no colpali-engine)
uv pip install -e '.[gemma]'        # Gemma 4 pipeline, bf16/fp16 on MPS
uv pip install -e '.[mlx]'          # Apple-Silicon quantized Gemma (q8/q4) for the sweep
uv pip install -e '.[faiss]'        # FAISS backend (ivfpq / hnsw / faiss-flat)
uv pip install -e '.[pdf]'          # PDF ingestion (PyMuPDF)
uv pip install -e '.[vidore,plot]'  # ViDoRe loader + sweep plots
```

ColQwen2 uses the transformers-native `ColQwen2ForRetrieval` on the **merged** checkpoint
`vidore/colqwen2-v1.0-hf` (no LoRA adapter, no colpali-engine, scored with
`processor.score_retrieval`). Point `--colqwen-model` at a local dir to run offline:

```bash
hf download vidore/colqwen2-v1.0-hf --local-dir ./model_cache/colqwen2-v1.0-hf
```

Gemma loads the same way (`--gemma-model`). Both ride one transformers-5.x environment.
Pass `--local-only` to forbid network fetches. The loader sets the MPS flags
(`PYTORCH_ENABLE_MPS_FALLBACK`, watermark off, tokenizer parallelism off) for you.

## Run

```bash
# Your own pages: a folder of *.png / *.jpg / *.pdf + queries.jsonl
# PDFs are rendered to pages named "<pdfstem>_p<n>"; reference them in queries.jsonl:
#   {"query": "What is Scaled Dot-Product Attention?", "image": "attention_p3"}
saaransh-demo --corpus ./sample_docs

# All three arms (add the MaxSim ceiling), offline against your cached ColQwen2
saaransh-demo --corpus ./sample_docs --maxsim --local-only --cache-dir ./model_cache

# ViDoRe subset, both pipelines
saaransh-demo --vidore vidore/docvqa_test_subsampled --limit 200

# MUVERA with the calibrated eigenbasis mode (your SpectralQuant-inspired Mode 5)
saaransh-demo --corpus ./sample_docs --muvera-mode calibrated_eigenbasis --k 6

# The headline experiment: retrieval quality vs Gemma weight precision
saaransh-sweep --vidore vidore/docvqa_test_subsampled --limit 200 --precisions bf16,q8,q4
```

Example table:

```
pipeline                             dim    B/doc  recall@1  recall@5    ndcg@5    mrr@10   idx s   qry s
------------------------------------------------------------------------------------------------------------
colqwen2+muvera[DEFAULT_IDENTITY]   8192    32768     ...       ...        ...       ...      ...     ...
gemma4-12b[transformers/bf16/mean]  3840    15360     ...       ...        ...       ...      ...     ...
```

## Index backends

The index is a thin layer behind one interface (`add` / `search` / `stats`), so the
comparison stays about representation, not infra:

| `--index`    | backend | use |
|--------------|---------|-----|
| `flat`       | numpy (no faiss)   | default; exact; `--int8` adds scalar quant |
| `faiss-flat` | faiss `IndexFlatIP`| exact, faster at scale |
| `ivfpq`      | faiss `IndexIVFPQ` | the production ANN path; `--nlist --nprobe --pq-m --pq-nbits` |
| `hnsw`       | faiss `IndexHNSWFlat` | graph ANN |

Metric handling is uniform: both `ip` (MUVERA) and `cosine` (Gemma) reduce to **inner
product** at the index — the embedder owns normalization, so MUVERA FDEs stay raw and
keep their Chamfer approximation. `ivfpq` mirrors a production OpenSearch IVF+PQ store
(e.g. `nlist=4096, nprobe=128, pq_m=16` on a large ColQwen2 corpus); `pq_m` must divide
the FDE dimension, so pair it with `--fde-compress` to a PQ-friendly width. PQ is where a
real codec slots in; the numpy `--int8` knob is the zero-dep stand-in.

```bash
saaransh-demo --vidore vidore/docvqa_test_subsampled --limit 500 \
  --index ivfpq --fde-compress 1024 --pq-m 16 --nlist 256 --nprobe 32
```

## Status: no real documents tested yet

The committed tests run on **synthetic data only** — random multi-vector bags through the
real pymuvera encoder, plus FAISS flat/IVFPQ parity checks — and confirm the wiring
(scoring, index, metrics) is correct. **No ColQwen2 / Gemma forward pass and no real page
image has been run.** The ViDoRe and local-corpus loaders are wired but unexecuted. The
first empirical retrieval numbers come from running this on your Mac; nothing here is a
benchmark.

## What you verify on device

**First run — Gemma 4 bring-up.** Download the instruct weights (Apache-2.0; if `hf
download` 401s, accept the license on the model page then `hf auth login`):

```bash
hf download google/gemma-4-12b-it --local-dir ./model_cache/gemma-4-12b-it
```

Then run the probe — it loads Gemma alone (fits bf16 in 32 GB), pushes one image
through, and prints the hidden-state contract the embedder depends on:

```bash
HF_HUB_OFFLINE=1 python scripts/probe_gemma.py \
  --model ./model_cache/gemma-4-12b-it --image ./sample_docs/page.png
```

(Pass the `--local-dir` folder as `--model` — `from_pretrained` takes a path directly, so
no hub-cache layout and no re-download. `--offline`/`HF_HUB_OFFLINE=1` forbids any fetch.)

The embedder is pre-aligned to the official card API (`AutoModelForMultimodalLM` with an
`AutoModelForImageTextToText` fallback, image-before-text, `enable_thinking=False`,
per-item processing). The probe confirms the class, where `hidden_size` lives, and the
last-layer shape. The remaining `# VERIFY` is MLX hidden-state extraction in
`gemma4_pooled._mlx_hidden` — only needed once you move to the q4/q8 sweep; the
transformers bf16 path is the working baseline. Run the arms in separate invocations on
32 GB so you never hold Gemma (~24 GB bf16) and ColQwen2 at once.

## 32 GB memory budget (Gemma 4 12B)

| precision | weights | headroom on 32 GB | path |
|-----------|---------|-------------------|------|
| bf16      | ~24 GB  | tight; expect swap under load | transformers (MPS) |
| q8        | ~12.5 GB| comfortable (~18 GB free)     | mlx |
| q4        | ~6.5 GB | luxurious                     | mlx |

bf16 is the quality reference; q8 is the practical default; q4 is where the
retrieval-quality-vs-bits curve gets interesting.

## Knobs worth a figure in the writeup

- **Weight precision** (`saaransh-sweep`): nDCG/Recall vs bf16/q8/q4 — the curve almost
  nobody publishes (everyone reports perplexity instead).
- **Visual token budget** (`--visual-tokens 70|140|280|560|1120`): unique to Gemma 4;
  detail vs latency vs memory.
- **MUVERA mode** (`--muvera-mode`): DEFAULT_IDENTITY baseline vs CALIBRATED_EIGENBASIS
  and the FDE-dimension / quality tradeoff (`--fde-compress`).
- **Vector storage** (`--index ivfpq` / `--int8`): FAISS IVF+PQ codes (or the zero-dep
  int8 path) — quantize the *stored* vectors and read storage vs quality independently
  of model weights. This is the slot for a PQ / dedicated codec.

Apache-2.0.