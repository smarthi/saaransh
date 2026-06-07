# saaransh

**Single-vector multimodal document retrieval bench — how much late-interaction quality survives when you collapse a page to one vector? Runs on a 32 GB MacBook.**

Three retrievers, one corpus, one set of metrics:

- **ColQwen2 MaxSim** — full late interaction. The quality *ceiling*.
- **ColQwen2 + MUVERA** — the same patch bag collapsed to one vector. The *bridge*.
- **Gemma 4 12B pooled** — a generative, encoder-free VLM used as a zero-shot embedder. The *control*.

All three score through the same index and metrics, so the comparison is about
representation, not infrastructure. Headline finding: **the embedding has to be trained
to be an embedding** — see [Results](#results).

## The name

**सारांश (sārāṃśa)** — Sanskrit/Hindi for *summary*, *gist*, or *the distilled essence*.
A fitting name for a bench about reducing a whole document page to a single vector: the
question throughout is how much of a page's retrievable essence survives when you collapse
its ~1,000 late-interaction patch vectors down to one. *saaransh* is exactly that — the
gist that's left after the collapse.

## The three arms

Three retrievers, same corpus and metrics:

**Baseline — ColQwen2 exact MaxSim.** Keeps the full ~750-vector bag per page and scores
with native late interaction (`processor.score_retrieval`). The **quality ceiling** and
the fat end of the storage axis (~385 KB/page). It answers the question that makes the whole
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

# All three arms on a ViDoRe subset (run separately on 32 GB to avoid co-resident models)
saaransh-demo --only maxsim   --vidore vidore/docvqa_test_subsampled --limit 200 \
  --colqwen-model ./model_cache/colqwen2-v1.0-hf
saaransh-demo --only colqwen2 --vidore vidore/docvqa_test_subsampled --limit 200 \
  --colqwen-model ./model_cache/colqwen2-v1.0-hf --muvera-mode calibrated_eigenbasis --k 6 --reps 8
saaransh-demo --only gemma    --vidore vidore/docvqa_test_subsampled --limit 200 \
  --gemma-model ./model_cache/gemma-4-12b-it --pooling image

# The MUVERA frontier: cache ColQwen2 bags once (~250 s), then sweep configs in seconds each
saaransh-cache       --vidore vidore/docvqa_test_subsampled --limit 200 \
  --colqwen-model ./model_cache/colqwen2-v1.0-hf --out ./cache/docvqa200
saaransh-muvera-sweep --cache ./cache/docvqa200 --ceiling \
  --modes default_identity,calibrated_eigenbasis --k 4,6,8 --reps 4,8 \
  --compress none,32768,8192,2048 --plot frontier.png
```

Example output (DocVQA-200, all three arms):

```
pipeline                             dim    B/doc  recall@1  recall@5    ndcg@5    mrr@10
-----------------------------------------------------------------------------------------
colqwen2-maxsim (ceiling)            128   385249     0.565     0.735     0.660     0.642
colqwen2+muvera[CALIBRATED]         8192    32768     0.320     0.530     0.430     0.392
gemma4-12b[transformers/bf16/mean]  3840    15360     0.010     0.045     0.028     0.024
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

## Results

On a 200-page DocVQA subsample (ViDoRe), one relevant page per query, bf16 on Apple Silicon:

| arm | nDCG@5 | R@1 | R@5 | bytes/page | % of ceiling |
|---|---|---|---|---|---|
| ColQwen2 MaxSim (ceiling) | **0.660** | 0.565 | 0.735 | ~385 KB | 100% |
| ColQwen2 + MUVERA (calibrated, 32 KB) | 0.430 | 0.320 | 0.530 | 32 KB | 65% |
| Gemma 4 12B pooled | 0.028 | 0.010 | 0.045 | 15 KB | 4% |

The MUVERA storage/quality frontier (best nDCG@5 achievable at each storage budget):

| storage/page | best config | nDCG@5 | % of ceiling |
|---|---|---|---|
| 32 KB | calibrated, k4/r4 | 0.430 | 65% |
| 128 KB | calibrated, k6/r4 | 0.475 | 72% |
| 512 KB | calibrated, k8/r4 | 0.507 | 77% |
| 1 MB | calibrated, k8/r8 | 0.533 | 81% |

What it shows:

- **MUVERA is a real but lossy bridge** — ~65% of late-interaction quality at ~12×
  compression, in one ANN-friendly vector. It never reaches parity: even uncompressed
  (1 MB/page) it tops out at 81%, so its value is the small-storage regime, not the high end.
- **Calibration owns the frontier.** `CALIBRATED_EIGENBASIS` beats the default SimHash
  construction at every storage level; and a compact FDE built directly beats a large one
  projected down to the same size.
- **A generative VLM is not a zero-shot retriever.** Gemma 4 12B reads documents well, but
  its pooled hidden states sit at chance across mean / last-token / image-only pooling —
  nothing trained its activations into a comparable query/document space.

The thesis: *the embedding has to be trained to be an embedding.* MUVERA compresses an
already-trained retrieval signal and degrades gracefully because there's real structure to
preserve; a generative decoder has none to compress.

Reproduce the frontier: `saaransh-cache` once, then `saaransh-muvera-sweep --ceiling --plot frontier.png`.

**Scope:** one dataset, 200 examples, one-gold-page relevance. A consistent yardstick across
methods and a directional result — not a leaderboard claim. The committed tests run on
synthetic data (real pymuvera scoring, FAISS parity); the numbers above come from real runs.

## Running the models (offline)

Download the merged checkpoints once, then point `--colqwen-model` / `--gemma-model` at the
local folders and pass `--local-only` to run without network:

```bash
hf download vidore/colqwen2-v1.0-hf --local-dir ./model_cache/colqwen2-v1.0-hf
hf download google/gemma-4-12b-it    --local-dir ./model_cache/gemma-4-12b-it
```

ColQwen2 (`ColQwen2ForRetrieval`) and Gemma 4 (`AutoModelForMultimodalLM`) both load on one
transformers-5.x environment; the loader sets the MPS flags for you. On 32 GB, run the arms
in separate invocations so you never hold Gemma (~24 GB bf16) and ColQwen2 at once.

`scripts/probe_gemma.py` loads Gemma alone and prints its hidden-state contract — handy if a
future transformers bump shifts the API. The one path still marked `# VERIFY` is MLX
hidden-state extraction (`gemma4_pooled._mlx_hidden`), needed only for the q4/q8
weight-precision sweep; the transformers bf16/fp16 path is the verified baseline behind all
the numbers above.

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