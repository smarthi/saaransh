#!/usr/bin/env bash
# Run the full gap-filling pipeline: 3 new ViDoRe datasets + precision sweep + failure analysis.
# Assumes: saaransh installed (uv pip install -e '.[colqwen2,gemma,mlx,faiss,vidore,plot]'),
# the cache.py/cli.py query_texts patch applied, and model checkpoints already downloaded
# (or --local-only removed below to let it fetch on first run).
#
# Usage: ./run_all.sh [--local-only]
set -uo pipefail  # not -e: one dataset's failure shouldn't kill the rest

LOCAL_ONLY="${1:-}"
MODEL_CACHE_DIR="./model_cache"
COLQWEN_MODEL="vidore/colqwen2-v1.0-hf"
GEMMA_MODEL="google/gemma-4-12b-it"
LIMIT=200
OUT_ROOT="./results"
mkdir -p "$OUT_ROOT"

DATASETS=(
  "docvqa:vidore/docvqa_test_subsampled"
  "arxivqa:vidore/arxivqa_test_subsampled"
  "infovqa:vidore/infovqa_test_subsampled"
  "tabfquad:vidore/tabfquad_test_subsampled"
)

for entry in "${DATASETS[@]}"; do
  name="${entry%%:*}"
  vidore="${entry#*:}"
  dir="$OUT_ROOT/$name"
  mkdir -p "$dir"
  echo "=== $name ($vidore) ==="

  # 1. Cache ColQwen2 bags once (includes the query_texts patch)
  saaransh-cache --vidore "$vidore" --limit "$LIMIT" \
    --colqwen-model "$COLQWEN_MODEL" --cache-dir "$MODEL_CACHE_DIR" $LOCAL_ONLY \
    --out "$dir/cache" \
    || { echo "!! cache failed for $name, skipping"; continue; }

  # 2. MUVERA frontier sweep + ceiling, same grid as the original docvqa run
  saaransh-muvera-sweep --cache "$dir/cache" --ceiling \
    --modes default_identity,calibrated_eigenbasis --k 4,6,8 --reps 4,8 \
    --compress none,32768,8192,2048 \
    --out-csv "$dir/muvera_sweep.csv" --plot "$dir/frontier.png" \
    || echo "!! sweep failed for $name"

  # 3. Failure analysis at the strongest frontier config
  python scripts/failure_analysis.py --cache "$dir/cache" \
    --mode calibrated_eigenbasis --k 8 --reps 8 \
    --out "$dir/failure_report.csv" \
    || echo "!! failure analysis failed for $name"
done

# 4. Gemma weight-precision sweep — only needs running once, reuses docvqa corpus
echo "=== gemma precision sweep (docvqa) ==="
saaransh-sweep --vidore vidore/docvqa_test_subsampled --limit "$LIMIT" \
  --gemma-model "$GEMMA_MODEL" --cache-dir "$MODEL_CACHE_DIR" --precisions bf16 $LOCAL_ONLY \
  --out "$OUT_ROOT/gemma_precision_sweep.csv" \
  || echo "!! gemma precision sweep failed"

echo "=== done — now run: python scripts/tabulate_results.py --root $OUT_ROOT ==="