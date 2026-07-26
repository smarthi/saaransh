#!/usr/bin/env python
"""Probe Gemma 4 12B's MLX (q8/q4) hidden-state contract — run once on your Mac.

Mirrors probe_gemma.py, but for the MLX backend. The transformers path was
confirmed this way before being trusted; this script exists so the MLX
`return_hidden=True` wiring in gemma4_pooled.py gets the same treatment before
it goes into the precision sweep. Loads ONLY the quantized MLX checkpoint (much
smaller than bf16, should comfortably fit even alongside other work), pushes
one image and one text query through, and prints:
  - which mlx-vlm repo/checkpoint actually loaded
  - the prepare_inputs() key/shapes (does image packing work the same as HF?)
  - whether return_hidden=True actually returns something in .hidden_states
  - the pooled vector shape / norm, so it can be sanity-compared against the
    transformers bf16 pooled vector for the SAME image+query (they won't be
    identical — different precision, different backend — but should be in a
    similar ballpark, not degenerate/all-zero/NaN)

Usage:
  python scripts/probe_gemma_mlx.py --image /path/to/page.png --precision q8
  python scripts/probe_gemma_mlx.py --image page.png --precision q4 \
      --model mlx-community/gemma-4-12B-4bit
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="mlx-community repo id; default is derived from --precision")
    ap.add_argument("--image", required=True, help="any page image (png/jpg)")
    ap.add_argument("--precision", default="q8", choices=["q8", "q4"])
    args = ap.parse_args()

    try:
        import mlx.core as mx
        from mlx_vlm import load
        from mlx_vlm.utils import prepare_inputs
    except Exception as e:
        print(f"!! mlx / mlx_vlm not importable: {e}")
        print("   Install the mlx extra on a Mac:  uv pip install -e '.[mlx]'")
        return 1

    from PIL import Image

    bits = 8 if args.precision == "q8" else 4
    repo = args.model or f"mlx-community/gemma-4-12B-{bits}bit"
    print(f"loading: {repo}")
    model, processor = load(repo)
    print(f"loaded. model type: {type(model).__name__}")
    print(f"hidden_size (config): {getattr(model.config, 'hidden_size', 'NOT FOUND')}")

    img = Image.open(args.image).convert("RGB")
    img_prompt = "Represent this document image for retrieval."
    qry_prompt = "Represent this query for retrieving a matching document image: what is shown in the document?"

    def run(label, images, prompt):
        inputs = prepare_inputs(processor, images=images, prompts=[prompt])
        print(f"\n[{label}] prepare_inputs() keys: {sorted(inputs.keys())}")
        for k, v in inputs.items():
            if hasattr(v, "shape"):
                print(f"    {k}: {tuple(v.shape)} {v.dtype}")
        mask = inputs.pop("attention_mask", None)
        out = model(mask=mask, return_hidden=True, **inputs)
        has_hs = getattr(out, "hidden_states", None) is not None and len(out.hidden_states) > 0
        print(f"    return_hidden actually populated hidden_states: {has_hs}")
        if not has_hs:
            print("    !! nothing captured — check capture_layer_ids / hidden_sink wiring "
                  "against your installed mlx-vlm version, this probe's assumptions may be stale.")
            return None
        hidden = out.hidden_states[-1]
        print(f"    hidden_states[-1] shape: {tuple(hidden.shape)} dtype={hidden.dtype}")
        if mask is not None:
            m = mask[0].astype(hidden.dtype)
            pooled = (hidden[0] * m[:, None]).sum(axis=0) / mx.maximum(m.sum(), 1)
        else:
            pooled = hidden[0].mean(axis=0)
        pooled_np = pooled.astype(mx.float32)
        norm = float(mx.sqrt(mx.sum(pooled_np * pooled_np)))
        has_nan = bool(mx.any(pooled_np != pooled_np))  # NaN != NaN trick, no isnan needed
        print(f"    pooled dim={pooled.shape[0]}  L2={norm:.3f}  any_nan={has_nan}")
        return norm, has_nan

    print("=== IMAGE ===")
    img_result = run("IMAGE", [img], img_prompt)
    print("\n=== QUERY (text-only) ===")
    qry_result = run("QUERY", None, qry_prompt)

    print("\n=== SUMMARY ===")
    ok = img_result is not None and qry_result is not None
    if ok:
        img_norm, img_nan = img_result
        qry_norm, qry_nan = qry_result
        healthy = not img_nan and not qry_nan and img_norm > 0 and qry_norm > 0
        print(f"image_norm={img_norm:.3f}  query_norm={qry_norm:.3f}  "
              f"looks healthy (nonzero, no NaN)={healthy}")
        if not healthy:
            print("!! degenerate output — do not trust this in the precision sweep yet, "
                  "paste this full output back before proceeding.")
    else:
        print("!! return_hidden wiring did not work as expected — paste this full output back "
              "so gemma4_pooled.py's _mlx_hidden can be corrected against what your mlx-vlm "
              "version actually does.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())