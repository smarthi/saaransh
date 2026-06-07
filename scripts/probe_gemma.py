#!/usr/bin/env python
"""Probe Gemma 4 12B's embedding contract — run once on your Mac, paste the output.

Loads Gemma 4 ONLY (no ColQwen2, so it fits bf16 in 32 GB), pushes one image and one
text query through, and prints exactly what saaransh's Gemma embedder relies on:
  - which model class loaded
  - where hidden_size lives on the config
  - the apply_chat_template input keys/shapes (does image packing work?)
  - hidden_states availability + last-layer shape (VERIFY #1 + the pooling contract)
  - the pooled vector shape / norm

Usage:
  python probe_gemma.py --image /path/to/page.png
  python probe_gemma.py --image page.png --model google/gemma-4-12b-it --cache-dir ./model_cache
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-12b-it",
                    help="repo id OR a local folder (e.g. ./model_cache/gemma-4-12b-it from --local-dir)")
    ap.add_argument("--image", required=True, help="any page image (png/jpg)")
    ap.add_argument("--cache-dir", default=None, help="HF hub cache dir (only for repo-id loads)")
    ap.add_argument("--offline", action="store_true", help="forbid any network fetch (HF_HUB_OFFLINE=1)")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp16"])
    args = ap.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    import transformers
    from PIL import Image

    is_local = os.path.isdir(args.model)
    print(f"loading from: {args.model}  ({'local dir' if is_local else 'repo id / hub cache'})"
          f"{'  [offline]' if args.offline else ''}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    print(f"transformers {transformers.__version__} | device={device} | dtype={dtype}")

    # ── load (try the documented class, then the fallback) ──────────────
    model = loaded = None
    for cls_name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            print(f"  {cls_name}: not present in this transformers")
            continue
        try:
            model = cls.from_pretrained(
                args.model, dtype=dtype, device_map=device, cache_dir=args.cache_dir
            ).eval()
            loaded = cls_name
            break
        except Exception as e:
            print(f"  {cls_name} failed: {type(e).__name__}: {str(e)[:160]}")
    if model is None:
        print("!! No multimodal class loaded — check transformers version / model files.")
        return 1
    print(f"LOADED via {loaded}")

    cfg = model.config
    hs = getattr(cfg, "hidden_size", None)
    hs_loc = "config.hidden_size"
    if not hs and getattr(cfg, "text_config", None) is not None:
        hs = getattr(cfg.text_config, "hidden_size", None)
        hs_loc = "config.text_config.hidden_size"
    print(f"hidden_size = {hs}  (at {hs_loc})")
    print(f"config type = {type(cfg).__name__}")

    processor = transformers.AutoProcessor.from_pretrained(args.model, cache_dir=args.cache_dir)
    img = Image.open(args.image).convert("RGB")

    def run(content, label):
        msgs = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=False, enable_thinking=False,
        ).to(device)
        print(f"\n[{label}] input keys: {sorted(inputs.keys())}")
        for k, v in inputs.items():
            if hasattr(v, "shape"):
                print(f"    {k}: {tuple(v.shape)} {v.dtype}")
        if "mm_token_type_ids" in inputs:
            mt = inputs["mm_token_type_ids"][0]
            vals, counts = mt.unique(return_counts=True)
            print(f"    mm_token_type_ids values: {vals.tolist()} counts: {counts.tolist()}"
                  f"  (image marker = the value whose count ~= visual budget)")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        has_hs = getattr(out, "hidden_states", None) is not None
        print(f"    output_hidden_states present: {has_hs}")
        if has_hs:
            last = out.hidden_states[-1]
            print(f"    n_layers(hidden_states)={len(out.hidden_states)}  last={tuple(last.shape)}")
            mask = inputs["attention_mask"][0].bool()
            pooled = last[0][mask].mean(dim=0).float().cpu()
            print(f"    pooled(mean) dim={pooled.shape[0]}  L2={pooled.norm().item():.3f}")
        return has_hs

    ok_img = run([{"type": "image", "image": img},
                  {"type": "text", "text": "Represent this document image for retrieval."}], "IMAGE")
    ok_qry = run([{"type": "text", "text": "Represent this query: what is shown in the document?"}], "QUERY")

    print("\n=== SUMMARY ===")
    print(f"class={loaded}  hidden_size={hs} ({hs_loc})  image_hs={ok_img}  query_hs={ok_qry}")
    print("Paste everything above back and I'll lock gemma4_pooled.py to match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
