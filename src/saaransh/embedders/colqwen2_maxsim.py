"""Baseline — ColQwen2 exact MaxSim (full late interaction), no MUVERA.

This is the quality ceiling the single-vector pipelines are measured against. It
keeps the full multi-vector bag per page and scores with the native ColBERT-style
MaxSim via `processor.score_multi_vector` (the scorer from the user's Sep-2025
pipeline). Storage is the fat end of the comparison: ~n_patches x token_dim x 4
bytes per page (~0.5 MB), vs ~32 KB for a MUVERA FDE and ~15 KB for a Gemma vector.

Not a SingleVectorEmbedder — it never produces one vector — so the runner scores it
through `build_and_eval_maxsim`, not the flat index.
"""

from __future__ import annotations

import numpy as np

try:
    import torch  # noqa: F401
except Exception:  # pragma: no cover
    torch = None


class ColQwen2MaxSimRetriever:
    metric = "maxsim"

    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v1.0-hf",
        token_dim: int = 128,
        *,
        device: str | None = None,
        cache_dir: str | None = "./model_cache",
        local_files_only: bool = False,
        batch_size: int = 4,
    ) -> None:
        if torch is None:
            raise ImportError("Install the colqwen2 extra:  uv pip install -e '.[colqwen2]'")
        from saaransh.embedders.colqwen2_backbone import load_colqwen2

        self.name = "colqwen2-maxsim (baseline)"
        self.token_dim = token_dim
        self.batch_size = batch_size
        self.model, self.processor, self.device, _ = load_colqwen2(
            model_name, device=device, cache_dir=cache_dir, local_files_only=local_files_only
        )
        # Overridable for testing the rank/eval glue without the real model.
        self._score = self.processor.score_retrieval

    def _embed(self, items: list, kind: str) -> list:
        import torch

        out: list = []
        for i in range(0, len(items), self.batch_size):
            chunk = items[i : i + self.batch_size]
            if kind == "image":
                inputs = self.processor(images=chunk, return_tensors="pt").to(self.device)
            else:
                inputs = self.processor(text=list(chunk), return_tensors="pt").to(self.device)
            with torch.no_grad():
                emb = self.model(**inputs).embeddings
            mask = inputs["attention_mask"].bool()
            for e, m in zip(emb, mask, strict=True):
                out.append(e[m].float().cpu())  # unpadded (n_i, token_dim)
        return out

    def embed_images(self, images: list) -> list:
        return self._embed(images, "image")

    def embed_queries(self, queries: list[str]) -> list:
        return self._embed(queries, "query")

    def rank(self, query_mv: list, doc_mv: list, k: int) -> np.ndarray:
        scores = self._score(query_mv, doc_mv)  # (n_queries, n_docs)
        scores = np.asarray(scores.float().cpu() if hasattr(scores, "cpu") else scores)
        k = min(k, scores.shape[1])
        idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1)
        return np.take_along_axis(idx, order, axis=1)

    @staticmethod
    def bytes_per_doc(doc_mv: list, token_dim: int) -> float:
        avg_patches = float(np.mean([len(d) for d in doc_mv])) if doc_mv else 0.0
        return avg_patches * token_dim * 4.0
