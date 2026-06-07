"""Pipeline 1 — ColQwen2 (multi-vector / late interaction) collapsed to a single
FDE vector with MUVERA (pymuvera).

A document page becomes ~1000 patch vectors in ColQwen2; MUVERA collapses that bag
to one fixed-dimensional encoding whose inner product approximates the MaxSim
(Chamfer) score. Query tokens are collapsed the same way (SUM aggregation), so
`q_fde @ d_fde` ranks documents without ever materializing the full MaxSim.

Token dimension gotcha (asserted below):
    ColQwen2     -> 128
    ColQwen3.5   -> 320   (using 128 silently truncates 60% of the representation)
"""

from __future__ import annotations

import numpy as np

from saaransh.embedders.base import unpad_multivectors

try:  # heavy deps optional
    import torch  # noqa: F401
except Exception:  # pragma: no cover
    torch = None

from pymuvera import MUVERAEncoder, ProjectionType


class ColQwen2MuveraEmbedder:
    metric = "ip"  # FDE inner product ~ Chamfer; do NOT normalize.

    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v1.0-hf",
        token_dim: int = 128,
        *,
        device: str | None = None,
        cache_dir: str | None = "./model_cache",
        local_files_only: bool = False,
        num_simhash_projections: int = 4,
        num_repetitions: int = 4,
        projection_type: ProjectionType = ProjectionType.DEFAULT_IDENTITY,
        final_projection_dimension: int | None = None,
        seed: int = 42,
        batch_size: int = 4,
    ) -> None:
        if torch is None:
            raise ImportError("Install the colqwen2 extra:  uv pip install -e '.[colqwen2]'")
        from saaransh.embedders.colqwen2_backbone import load_colqwen2

        self.name = f"colqwen2+muvera[{projection_type.name}]"
        self.token_dim = token_dim
        self.batch_size = batch_size
        self.model, self.processor, self.device, _ = load_colqwen2(
            model_name, device=device, cache_dir=cache_dir, local_files_only=local_files_only
        )
        self.encoder = MUVERAEncoder(
            dimension=token_dim,
            num_simhash_projections=num_simhash_projections,
            num_repetitions=num_repetitions,
            projection_type=projection_type,
            final_projection_dimension=final_projection_dimension,
            fill_empty_partitions=True,
            seed=seed,
        )
        self.dim = self.encoder.fde_dimension
        self._needs_calibration = projection_type == ProjectionType.CALIBRATED_EIGENBASIS

    def _check_dim(self, got: int) -> None:
        if got != self.token_dim:
            raise ValueError(
                f"ColQwen token dim is {got} but MUVERA was built for {self.token_dim}. "
                f"ColQwen3.5 emits 320-d tokens; using 128 truncates ~60% of the representation. "
                f"Pass token_dim={got}."
            )

    def _embed_raw(self, items: list, kind: str) -> list[np.ndarray]:
        import torch

        bags: list[np.ndarray] = []
        for i in range(0, len(items), self.batch_size):
            chunk = items[i : i + self.batch_size]
            if kind == "image":
                inputs = self.processor(images=chunk, return_tensors="pt").to(self.device)
            else:
                inputs = self.processor(text=list(chunk), return_tensors="pt").to(self.device)
            with torch.no_grad():
                emb = self.model(**inputs).embeddings  # (B, T, 128)
            self._check_dim(emb.shape[-1])
            bags += unpad_multivectors(
                emb.float().cpu().numpy(), inputs["attention_mask"].cpu().numpy()
            )
        return bags

    def calibrate(self, sample_images: list) -> None:
        if not self._needs_calibration:
            return
        patches = np.concatenate(self._embed_raw(sample_images, "image"), axis=0)
        self.encoder.calibrate(patches)
        self.dim = self.encoder.fde_dimension

    def embed_images(self, images: list) -> np.ndarray:
        bags = self._embed_raw(images, "image")
        return np.stack([self.encoder.encode_document(b) for b in bags]).astype("float32")

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        bags = self._embed_raw(queries, "query")
        return np.stack([self.encoder.encode_query(b) for b in bags]).astype("float32")
