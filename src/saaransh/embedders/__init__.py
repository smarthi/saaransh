"""Embedder backends. Heavy model deps are imported lazily inside each module."""

from saaransh.embedders.base import SingleVectorEmbedder, l2_normalize

__all__ = ["SingleVectorEmbedder", "l2_normalize"]
