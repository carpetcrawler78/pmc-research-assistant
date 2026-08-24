"""
lib/embedder.py
Capstone 1 — PMC Research Assistant  v2

SentenceTransformer wrapper that always returns L2-normalised float32 vectors
(required for cosine similarity via FAISS IndexFlatIP dot product).

Default model: pritamdeka/S-PubMedBert-MS-MARCO
  - PubMedBERT fine-tuned on MS-MARCO for sentence similarity
  - 768-dim output, 512-token limit
  - CPU-friendly; GPU used automatically if available

Public API
----------
Embedder(model_name, hf_token, batch_size)
  .encode(texts, show_progress) -> np.ndarray  shape (n, dim)
  .encode_single(text)          -> np.ndarray  shape (1, dim)
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Thin wrapper around SentenceTransformer with L2-normalisation."""

    def __init__(
        self,
        model_name: str,
        hf_token: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model      = SentenceTransformer(model_name, token=hf_token)
        self.batch_size = batch_size
        self.dim        = self.model.get_sentence_embedding_dimension()
        print(f"Embedder  : {model_name}")
        print(f"Device    : {self.model.device}   dim={self.dim}")

    def encode(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of strings into L2-normalised float32 vectors.

        Returns
        -------
        np.ndarray of shape (len(texts), dim)
        """
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode one string; returns shape (1, dim)."""
        return self.encode([text], show_progress=False)
