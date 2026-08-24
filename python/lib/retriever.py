"""
lib/retriever.py
Capstone 1 — PMC Research Assistant  v2

Hybrid FAISS (dense) + BM25 (sparse) retrieval with Reciprocal Rank Fusion.
Chunk-level hits are deduplicated to article level before returning.

Public API
----------
load_index(faiss_path, meta_path, bm25_path) -> (index, meta, bm25)

retrieve(query_text, query_vec, index, meta, bm25,
         top_k_faiss, top_k_bm25, top_k_output, min_score, rrf_k)
    -> list[dict]   one dict per unique article, sorted by RRF score
"""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_index(faiss_path: Path, meta_path: Path, bm25_path: Path):
    """Load FAISS index, meta list, and BM25 model from disk."""
    index = faiss.read_index(str(faiss_path))
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    print(f"Index loaded : {index.ntotal} vectors | {len(meta)} chunks")
    return index, meta, bm25


# ── Search legs ───────────────────────────────────────────────────────────────

def _faiss_search(
    index,
    query_vec: np.ndarray,
    top_k: int,
    min_score: float,
) -> list[tuple[int, float]]:
    """Return (chunk_idx, cosine_score) pairs above min_score, best-first."""
    scores, indices = index.search(query_vec.reshape(1, -1), top_k)
    return [
        (int(i), float(s))
        for i, s in zip(indices[0], scores[0])
        if i >= 0 and s >= min_score
    ]


def _bm25_search(
    bm25,
    query: str,
    top_k: int,
) -> list[tuple[int, float]]:
    """Return (chunk_idx, bm25_score) pairs with score > 0, best-first."""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]


# ── Fusion ────────────────────────────────────────────────────────────────────

def _rrf_merge(
    faiss_hits: list[tuple[int, float]],
    bm25_hits:  list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion:
        rrf_score(chunk) = 1/(k + rank_faiss) + 1/(k + rank_bm25)
    Returns merged list sorted by rrf_score descending.
    """
    scores: dict[int, float] = {}
    for rank, (idx, _) in enumerate(faiss_hits):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, (idx, _) in enumerate(bm25_hits):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Deduplication ─────────────────────────────────────────────────────────────

def _dedup_articles(
    hits: list[tuple[int, float]],
    meta: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Collapse chunk-level hits to unique articles.
    Keeps the highest RRF score per pmcid.
    Returns up to top_k article dicts with 'retrieval_score' attached.
    """
    seen: dict[str, dict] = {}
    for idx, score in hits:
        entry = meta[idx]
        pmcid = entry["pmcid"]
        if pmcid not in seen or score > seen[pmcid]["retrieval_score"]:
            seen[pmcid] = {**entry, "retrieval_score": score}
    ranked = sorted(seen.values(), key=lambda x: x["retrieval_score"], reverse=True)
    return ranked[:top_k]


# ── Main entry point ──────────────────────────────────────────────────────────

def retrieve(
    query_text: str,
    query_vec:  np.ndarray,
    index,
    meta:       list[dict],
    bm25,
    top_k_faiss:  int   = 20,
    top_k_bm25:   int   = 20,
    top_k_output: int   = 10,
    min_score:    float = 0.45,
    rrf_k:        int   = 60,
) -> list[dict]:
    """
    Hybrid retrieval: FAISS dense search + BM25 sparse search, fused via RRF.

    Parameters
    ----------
    query_text  : raw query string (used by BM25)
    query_vec   : L2-normalised embedding of the query (used by FAISS)

    Returns
    -------
    List of article-level dicts sorted by RRF score, each containing:
      pmcid, title, cluster_name, kwd_lst_cl, tfidf_candidates,
      chunk_idx, chunk_text, retrieval_score
    """
    faiss_hits = _faiss_search(index, query_vec, top_k_faiss, min_score)
    bm25_hits  = _bm25_search(bm25, query_text, top_k_bm25)
    merged     = _rrf_merge(faiss_hits, bm25_hits, k=rrf_k)
    return _dedup_articles(merged, meta, top_k_output)
