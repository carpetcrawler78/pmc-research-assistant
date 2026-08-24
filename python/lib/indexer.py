"""
lib/indexer.py
Capstone 1 — PMC Research Assistant  v2

Build and persist FAISS IndexFlatIP + BM25Okapi from a DataFrame of articles.

Each article is split into chunks (via lib.chunker); each chunk becomes one
FAISS vector. The meta list maps every vector index back to its source article
so that retrieval can aggregate keywords at article level.

Public API
----------
build_index(df, embedder, text_fields, chunk_max_chars, chunk_overlap_chars,
            index_dir, faiss_index_path, faiss_meta_path, bm25_path)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from .chunker import build_text, chunk_text
from .embedder import Embedder


def build_index(
    df: pd.DataFrame,
    embedder: Embedder,
    text_fields: list[str],
    chunk_max_chars: int,
    chunk_overlap_chars: int,
    index_dir: Path,
    faiss_index_path: Path,
    faiss_meta_path: Path,
    bm25_path: Path,
) -> None:
    """
    Chunk every article, embed all chunks, build FAISS + BM25, save to disk.

    Meta structure (one entry per chunk)
    -------------------------------------
    {
        pmcid            : str
        title            : str
        cluster_name     : str
        kwd_lst_cl       : list[str]   — cleaned ground-truth keywords
        tfidf_candidates : list[str]   — baseline-predicted candidates
        chunk_idx        : int         — chunk position within the article
        chunk_text       : str         — raw chunk content
    }
    """
    index_dir.mkdir(parents=True, exist_ok=True)

    all_meta:    list[dict]       = []
    bm25_corpus: list[list[str]] = []

    print(f"Chunking {len(df)} articles  fields={text_fields} ...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        full_text = build_text(row.to_dict(), text_fields)
        chunks    = chunk_text(
            full_text,
            max_chars=chunk_max_chars,
            overlap_chars=chunk_overlap_chars,
        )
        for i, chunk in enumerate(chunks):
            all_meta.append({
                "pmcid":            str(row.get("pmcid") or ""),
                "title":            str(row.get("title") or ""),
                "cluster_name":     str(row.get("cluster_name") or ""),
                "kwd_lst_cl":       list(row.get("kwd_lst_cl") or []),
                "tfidf_candidates": list(row.get("tfidf_candidates") or []),
                "chunk_idx":        i,
                "chunk_text":       chunk,
            })
            bm25_corpus.append(chunk.lower().split())

    n_chunks = len(all_meta)
    avg      = n_chunks / max(len(df), 1)
    print(f"Total chunks : {n_chunks}  (avg {avg:.1f} per article)")

    # ── Embed ─────────────────────────────────────────────────────────────────
    chunk_texts = [m["chunk_text"] for m in all_meta]
    vecs        = embedder.encode(chunk_texts)          # (n_chunks, dim) float32

    # ── FAISS ─────────────────────────────────────────────────────────────────
    dim   = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, str(faiss_index_path))
    print(f"FAISS saved  : {faiss_index_path}  ({index.ntotal} vectors, dim={dim})")

    # ── Meta ──────────────────────────────────────────────────────────────────
    with open(faiss_meta_path, "wb") as f:
        pickle.dump(all_meta, f)
    print(f"Meta saved   : {faiss_meta_path}")

    # ── BM25 ──────────────────────────────────────────────────────────────────
    bm25 = BM25Okapi(bm25_corpus)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 saved   : {bm25_path}")
