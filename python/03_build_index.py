"""
03_build_index.py
Capstone 1 — PMC Research Assistant  v2

STAGE 3b — Build FAISS + BM25 index (Plan A).

Reads clean_articles_enriched.pkl, concatenates title + abstract + introduction
per article, chunks via RecursiveCharacterSplitter, embeds with
S-PubMedBERT-MS-MARCO, and saves:
  - big_data/faiss_v2/pubmedbert.index   FAISS IndexFlatIP
  - big_data/faiss_v2/pubmedbert_meta.pkl
  - big_data/faiss_v2/bm25.pkl           BM25Okapi

To swap embedding model or text fields, edit python/config.py only.

Usage:
    python python/03_build_index.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pickle

import faiss
import pandas as pd

from config import (
    BATCH_SIZE,
    BM25_PATH,
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP_CHARS,
    EMBED_MODEL,
    FAISS_INDEX,
    FAISS_META,
    HF_TOKEN,
    INDEX_DIR,
    PICKLE_IN,
    PROJECT_ROOT,
    TEXT_FIELDS,
)
from lib.embedder import Embedder
from lib.indexer import build_index
from lib.plotter import plot_chunk_stats, plot_embedding_2d

PLOTS_DIR = PROJECT_ROOT / "plots"

if __name__ == "__main__":

    if not PICKLE_IN.exists():
        print(f"ERROR: input pickle not found: {PICKLE_IN}")
        print("Run python/02_enrich_cluster_labels.py first.")
        sys.exit(1)

    print(f"Loading {PICKLE_IN.name} ...")
    df = pd.read_pickle(PICKLE_IN)
    print(f"  {len(df)} articles,  columns: {list(df.columns)}")

    # Warn if requested text fields are missing
    missing = [f for f in TEXT_FIELDS if f not in df.columns]
    if missing:
        print(f"  WARNING: columns not found in pickle, will be skipped: {missing}")

    embedder = Embedder(EMBED_MODEL, hf_token=HF_TOKEN, batch_size=BATCH_SIZE)

    build_index(
        df=df,
        embedder=embedder,
        text_fields=TEXT_FIELDS,
        chunk_max_chars=CHUNK_MAX_CHARS,
        chunk_overlap_chars=CHUNK_OVERLAP_CHARS,
        index_dir=INDEX_DIR,
        faiss_index_path=FAISS_INDEX,
        faiss_meta_path=FAISS_META,
        bm25_path=BM25_PATH,
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    with open(FAISS_META, "rb") as f:
        meta = pickle.load(f)

    plot_chunk_stats(meta, PLOTS_DIR)

    index = faiss.read_index(str(FAISS_INDEX))
    plot_embedding_2d(index, meta, PLOTS_DIR, sample_n=800)

    print(f"\nDone. Plots saved to {PLOTS_DIR}")
    print("Continue with:  python python/04_rag_pipeline.py")
