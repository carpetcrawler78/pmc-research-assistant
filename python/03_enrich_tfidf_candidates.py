"""
03_enrich_tfidf_candidates.py
Capstone 1 — PMC Research Assistant

STAGE 3: TF-IDF candidate keywords — enrich DataFrame with top-k keyword
suggestions from the trained baseline model, then build the FAISS index.

Reads clean_articles_enriched.pkl (output of stage 2), loads the trained
TF-IDF + LSA + LinearSVC model bundle, adds tfidf_candidates per article,
and builds the FAISS semantic index using allenai-specter embeddings.

Input:  big_data/baseline/pickle/clean_articles_enriched.pkl
        models/baseline_v1_clf.pkl
Output: big_data/baseline/pickle/clean_articles_enriched.pkl  (updated)
        big_data/faiss/specter.index
        big_data/faiss/specter_meta.pkl

Usage:
    python python/03_enrich_tfidf_candidates.py
"""

import pickle
import numpy as np
import pandas as pd
import faiss
import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()
_HF_TOKEN = os.getenv("HF_TOKEN")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
PICKLE_IN     = PROJECT_ROOT / "big_data" / "baseline" / "pickle" / "clean_articles_enriched.pkl"
MODEL_PATH    = PROJECT_ROOT / "models" / "baseline_v1_clf.pkl"
FAISS_DIR     = PROJECT_ROOT / "big_data" / "faiss"
FAISS_INDEX   = FAISS_DIR / "specter.index"
FAISS_META    = FAISS_DIR / "specter_meta.pkl"

# ── Config ────────────────────────────────────────────────────────────────────
TOP_K_CANDIDATES = 10    # TF-IDF top-k keywords stored per article
EMBED_MODEL      = "allenai-specter"
BATCH_SIZE       = 64


# ─────────────────────────────────────────────────────────────────────────────
# PART A — TF-IDF candidates
# ─────────────────────────────────────────────────────────────────────────────

def add_tfidf_candidates(df: pd.DataFrame, model_path: Path, top_k: int) -> pd.DataFrame:
    """
    Use trained TF-IDF + LSA + LinearSVC model to predict top-k keyword
    candidates per article. Stored as a list of strings in tfidf_candidates.
    Falls back to empty list if model file is missing.
    """
    if not model_path.exists():
        print(f"  ⚠  Model not found at {model_path} — skipping TF-IDF candidates.")
        print("     Train baseline_II_v1 first and save as models/baseline_v1_clf.pkl")
        df["tfidf_candidates"] = [[] for _ in range(len(df))]
        return df

    print(f"Loading baseline model from {model_path.name} ...")
    bundle = pickle.load(open(model_path, "rb"))
    tfidf = bundle["tfidf"]
    svd   = bundle["svd"]
    clf   = bundle["clf"]
    mlb   = bundle["mlb"]

    # Build input text (same as training: title + abstract + introduction)
    text = (
        df["title"].fillna("") + " " +
        df["abstract"].fillna("") + " " +
        df["introduction"].fillna("")
    )

    print(f"Transforming {len(df)} articles with TF-IDF + LSA ...")
    X     = tfidf.transform(text)
    X_lsa = svd.transform(X)

    print(f"Predicting top-{top_k} keyword candidates ...")
    proba  = clf.predict_proba(X_lsa)                  # shape: (n_articles, n_classes)
    top_idx = np.argsort(proba, axis=1)[:, -top_k:][:, ::-1]  # highest prob first
    candidates = [mlb.classes_[idx].tolist() for idx in top_idx]

    df["tfidf_candidates"] = candidates
    print(f"  ✓ tfidf_candidates added ({top_k} per article)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PART B — FAISS index
# ─────────────────────────────────────────────────────────────────────────────

def build_faiss_index(df: pd.DataFrame, model_name: str, batch_size: int):
    """
    Embed title + abstract with allenai-specter and build a FAISS IndexFlatIP
    (cosine similarity via L2-normalised vectors).
    """
    print(f"\nLoading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name, token=_HF_TOKEN)

    # specter was trained on "title [SEP] abstract"
    texts = [
        f"{row.get('title', '')} [SEP] {row.get('abstract', '')}"
        for _, row in df.iterrows()
    ]

    print(f"Embedding {len(texts)} articles (batch_size={batch_size}) ...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 norm → IndexFlatIP = cosine similarity
    )

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    print(f"  ✓ FAISS index built — {index.ntotal} vectors, dim={dim}")

    # Metadata for retrieval
    meta = df[["title", "abstract", "kwd_lst_cl",
               "cluster_label", "cluster_name",
               "tfidf_candidates", "journal", "year"]].to_dict("records")

    return index, meta


# ─────────────────────────────────────────────────────────────────────────────
# PART C — Sanity check
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check(index, meta, model_name: str):
    """Query the index with a test abstract and print top-3 results."""
    print("\n── Sanity check ──")
    model  = SentenceTransformer(model_name, token=_HF_TOKEN)
    query  = "LRRK2 mutations and their downstream effects on dopamine neurons in Parkinson disease"
    q_vec  = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(q_vec, k=3)

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        m = meta[idx]
        print(f"\n  #{rank}  score={score:.3f}")
        print(f"  Title:   {m.get('title', 'N/A')[:80]}")
        print(f"  Cluster: {m.get('cluster_name', 'N/A')}")
        print(f"  Keywords:{m.get('kwd_lst_cl', [])[:5]}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Load enriched DataFrame (must have cluster_label from stage 2)
    print(f"Loading {PICKLE_IN.name} ...")
    df = pickle.load(open(PICKLE_IN, "rb"))
    print(f"  {len(df)} articles, columns: {df.columns.tolist()}")

    # Part A — TF-IDF candidates
    df = add_tfidf_candidates(df, MODEL_PATH, TOP_K_CANDIDATES)

    # Part B — FAISS index
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    index, meta = build_faiss_index(df, EMBED_MODEL, BATCH_SIZE)

    # Save updated DataFrame
    with open(PICKLE_IN, "wb") as f:
        pickle.dump(df, f)
    print(f"\n✓ Updated DataFrame saved → {PICKLE_IN}")

    # Save FAISS index + metadata
    faiss.write_index(index, str(FAISS_INDEX))
    with open(FAISS_META, "wb") as f:
        pickle.dump(meta, f)
    print(f"✓ FAISS index saved      → {FAISS_INDEX}")
    print(f"✓ FAISS metadata saved   → {FAISS_META}")

    # Part C — Sanity check
    sanity_check(index, meta, EMBED_MODEL)

    print("\n✓ Stage 3 complete — ready for RAG pipeline.")


if __name__ == "__main__":
    main()
