"""
build_faiss_index.py
Capstone 1 — PMC Research Assistant
Builds a FAISS index over abstracts using allenai-specter embeddings.

Input:  big_data/baseline/pickle/clean_articles.pkl
Output: big_data/faiss/specter.index
        big_data/faiss/specter_meta.pkl   (pmid, title, abstract, kwd_lst_cl)

Usage:
    python tools/build_faiss_index.py
"""

import pickle
import numpy as np
import os
from pathlib import Path
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
_HF_TOKEN = os.getenv("HF_TOKEN")

# ── Paths ────────────────────────────────────────────────────────────────────
CWD          = Path(__file__).parent.parent   # project root (tools/ is one level down)
BIG_DATA     = CWD / "big_data"
PICKLE_IN    = BIG_DATA / "baseline" / "pickle" / "clean_articles.pkl"
FAISS_DIR    = BIG_DATA / "faiss"
INDEX_OUT    = FAISS_DIR / "specter.index"
META_OUT     = FAISS_DIR / "specter_meta.pkl"

FAISS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading clean_articles.pkl ...")
df = pickle.load(open(PICKLE_IN, "rb")) if PICKLE_IN.suffix == ".pkl" else __import__("pandas").read_pickle(PICKLE_IN)
print(f"  {len(df)} articles loaded")

# Keep only rows with non-empty abstracts
df = df[df["abstract"].str.strip().str.len() > 20].reset_index(drop=True)
print(f"  {len(df)} articles after abstract filter")

abstracts = df["abstract"].tolist()

# ── Embed ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "allenai-specter"
print(f"Loading model: {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME, token=_HF_TOKEN)

print("Encoding abstracts (this takes a few minutes) ...")
embeddings = model.encode(
    abstracts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,   # cosine similarity via inner product
)
print(f"  Embeddings shape: {embeddings.shape}")

# ── Build FAISS index ─────────────────────────────────────────────────────────
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)   # Inner Product = cosine (since normalized)
index.add(embeddings.astype(np.float32))
print(f"  FAISS index: {index.ntotal} vectors, dim={dim}")

faiss.write_index(index, str(INDEX_OUT))
print(f"  Saved: {INDEX_OUT}")

# ── Save metadata ─────────────────────────────────────────────────────────────
meta_cols = [c for c in ["pmid", "title", "abstract", "kwd_lst_cl"] if c in df.columns]
meta = df[meta_cols].to_dict(orient="records")

with open(META_OUT, "wb") as f:
    pickle.dump(meta, f)
print(f"  Saved: {META_OUT}  ({len(meta)} records, cols: {meta_cols})")

# ── Quick sanity check ────────────────────────────────────────────────────────
print("\n── Sanity check ──")
query = model.encode(["CRISPR gene editing cancer"], normalize_embeddings=True)
D, I = index.search(query.astype(np.float32), k=3)
for rank, (score, idx) in enumerate(zip(D[0], I[0])):
    title = meta[idx].get("title", "n/a")[:80]
    print(f"  [{rank+1}] score={score:.3f}  {title}")

print("\nDone.")
