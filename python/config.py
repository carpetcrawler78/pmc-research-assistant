"""
config.py
Capstone 1 — PMC Research Assistant  v2

Central configuration for the Plan-A pipeline.
All scripts import from here so that a single change propagates everywhere.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent

PICKLE_IN    = PROJECT_ROOT / "big_data" / "baseline" / "pickle" / "clean_articles_enriched.pkl"
TFIDF_MODEL  = PROJECT_ROOT / "models" / "baseline_v1_clf.pkl"

INDEX_DIR    = PROJECT_ROOT / "big_data" / "faiss_v2"
FAISS_INDEX  = INDEX_DIR / "pubmedbert.index"
FAISS_META   = INDEX_DIR / "pubmedbert_meta.pkl"
BM25_PATH    = INDEX_DIR / "bm25.pkl"

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBED_MODEL  = "pritamdeka/S-PubMedBert-MS-MARCO"   # domain-specific, 768-dim
EMBED_DIM    = 768
BATCH_SIZE   = 32

# ── Chunking ──────────────────────────────────────────────────────────────────
# Text fields concatenated per article before chunking.
# Swap this list when more sections become available (e.g. add "methods").
TEXT_FIELDS         = ["title", "abstract", "introduction"]
CHUNK_MAX_CHARS     = 1600   # ≈ 400 BERT tokens  (1 token ≈ 4 chars)
CHUNK_OVERLAP_CHARS = 200    # ≈  50 token overlap between adjacent chunks

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_FAISS    = 20    # dense candidates before dedup
TOP_K_BM25     = 20    # sparse candidates before dedup
TOP_K_OUTPUT   = 15    # final keyword suggestions shown to user
MIN_SCORE      = 0.45  # minimum cosine similarity (FAISS leg only)
RRF_K          = 60    # Reciprocal Rank Fusion constant

# ── RAG / LLM ─────────────────────────────────────────────────────────────────
# Groq (free, fast) is the default. Set GROQ_API_KEY in .env.
# Falls back to OpenAI if OPENAI_API_KEY is set instead.
LLM_PROVIDER       = "groq"                  # "groq" or "openai"
LLM_MODEL_GROQ     = "llama-3.3-70b-versatile"  # free on Groq
LLM_MODEL_OPENAI   = "gpt-4o-mini"
LLM_MODEL          = LLM_MODEL_GROQ if os.getenv("GROQ_API_KEY") else LLM_MODEL_OPENAI
TOP_K_TFIDF_DIRECT = 10    # direct TF-IDF predictions added to candidates

# ── Credentials ───────────────────────────────────────────────────────────────
HF_TOKEN    = os.getenv("HF_TOKEN")
OPENAI_KEY  = os.getenv("OPENAI_API_KEY")
GROQ_KEY    = os.getenv("GROQ_API_KEY")
