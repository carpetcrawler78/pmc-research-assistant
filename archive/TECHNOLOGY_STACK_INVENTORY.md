# PMC Research Assistant — Technology Stack Inventory

**Generated:** 2026-03-26
**Scope:** All `.py` and `.ipynb` files across all Capstone branches
**Search Terms:** pubmedBERT, FAISS, BM25, RRF, LLM models

---

## 1. Embedding Model (Retrieval)

### Active Model
- **Model Name:** `pritamdeka/S-PubMedBert-MS-MARCO`
- **Dimension:** 768-dim
- **Framework:** Hugging Face Transformers
- **Provider:** Hugging Face Hub (`pritamdeka/`)
- **Purpose:** Dense vector embeddings for semantic retrieval
- **Config Location:** `python/config.py:28`

### Alternative / Historical
- **SPECTER2** (768-dim, academic-focused embeddings)
  - Used in older branches / notebooks
  - Located in: `notebooks/specter_embeddings_kaggle.ipynb`
  - Status: Replaced by S-PubMedBERT

---

## 2. Dense Index (FAISS)

### Primary Configuration
- **Index Type:** FAISS IVFFlat
- **Similarity Metric:** Cosine similarity
- **Minimum Score Threshold:** 0.45
- **Top-K Retrieval:** 20 candidates (before dedup)
- **Index Files:**
  - `big_data/faiss_v2/pubmedbert.index` — Vector index
  - `big_data/faiss_v2/pubmedbert_meta.pkl` — Metadata (46,853 chunks, 7,702 articles)
- **Build Script:** `python/build_faiss_index.py`
- **Build Notebook:** `.claude/worktrees/happy-wescoff/notebooks/05_build_faiss_index.ipynb`
- **Chunking Strategy:**
  - Max chunk size: 1,600 characters (≈ 400 BERT tokens)
  - Overlap: 200 characters (≈ 50 token overlap)
  - Text fields: `["title", "abstract", "introduction"]`
- **Status:** ✓ Production (Note: FAISS vectors based on embeddings from contaminated chunks; Re-embedding queued for next session)

---

## 3. Sparse Index (BM25)

### Configuration
- **Algorithm:** BM25 (Okapi BM25 variant)
- **Library:** `rank_bm25` (Python library)
- **Top-K Retrieval:** 20 candidates (before fusion)
- **Index File:** `big_data/faiss_v2/bm25.pkl`
- **Used In:**
  - `eval_v1_bm25only.py` — Baseline (BM25-only retrieval)
  - `eval_v2_hybrid_8b.py` — Hybrid leg (combined with FAISS)
  - `eval_v3_hybrid_70b.py` — Hybrid leg
- **Status:** ✓ Active

### Files Using BM25
| File | Context |
|------|---------|
| `python/04_rag_pipeline.py` | Primary RAG pipeline |
| `python/lib/retriever.py` | Retrieval logic |
| `python/lib/indexer.py` | Index building |
| `python/03_build_index.py` | Index construction |
| `python/03_enrich_tfidf_candidates.py` | TF-IDF enrichment |

---

## 4. Retrieval Fusion (RRF)

### Reciprocal Rank Fusion
- **Fusion Method:** RRF (Reciprocal Rank Fusion)
- **RRF Constant (k):** 60
- **Formula:** Score = Σ 1 / (k + rank)
- **Combining:** BM25 sparse rankings + FAISS dense rankings
- **Output:** Top-15 papers (final dedup)
- **Implemented In:** `python/lib/retriever.py`
- **Used By:** V2 (Hybrid 8B) and V3 (Hybrid 70B) evaluation variants

### RRF Configuration Files
| File | Line | Notes |
|------|------|-------|
| `python/config.py` | 44 | `RRF_K = 60` |
| `python/lib/retriever.py` | - | RRF fusion logic |
| `python/04_rag_pipeline.py` | - | Pipeline orchestration |

---

## 5. Language Models (LLM)

### Groq-Hosted Models

#### V1 Baseline (BM25-only retrieval)
- **Model:** `llama-3.1-8b-instant`
- **Provider:** Groq API (free tier)
- **Temperature:** Default (≈0.7)
- **Max Tokens:** 800
- **Script:** `python/eval_v1_bm25only.py`
- **Use Case:** Baseline for retrieval comparison

#### V2 Hybrid Retrieval (Fair model comparison)
- **Model:** `llama-3.1-8b-instant`
- **Provider:** Groq API (free tier)
- **Retrieval:** Hybrid (BM25 + FAISS + RRF)
- **Max Tokens:** 800
- **Script:** `python/eval_v2_hybrid_8b.py`
- **Use Case:** Isolate retrieval improvements (same LLM as V1)

#### V3 Production (Stronger reasoning)
- **Model:** `llama-3.3-70b-versatile`
- **Provider:** Groq API (free tier — production model)
- **Retrieval:** Hybrid (BM25 + FAISS + RRF)
- **Max Tokens:** 1,200
- **Script:** `python/eval_v3_hybrid_70b.py`
- **Use Case:** Production-grade QA with stronger reasoning

### OpenAI Fallback
- **Model:** `gpt-4o-mini`
- **Provider:** OpenAI API
- **Trigger:** When `GROQ_API_KEY` not set; `OPENAI_API_KEY` available
- **Config:** `python/config.py:51-52`
- **Status:** Fallback only (not used in main evaluations)

### RAG LLM Configuration
```python
# From python/config.py
LLM_PROVIDER       = "groq"
LLM_MODEL_GROQ     = "llama-3.3-70b-versatile"
LLM_MODEL_OPENAI   = "gpt-4o-mini"
LLM_MODEL          = LLM_MODEL_GROQ if GROQ_KEY else LLM_MODEL_OPENAI
```

---

## 6. Files by Technology

### pubmedBERT / Embeddings
| File | Type | Purpose |
|------|------|---------|
| `python/lib/embedder.py` | .py | Embedding generation & caching |
| `python/config.py` | .py | Model config |
| `python/03_build_index.py` | .py | Index building with embeddings |
| `notebooks/specter_embeddings_kaggle.ipynb` | .ipynb | Historical SPECTER2 embeddings |
| `.claude/worktrees/happy-wescoff/notebooks/04_clustering_semantic.ipynb` | .ipynb | Semantic clustering |
| `.claude/worktrees/happy-wescoff/notebooks/05_build_faiss_index.ipynb` | .ipynb | FAISS index construction |

### FAISS
| File | Type | Purpose |
|------|------|---------|
| `python/build_faiss_index.py` | .py | Index builder |
| `python/lib/indexer.py` | .py | Indexing logic |
| `python/04_rag_pipeline.py` | .py | Pipeline integration |
| `python/lib/retriever.py` | .py | FAISS retrieval |
| `.claude/worktrees/happy-wescoff/notebooks/05_build_faiss_index.ipynb` | .ipynb | Build notebook |

### BM25
| File | Type | Purpose |
|------|------|---------|
| `python/lib/retriever.py` | .py | BM25 + FAISS + RRF logic |
| `python/lib/indexer.py` | .py | BM25 index building |
| `python/04_rag_pipeline.py` | .py | Pipeline integration |
| `python/03_enrich_tfidf_candidates.py` | .py | TF-IDF enrichment |
| `python/eval_v1_bm25only.py` | .py | BM25-only baseline |
| `python/eval_v2_hybrid_8b.py` | .py | BM25 + FAISS hybrid |
| `python/eval_v3_hybrid_70b.py` | .py | BM25 + FAISS hybrid |

### RRF (Reciprocal Rank Fusion)
| File | Type | Purpose |
|------|------|---------|
| `python/lib/retriever.py` | .py | RRF implementation |
| `python/config.py` | .py | RRF_K constant (60) |
| `python/04_rag_pipeline.py` | .py | Pipeline orchestration |
| `python/eval_v2_hybrid_8b.py` | .py | Uses RRF fusion |
| `python/eval_v3_hybrid_70b.py` | .py | Uses RRF fusion |

### LLM Integration
| File | Type | LLM Models |
|------|------|-----------|
| `python/config.py` | .py | Config: llama-3.3-70b, gpt-4o-mini |
| `python/04_rag_pipeline.py` | .py | RAG generation |
| `python/eval_v1_bm25only.py` | .py | llama-3.1-8b-instant |
| `python/eval_v2_hybrid_8b.py` | .py | llama-3.1-8b-instant |
| `python/eval_v3_hybrid_70b.py` | .py | llama-3.3-70b-versatile |
| `evaluation/eval.py` | .py | Evaluation with LLM |
| `evaluation/eval_thomas_with_markus_head.py` | .py | Custom eval with LLM |

---

## 7. Evaluation Infrastructure

### Eval Scripts (All using Groq)
| Variant | Retrieval | LLM Model | Eval Script |
|---------|-----------|-----------|------------|
| V1 | BM25 only | llama-3.1-8b-instant | `eval_v1_bm25only.py` |
| V2 | Hybrid (BM25+FAISS+RRF) | llama-3.1-8b-instant | `eval_v2_hybrid_8b.py` |
| V3 | Hybrid (BM25+FAISS+RRF) | llama-3.3-70b-versatile | `eval_v3_hybrid_70b.py` |

### Evaluation Results
- **Format:** JSON
- **Files:**
  - `evaluation/custom_v1_bm25.json`
  - `evaluation/custom_v2_hybrid_8b.json`
  - `evaluation/custom_v3_hybrid_70b.json`
- **Metrics:** Retrieval precision@K, LLM accuracy/completeness/citation quality (1–5 scale)
- **Questions Evaluated:** 8 clinical questions across 4 difficulty zones

---

## 8. Summary Table

| Component | Technology | Status | File |
|-----------|------------|--------|------|
| **Embedding** | S-PubMedBERT-MS-MARCO (768-dim) | ✓ Active | `config.py:28` |
| **Dense Index** | FAISS IVFFlat (cosine sim) | ✓ Active | `pubmedbert.index` |
| **Sparse Index** | BM25 (rank_bm25 lib) | ✓ Active | `bm25.pkl` |
| **Fusion** | RRF (k=60) | ✓ Active | `lib/retriever.py` |
| **LLM (Baseline)** | llama-3.1-8b-instant (Groq) | ✓ Used | `eval_v1_*.py` |
| **LLM (Hybrid)** | llama-3.1-8b-instant (Groq) | ✓ Used | `eval_v2_*.py` |
| **LLM (Production)** | llama-3.3-70b-versatile (Groq) | ✓ Used | `eval_v3_*.py` |
| **LLM (Fallback)** | gpt-4o-mini (OpenAI) | ○ Fallback | `config.py:51` |

---

## 9. Known Issues & Notes

### Reference Bleed (Fixed)
- **Issue:** Initial FAISS index built with contaminated titles (Title + Reference list concatenated)
- **Fix Applied:** `chunk_text` patched directly in metadata (March 22, 2026)
- **Status:** Reference contamination reduced 99.7% → 1.1%
- **Outstanding:** FAISS vector embeddings still based on old chunks; scheduled re-embedding on Kaggle T4

### Chunking Configuration
- Text fields: `["title", "abstract", "introduction"]` (methods not yet included)
- Max chunk: 1,600 chars ≈ 400 BERT tokens
- Overlap: 200 chars ≈ 50 tokens

### API Keys Required
- `GROQ_API_KEY` — for Groq models (primary)
- `OPENAI_API_KEY` — for fallback (optional)
- `HF_TOKEN` — for S-PubMedBERT embeddings (optional but recommended)

---

## 10. Cross-Branch Deployment

All core technologies (`pubmedBERT`, `FAISS`, `BM25`, `RRF`, `LLMs`) are implemented in:
- **Branch:** `thomas` / `claude/happy-wescoff` (identical, 39 changed files)
- **Alternative branches:**
  - `marlo-80`: RAG_Baseline variants (notebooks only, no production pipeline)
  - `gettingdata` / `viktor`: Data extraction focused (EDA notebooks)

---

**Generated:** 2026-03-26
**Last Update:** Configuration verified in `python/config.py` and evaluation scripts
