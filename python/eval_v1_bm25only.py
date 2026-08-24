"""
eval_v1_bm25only.py
Capstone 1 — PMC Research Assistant

EVALUATION VARIANT 1 — BM25-only retrieval + Llama 8B (Baseline)

Retrieval:  BM25 keyword search ONLY (no FAISS, no embeddings, no RRF)
LLM:        llama-3.1-8b-instant  (Groq, free tier)

Runs 8 custom open-ended medical questions and saves full answers as JSON.
This represents the simplest possible retrieval baseline.

Usage:
    cd Capstone1-text-prediction-for-scientific-papers
    python python/eval_v1_bm25only.py
    python python/eval_v1_bm25only.py --out evaluation/custom_v1_bm25.json
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from config import BM25_PATH, FAISS_META, GROQ_KEY, PROJECT_ROOT, TOP_K_BM25, TOP_K_OUTPUT

load_dotenv()

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL = "llama-3.1-8b-instant"   # 8B — consistent budget across all variants
SLEEP = 4.0                       # seconds between Groq calls (rate limit safety)

# ── 8 test questions — 4 categories ──────────────────────────────────────────
QUESTIONS = [
    # Category 1: Core domain — Oncology (strongly in index)
    {
        "id": "Q1",
        "category": "Core — Oncology",
        "in_distribution": True,
        "question": "Why do some patients with non-small cell lung cancer stop responding to checkpoint inhibitors?",
    },
    {
        "id": "Q2",
        "category": "Core — Oncology",
        "in_distribution": True,
        "question": "Can PD-L1 levels help predict response to immunotherapy in solid tumors?",
    },
    # Category 2: Moderate coverage — Neurology
    {
        "id": "Q3",
        "category": "Moderate — Neurology",
        "in_distribution": True,
        "question": "Are there reliable blood tests for early Alzheimer's disease?",
    },
    {
        "id": "Q4",
        "category": "Moderate — Neurology",
        "in_distribution": True,
        "question": "What is the clinical impact of LRRK2 mutations in Parkinson's disease?",
    },
    # Category 3: Borderline — rare/specific topics
    {
        "id": "Q5",
        "category": "Borderline — Rare/Autoimmune",
        "in_distribution": True,
        "question": "How is refractory autoimmune encephalitis typically treated?",
    },
    {
        "id": "Q6",
        "category": "Borderline — Gene Therapy",
        "in_distribution": True,
        "question": "How effective is gene therapy for spinal muscular atrophy in children?",
    },
    # Category 4: Out-of-distribution (topics likely underrepresented in oa_noncomm index)
    {
        "id": "Q7",
        "category": "Out-of-distribution — Ophthalmology",
        "in_distribution": False,
        "question": "What outcomes can be expected after corneal transplantation for keratoconus?",
    },
    {
        "id": "Q8",
        "category": "Out-of-distribution — Neonatology",
        "in_distribution": False,
        "question": "Does early nutrition affect long-term cognitive outcomes in preterm infants?",
    },
]


# ── BM25-only retrieval ───────────────────────────────────────────────────────

def bm25_retrieve(bm25, meta: list[dict], query: str, top_k_bm25: int, top_k_output: int) -> list[dict]:
    """BM25-only retrieval — no embeddings, no FAISS, no RRF."""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:top_k_bm25]
    hits = [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

    # Deduplicate to article level (keep highest BM25 score per pmcid)
    seen: dict[str, dict] = {}
    for idx, score in hits:
        entry = meta[idx]
        pmcid = entry["pmcid"]
        if pmcid not in seen or score > seen[pmcid]["retrieval_score"]:
            seen[pmcid] = {**entry, "retrieval_score": score}

    ranked = sorted(seen.values(), key=lambda x: x["retrieval_score"], reverse=True)
    return ranked[:top_k_output]


# ── Doctor prompt (shared with main pipeline) ─────────────────────────────────

def build_doctor_prompt(query: str, retrieved: list[dict], n_papers: int = 4, excerpt_len: int = 150) -> str:
    # 4 papers × 150-char excerpts → ~3k tokens, safely under Groq 8B 6k TPM limit
    papers = "\n\n".join(
        f"[{i}] Title: {(d.get('title') or '?')[:80]}\n"
        f"    PMCID: {d.get('pmcid', '?')}\n"
        f"    Keywords: {', '.join(d.get('kwd_lst_cl', [])[:4])}\n"
        f"    Excerpt: {(d.get('chunk_text') or '')[:excerpt_len]}"
        for i, d in enumerate(retrieved[:n_papers], 1)
    )
    return f"""You are a biomedical research assistant helping a clinician.

QUESTION:
{query}

RETRIEVED PAPERS:
{papers}

INSTRUCTIONS:
1. Write a clear, evidence-based answer (2-3 paragraphs) in English.
2. Cite papers by their number [1], [2], etc.
3. After the answer, add a short "REFERENCES:" section.
4. If evidence is insufficient, say so honestly.
5. Do NOT invent information not in the provided papers.
"""


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_llm(client, prompt: str, model: str, sleep: float,
             query: str = "", retrieved: list = None) -> str:
    current_prompt = prompt
    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": current_prompt}],
                temperature=0.3,
                max_tokens=800,
            )
            time.sleep(sleep)
            return response.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e)
            print(f"  LLM call failed (attempt {attempt}/3): {e}")
            if "413" in err_str and retrieved and query and attempt < 3:
                # Shrink prompt: reduce papers and excerpt length
                n = max(2, 4 - attempt)
                ex = max(80, 150 - attempt * 30)
                print(f"  413 → shrinking prompt to {n} papers × {ex}-char excerpts...")
                current_prompt = build_doctor_prompt(query, retrieved, n_papers=n, excerpt_len=ex)
                time.sleep(60)   # long pause after 413 — Groq drops connection otherwise
            elif attempt < 3:
                wait = sleep * (2 ** attempt)
                print(f"  Retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                return f"ERROR: {e}"


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="evaluation/custom_v1_bm25.json")
    p.add_argument("--top-k", type=int, default=TOP_K_OUTPUT, help="Articles to retrieve")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = PROJECT_ROOT / args.out

    print("=" * 60)
    print("EVAL V1 — BM25-only + Llama 8B")
    print("=" * 60)

    # Load BM25 + meta only (no FAISS, no embedder)
    print("\nLoading BM25 index...")
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    with open(FAISS_META, "rb") as f:
        meta = pickle.load(f)
    print(f"BM25 loaded | {len(meta)} chunks in metadata")

    # Groq client
    if not GROQ_KEY:
        print("ERROR: GROQ_API_KEY not set in .env")
        sys.exit(1)
    client = OpenAI(api_key=GROQ_KEY, base_url="https://api.groq.com/openai/v1")

    results = []

    for q in QUESTIONS:
        print(f"\n[{q['id']}] {q['category']}")
        print(f"  Q: {q['question'][:80]}...")

        retrieved = bm25_retrieve(bm25, meta, q["question"], TOP_K_BM25, args.top_k)
        print(f"  Retrieved: {len(retrieved)} articles | top score: {retrieved[0]['retrieval_score']:.4f}" if retrieved else "  Retrieved: 0 articles (no BM25 hits)")

        prompt = build_doctor_prompt(q["question"], retrieved)
        print(f"  Calling {MODEL}...")
        answer = call_llm(client, prompt, MODEL, SLEEP,
                          query=q["question"], retrieved=retrieved)
        print(f"  Answer length: {len(answer)} chars")

        results.append({
            "variant": "V1_BM25only",
            "model": MODEL,
            "id": q["id"],
            "category": q["category"],
            "in_distribution": q["in_distribution"],
            "question": q["question"],
            "n_retrieved": len(retrieved),
            "top_retrieval_score": retrieved[0]["retrieval_score"] if retrieved else 0.0,
            "retrieved_titles": [d.get("title", "?")[:80] for d in retrieved[:5]],
            "retrieved_clusters": list({d.get("cluster_name", "?") for d in retrieved[:10]}),
            "answer": answer,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Saved {len(results)} results → {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
