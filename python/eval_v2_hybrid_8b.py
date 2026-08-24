"""
eval_v2_hybrid_8b.py
Capstone 1 — PMC Research Assistant

EVALUATION VARIANT 2 — Hybrid retrieval (BM25 + FAISS/PubMedBERT + RRF) + Llama 8B

Retrieval:  BM25 + FAISS (PubMedBERT embeddings) fused via RRF  ← full pipeline
LLM:        llama-3.1-8b-instant  (Groq, free tier — same as V1 for fair comparison)

Runs the same 8 questions as V1, saves full answers as JSON.
Use the same model (8B) as V1 to isolate the retrieval improvement.

Usage:
    cd Capstone1-text-prediction-for-scientific-papers
    python python/eval_v2_hybrid_8b.py
    python python/eval_v2_hybrid_8b.py --out evaluation/custom_v2_hybrid_8b.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    BM25_PATH, EMBED_MODEL, FAISS_INDEX, FAISS_META,
    GROQ_KEY, HF_TOKEN, MIN_SCORE, PROJECT_ROOT,
    RRF_K, TOP_K_BM25, TOP_K_FAISS, TOP_K_OUTPUT,
)
from lib.embedder import Embedder
from lib.retriever import load_index, retrieve

load_dotenv()

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL = "llama-3.1-8b-instant"   # same as V1 — retrieval is the only variable
SLEEP = 4.0

# ── 8 test questions (identical to V1 and V3) ─────────────────────────────────
QUESTIONS = [
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


# ── Doctor prompt ─────────────────────────────────────────────────────────────

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
                n = max(2, 4 - attempt)
                ex = max(80, 150 - attempt * 30)
                print(f"  413 → shrinking prompt to {n} papers × {ex}-char excerpts...")
                current_prompt = build_doctor_prompt(query, retrieved, n_papers=n, excerpt_len=ex)
                time.sleep(60)
            elif attempt < 3:
                wait = sleep * (2 ** attempt)
                print(f"  Retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                return f"ERROR: {e}"


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="evaluation/custom_v2_hybrid_8b.json")
    p.add_argument("--top-k", type=int, default=TOP_K_OUTPUT)
    return p.parse_args()


def main():
    args = parse_args()
    out_path = PROJECT_ROOT / args.out

    print("=" * 60)
    print("EVAL V2 — Hybrid (BM25+FAISS+RRF) + Llama 8B")
    print("=" * 60)

    # Load full index + embedder
    print("\nLoading FAISS index + BM25 + embedder...")
    index, meta, bm25 = load_index(FAISS_INDEX, FAISS_META, BM25_PATH)
    embedder = Embedder(EMBED_MODEL, hf_token=HF_TOKEN)
    print("Index loaded.")

    if not GROQ_KEY:
        print("ERROR: GROQ_API_KEY not set in .env")
        sys.exit(1)
    client = OpenAI(api_key=GROQ_KEY, base_url="https://api.groq.com/openai/v1")

    results = []

    for q in QUESTIONS:
        print(f"\n[{q['id']}] {q['category']}")
        print(f"  Q: {q['question'][:80]}...")

        query_vec = embedder.encode_single(q["question"])
        retrieved = retrieve(
            query_text=q["question"],
            query_vec=query_vec,
            index=index,
            meta=meta,
            bm25=bm25,
            top_k_faiss=TOP_K_FAISS,
            top_k_bm25=TOP_K_BM25,
            top_k_output=args.top_k,
            min_score=MIN_SCORE,
            rrf_k=RRF_K,
        )
        print(f"  Retrieved: {len(retrieved)} articles | top score: {retrieved[0]['retrieval_score']:.4f}" if retrieved else "  Retrieved: 0 articles")

        prompt = build_doctor_prompt(q["question"], retrieved)
        print(f"  Calling {MODEL}...")
        answer = call_llm(client, prompt, MODEL, SLEEP,
                          query=q["question"], retrieved=retrieved)
        print(f"  Answer length: {len(answer)} chars")

        results.append({
            "variant": "V2_Hybrid_8B",
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
