"""
05_evaluation.py
Capstone 1 — PMC Research Assistant  v2

STAGE 5 — Comprehensive evaluation & benchmarking.

Three evaluation suites:
  A. Keyword Prediction   — leave-one-out on known articles → F1, Precision, Recall
  B. Retrieval Quality    — self-retrieval + ablation (Hybrid vs FAISS-only vs BM25-only)
  C. Doctor Mode          — gold-standard Q&A → citation coverage, retrieval relevance

All results are saved as JSON + presentation-ready plots.

Usage:
    python python/05_evaluation.py                   # run all evals
    python python/05_evaluation.py --suite keyword   # keyword eval only
    python python/05_evaluation.py --suite retrieval  # retrieval eval only
    python python/05_evaluation.py --suite doctor     # doctor mode eval only
    python python/05_evaluation.py --sample 200       # smaller sample (faster)
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from dotenv import load_dotenv

from config import (
    BM25_PATH, EMBED_MODEL, FAISS_INDEX, FAISS_META,
    GROQ_KEY, HF_TOKEN, LLM_MODEL, MIN_SCORE, OPENAI_KEY,
    PROJECT_ROOT, RRF_K, TFIDF_MODEL, TOP_K_BM25, TOP_K_FAISS,
    TOP_K_OUTPUT, TOP_K_TFIDF_DIRECT,
)

load_dotenv()

EVAL_DIR  = PROJECT_ROOT / "evaluation"
PLOTS_DIR = PROJECT_ROOT / "plots" / "evaluation"

# ══════════════════════════════════════════════════════════════════════════════
#  GOLD-STANDARD TEST QUERIES (Doctor Mode)
# ══════════════════════════════════════════════════════════════════════════════

DOCTOR_GOLD = [
    {
        "query": "What are the latest immunotherapy approaches for non-small cell lung cancer?",
        "expected_clusters": ["prognosis / breast cancer / lung cancer"],
        "expected_keywords": ["immunotherapy", "lung cancer", "non-small cell", "checkpoint", "pembrolizumab"],
        "topic": "Oncology",
    },
    {
        "query": "How effective are machine learning models for predicting breast cancer prognosis?",
        "expected_clusters": ["prognosis / breast cancer / lung cancer", "machine learning / apoptosis / stem cells"],
        "expected_keywords": ["machine learning", "breast cancer", "prognosis", "prediction", "survival"],
        "topic": "ML/Oncology",
    },
    {
        "query": "What is the role of LRRK2 mutations in Parkinson's disease pathogenesis?",
        "expected_clusters": ["dementia / magnetic resonance imaging / alzheimer s", "machine learning / apoptosis / stem cells"],
        "expected_keywords": ["parkinson", "lrrk2", "mutation", "dopamine", "neurodegenerative"],
        "topic": "Neurology",
    },
    {
        "query": "What are current COVID-19 treatment protocols for hospitalized patients?",
        "expected_clusters": ["covid 19 / health care / public health", "sars cov / sars cov 2 / intensive care"],
        "expected_keywords": ["covid", "sars-cov-2", "treatment", "hospitalized", "remdesivir", "dexamethasone"],
        "topic": "Infectious Disease",
    },
    {
        "query": "How does cardiac rehabilitation improve outcomes after myocardial infarction?",
        "expected_clusters": ["heart failure / myocardial infarction / coronary artery"],
        "expected_keywords": ["cardiac", "rehabilitation", "myocardial infarction", "heart failure", "exercise"],
        "topic": "Cardiology",
    },
    {
        "query": "What are the risk factors and epidemiology of diabetes in developing countries?",
        "expected_clusters": ["epidemiology / risk factors / mortality"],
        "expected_keywords": ["diabetes", "risk factors", "epidemiology", "prevalence", "developing countries"],
        "topic": "Epidemiology",
    },
    {
        "query": "What is the effectiveness of stem cell therapy for spinal cord injury?",
        "expected_clusters": ["spinal cord / cord injury / bone marrow", "machine learning / apoptosis / stem cells"],
        "expected_keywords": ["stem cell", "spinal cord", "injury", "therapy", "regeneration"],
        "topic": "Regenerative Medicine",
    },
    {
        "query": "How is MRI used for early detection of Alzheimer's disease?",
        "expected_clusters": ["dementia / magnetic resonance imaging / alzheimer s"],
        "expected_keywords": ["alzheimer", "mri", "magnetic resonance", "dementia", "early detection", "biomarker"],
        "topic": "Neuroimaging",
    },
    {
        "query": "What are the outcomes of total knee arthroplasty in elderly patients?",
        "expected_clusters": ["knee arthroplasty / reliability / laparoscopy"],
        "expected_keywords": ["knee arthroplasty", "elderly", "outcome", "total knee replacement"],
        "topic": "Orthopedics",
    },
    {
        "query": "How effective is methotrexate for rheumatoid arthritis management?",
        "expected_clusters": ["safety / rheumatoid arthritis / mitochondria"],
        "expected_keywords": ["methotrexate", "rheumatoid arthritis", "treatment", "disease activity"],
        "topic": "Rheumatology",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _norm(kw: str) -> str:
    kw = kw.lower().strip()
    kw = re.sub(r"['\u2019\u2018`]s?\b", "", kw)
    kw = re.sub(r"\s+", " ", kw)
    return kw.strip()


def keyword_f1(predicted: list[str], ground_truth: list[str]) -> dict:
    """Compute F1, Precision, Recall for keyword sets (case-insensitive)."""
    pred_set = {_norm(k) for k in predicted}
    true_set = {_norm(k) for k in ground_truth}
    if not pred_set and not true_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_set or not true_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(pred_set & true_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall    = tp / len(true_set) if true_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp,
            "pred_count": len(pred_set), "true_count": len(true_set)}


def partial_keyword_match(predicted: list[str], ground_truth: list[str]) -> dict:
    """Softer matching: a prediction counts if it's a substring of any ground-truth (or vice versa)."""
    pred_norms = [_norm(k) for k in predicted]
    true_norms = [_norm(k) for k in ground_truth]
    tp = 0
    for p in pred_norms:
        for t in true_norms:
            if p in t or t in p:
                tp += 1
                break
    precision = tp / len(pred_norms) if pred_norms else 0.0
    recall    = tp / len(true_norms) if true_norms else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp}


def mrr_at_k(relevant_pmcids: set, retrieved: list[dict], k: int = 15) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant hit."""
    for i, d in enumerate(retrieved[:k]):
        if d.get("pmcid") in relevant_pmcids:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(relevant_pmcids: set, retrieved: list[dict], k: int = 5) -> float:
    """Precision@k: fraction of top-k that are relevant."""
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d.get("pmcid") in relevant_pmcids) / k


def ndcg_at_k(relevant_pmcids: set, retrieved: list[dict], k: int = 15) -> float:
    """Normalised Discounted Cumulative Gain."""
    dcg = 0.0
    for i, d in enumerate(retrieved[:k]):
        rel = 1.0 if d.get("pmcid") in relevant_pmcids else 0.0
        dcg += rel / np.log2(i + 2)
    # Ideal: all relevant first
    n_rel = min(len(relevant_pmcids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


def cluster_precision(expected_clusters: list[str], retrieved: list[dict], k: int = 10) -> float:
    """Fraction of top-k retrieved papers whose cluster matches expected."""
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for d in top if d.get("cluster_name", "") in expected_clusters)
    return hits / len(top)


def keyword_overlap_score(expected_keywords: list[str], retrieved: list[dict], k: int = 10) -> float:
    """Fraction of expected keywords found in the retrieved papers' keyword lists."""
    if not expected_keywords:
        return 0.0
    retrieved_kws = set()
    for d in retrieved[:k]:
        for kw in d.get("kwd_lst_cl", []):
            retrieved_kws.add(_norm(kw))
    expected_norm = {_norm(k) for k in expected_keywords}
    # Partial matching
    matched = 0
    for ek in expected_norm:
        for rk in retrieved_kws:
            if ek in rk or rk in ek:
                matched += 1
                break
    return matched / len(expected_norm)


# ══════════════════════════════════════════════════════════════════════════════
#  SUITE A: KEYWORD PREDICTION EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def eval_keyword_prediction(meta, index, bm25, embedder, sample_n=100, seed=42):
    """
    Leave-one-out evaluation:
      For each sampled article, use title+abstract as query, predict keywords,
      compare against ground-truth kwd_lst_cl.
    """
    from lib.retriever import retrieve, _faiss_search, _bm25_search, _rrf_merge, _dedup_articles

    # Build article lookup (deduplicated)
    articles = {}
    for m in meta:
        pid = m["pmcid"]
        if pid not in articles:
            articles[pid] = {
                "pmcid": pid,
                "title": m.get("title", ""),
                "kwd_lst_cl": m.get("kwd_lst_cl", []),
                "cluster_name": m.get("cluster_name", ""),
                "tfidf_candidates": m.get("tfidf_candidates", []),
            }

    # Filter: only articles with ≥3 keywords and a title
    eligible = [a for a in articles.values()
                if len(a["kwd_lst_cl"]) >= 3 and len(a["title"]) > 20]

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(eligible), size=min(sample_n, len(eligible)), replace=False)
    sample = [eligible[i] for i in sample_idx]

    print(f"\n{'='*70}")
    print(f"  KEYWORD PREDICTION EVALUATION  ({len(sample)} articles)")
    print(f"{'='*70}")

    results = []
    for i, art in enumerate(sample):
        query_text = f"{art['title']}"
        query_vec  = embedder.encode_single(query_text)

        retrieved = retrieve(
            query_text=query_text, query_vec=query_vec,
            index=index, meta=meta, bm25=bm25,
            top_k_faiss=TOP_K_FAISS, top_k_bm25=TOP_K_BM25,
            top_k_output=TOP_K_OUTPUT, min_score=MIN_SCORE, rrf_k=RRF_K,
        )

        # Exclude self from retrieved
        retrieved_no_self = [d for d in retrieved if d["pmcid"] != art["pmcid"]]

        # Aggregate keywords from retrieved papers (same logic as 04_rag_pipeline)
        counter = Counter()
        for doc in retrieved_no_self:
            score = doc.get("retrieval_score", 0.0)
            for kw in doc.get("kwd_lst_cl", []):
                counter[_norm(kw)] += 2.0 * score
            for kw in doc.get("tfidf_candidates", []):
                counter[_norm(kw)] += 1.0 * score
        predicted = [kw for kw, _ in counter.most_common(TOP_K_OUTPUT)]

        # Evaluate
        exact  = keyword_f1(predicted, art["kwd_lst_cl"])
        soft   = partial_keyword_match(predicted, art["kwd_lst_cl"])

        results.append({
            "pmcid": art["pmcid"],
            "title": art["title"][:80],
            "cluster": art["cluster_name"],
            "n_true_kw": len(art["kwd_lst_cl"]),
            "n_pred_kw": len(predicted),
            "exact_f1": exact["f1"],
            "exact_precision": exact["precision"],
            "exact_recall": exact["recall"],
            "soft_f1": soft["f1"],
            "soft_precision": soft["precision"],
            "soft_recall": soft["recall"],
            "self_retrieved": any(d["pmcid"] == art["pmcid"] for d in retrieved),
        })

        if (i + 1) % 25 == 0:
            avg_f1 = np.mean([r["exact_f1"] for r in results])
            print(f"  [{i+1}/{len(sample)}]  running avg exact F1 = {avg_f1:.3f}")

    # Summary
    summary = {
        "n_articles": len(results),
        "exact_f1_mean":    float(np.mean([r["exact_f1"] for r in results])),
        "exact_f1_std":     float(np.std([r["exact_f1"] for r in results])),
        "exact_prec_mean":  float(np.mean([r["exact_precision"] for r in results])),
        "exact_rec_mean":   float(np.mean([r["exact_recall"] for r in results])),
        "soft_f1_mean":     float(np.mean([r["soft_f1"] for r in results])),
        "soft_f1_std":      float(np.std([r["soft_f1"] for r in results])),
        "soft_prec_mean":   float(np.mean([r["soft_precision"] for r in results])),
        "soft_rec_mean":    float(np.mean([r["soft_recall"] for r in results])),
        "self_retrieval_rate": float(np.mean([r["self_retrieved"] for r in results])),
    }

    # Per-cluster breakdown
    cluster_results = {}
    for r in results:
        c = r["cluster"]
        if c not in cluster_results:
            cluster_results[c] = []
        cluster_results[c].append(r)

    cluster_summary = {}
    for c, rs in cluster_results.items():
        cluster_summary[c] = {
            "n": len(rs),
            "exact_f1_mean": float(np.mean([r["exact_f1"] for r in rs])),
            "soft_f1_mean":  float(np.mean([r["soft_f1"] for r in rs])),
        }

    print(f"\n  RESULTS:")
    print(f"    Exact F1:     {summary['exact_f1_mean']:.3f} ± {summary['exact_f1_std']:.3f}")
    print(f"    Exact Prec:   {summary['exact_prec_mean']:.3f}")
    print(f"    Exact Recall: {summary['exact_rec_mean']:.3f}")
    print(f"    Soft F1:      {summary['soft_f1_mean']:.3f} ± {summary['soft_f1_std']:.3f}")
    print(f"    Self-retrieval rate: {summary['self_retrieval_rate']:.1%}")
    print(f"\n  Per-cluster F1:")
    for c, cs in sorted(cluster_summary.items(), key=lambda x: x[1]["n"], reverse=True):
        print(f"    {c[:50]:50s}  n={cs['n']:3d}  exact={cs['exact_f1_mean']:.3f}  soft={cs['soft_f1_mean']:.3f}")

    return {"summary": summary, "cluster_summary": cluster_summary, "details": results}


# ══════════════════════════════════════════════════════════════════════════════
#  SUITE B: RETRIEVAL QUALITY (Ablation: Hybrid vs FAISS-only vs BM25-only)
# ══════════════════════════════════════════════════════════════════════════════

def eval_retrieval_ablation(meta, index, bm25, embedder, sample_n=100, seed=42):
    """
    Self-retrieval test: query with known article, measure if same-cluster
    articles rank high. Compare Hybrid vs FAISS-only vs BM25-only.
    """
    from lib.retriever import (_faiss_search, _bm25_search, _rrf_merge,
                               _dedup_articles)

    # Build cluster lookup
    article_cluster = {}
    for m in meta:
        article_cluster[m["pmcid"]] = m.get("cluster_name", "")

    # Get unique articles
    articles = {}
    for m in meta:
        pid = m["pmcid"]
        if pid not in articles:
            articles[pid] = {
                "pmcid": pid,
                "title": m.get("title", ""),
                "cluster_name": m.get("cluster_name", ""),
                "kwd_lst_cl": m.get("kwd_lst_cl", []),
            }

    eligible = [a for a in articles.values() if len(a["title"]) > 20]
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(eligible), size=min(sample_n, len(eligible)), replace=False)
    sample = [eligible[i] for i in sample_idx]

    print(f"\n{'='*70}")
    print(f"  RETRIEVAL ABLATION  ({len(sample)} articles)")
    print(f"  Comparing: Hybrid (FAISS+BM25) vs FAISS-only vs BM25-only")
    print(f"{'='*70}")

    methods = ["hybrid", "faiss_only", "bm25_only"]
    all_results = {m: [] for m in methods}

    for i, art in enumerate(sample):
        query_text = art["title"]
        query_vec  = embedder.encode_single(query_text)

        # Same-cluster articles = "relevant" (excluding self)
        same_cluster_pmcids = {
            pid for pid, cl in article_cluster.items()
            if cl == art["cluster_name"] and pid != art["pmcid"]
        }

        # --- Hybrid ---
        faiss_hits = _faiss_search(index, query_vec, TOP_K_FAISS, MIN_SCORE)
        bm25_hits  = _bm25_search(bm25, query_text, TOP_K_BM25)
        merged     = _rrf_merge(faiss_hits, bm25_hits, k=RRF_K)
        hybrid_ret = _dedup_articles(merged, meta, TOP_K_OUTPUT)
        hybrid_ret = [d for d in hybrid_ret if d["pmcid"] != art["pmcid"]]

        # --- FAISS-only ---
        faiss_only = _rrf_merge(faiss_hits, [], k=RRF_K)
        faiss_ret  = _dedup_articles(faiss_only, meta, TOP_K_OUTPUT)
        faiss_ret  = [d for d in faiss_ret if d["pmcid"] != art["pmcid"]]

        # --- BM25-only ---
        bm25_only  = _rrf_merge([], bm25_hits, k=RRF_K)
        bm25_ret   = _dedup_articles(bm25_only, meta, TOP_K_OUTPUT)
        bm25_ret   = [d for d in bm25_ret if d["pmcid"] != art["pmcid"]]

        for method, retrieved in [("hybrid", hybrid_ret), ("faiss_only", faiss_ret), ("bm25_only", bm25_ret)]:
            all_results[method].append({
                "pmcid": art["pmcid"],
                "cluster": art["cluster_name"],
                "mrr@15":        mrr_at_k(same_cluster_pmcids, retrieved, 15),
                "precision@5":   precision_at_k(same_cluster_pmcids, retrieved, 5),
                "precision@10":  precision_at_k(same_cluster_pmcids, retrieved, 10),
                "ndcg@15":       ndcg_at_k(same_cluster_pmcids, retrieved, 15),
                "cluster_prec@10": cluster_precision([art["cluster_name"]], retrieved, 10),
            })

        if (i + 1) % 25 == 0:
            h_mrr = np.mean([r["mrr@15"] for r in all_results["hybrid"]])
            print(f"  [{i+1}/{len(sample)}]  hybrid MRR@15 = {h_mrr:.3f}")

    # Summary per method
    summaries = {}
    for method in methods:
        rs = all_results[method]
        summaries[method] = {
            "mrr@15":          float(np.mean([r["mrr@15"] for r in rs])),
            "precision@5":     float(np.mean([r["precision@5"] for r in rs])),
            "precision@10":    float(np.mean([r["precision@10"] for r in rs])),
            "ndcg@15":         float(np.mean([r["ndcg@15"] for r in rs])),
            "cluster_prec@10": float(np.mean([r["cluster_prec@10"] for r in rs])),
        }

    print(f"\n  {'Method':<15s} {'MRR@15':>8s} {'P@5':>8s} {'P@10':>8s} {'NDCG@15':>8s} {'ClustP@10':>10s}")
    print(f"  {'-'*60}")
    for method in methods:
        s = summaries[method]
        label = {"hybrid": "Hybrid", "faiss_only": "FAISS-only", "bm25_only": "BM25-only"}[method]
        print(f"  {label:<15s} {s['mrr@15']:8.3f} {s['precision@5']:8.3f} "
              f"{s['precision@10']:8.3f} {s['ndcg@15']:8.3f} {s['cluster_prec@10']:10.3f}")

    return {"summaries": summaries, "details": all_results}


# ══════════════════════════════════════════════════════════════════════════════
#  SUITE C: DOCTOR MODE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def _build_eval_doctor_prompt(query: str, retrieved: list[dict]) -> str:
    """Shorter doctor prompt for evaluation — fits within Groq free-tier limits.
    Uses max 5 papers with truncated excerpts (200 chars instead of 400)."""
    papers = "\n".join(
        f"[{i}] {d.get('title', '?')[:80]}  PMCID:{d.get('pmcid','?')}\n"
        f"    Keywords: {', '.join(d.get('kwd_lst_cl', [])[:5])}\n"
        f"    Excerpt: {(d.get('chunk_text') or '')[:200]}"
        for i, d in enumerate(retrieved[:5], 1)
    )
    return f"""Answer the research question using ONLY these papers. Cite by [1],[2] etc.
Be concise (2-3 paragraphs). If evidence is insufficient, say so.

QUESTION: {query}

PAPERS:
{papers}

Answer:"""


def eval_doctor_mode(meta, index, bm25, embedder):
    """
    Evaluate doctor mode using gold-standard queries:
      - Cluster precision (do retrieved papers match expected topic?)
      - Keyword overlap (do retrieved papers cover expected medical terms?)
      - LLM citation coverage (does the answer cite papers? only if LLM available)
    """
    from lib.retriever import retrieve

    print(f"\n{'='*70}")
    print(f"  DOCTOR MODE EVALUATION  ({len(DOCTOR_GOLD)} gold-standard queries)")
    print(f"{'='*70}")

    results = []
    for q in DOCTOR_GOLD:
        query = q["query"]
        query_vec = embedder.encode_single(query)

        retrieved = retrieve(
            query_text=query, query_vec=query_vec,
            index=index, meta=meta, bm25=bm25,
            top_k_faiss=TOP_K_FAISS, top_k_bm25=TOP_K_BM25,
            top_k_output=TOP_K_OUTPUT, min_score=MIN_SCORE, rrf_k=RRF_K,
        )

        cl_prec = cluster_precision(q["expected_clusters"], retrieved, 10)
        kw_overlap = keyword_overlap_score(q["expected_keywords"], retrieved, 10)

        # Try LLM evaluation if available
        llm_result = None
        try:
            from openai import OpenAI
            if GROQ_KEY:
                client = OpenAI(api_key=GROQ_KEY, base_url="https://api.groq.com/openai/v1")
            elif OPENAI_KEY:
                client = OpenAI(api_key=OPENAI_KEY)
            else:
                client = None

            if client:
                # Build a SHORT eval prompt (max 5 papers, truncated excerpts)
                # to stay within Groq free-tier 12k TPM limit
                prompt = _build_eval_doctor_prompt(query, retrieved[:5])

                provider = "Groq" if GROQ_KEY else "OpenAI"
                # Rate-limit: wait between Groq calls (free tier = 12k TPM)
                if GROQ_KEY:
                    time.sleep(8)

                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=800,
                )
                answer = response.choices[0].message.content.strip()

                # Count citations [1], [2], etc.
                citations = set(re.findall(r"\[(\d+)\]", answer))
                n_citations = len(citations)
                # Check if "insufficient" disclaimer is present
                has_disclaimer = any(w in answer.lower() for w in
                    ["insufficient", "limited evidence", "not enough", "cannot fully"])

                llm_result = {
                    "answer_length": len(answer),
                    "n_citations": n_citations,
                    "citation_coverage": n_citations / min(5, len(retrieved)) if retrieved else 0,
                    "has_disclaimer": has_disclaimer,
                    "answer_preview": answer[:200],
                }
                print(f"  [{q['topic']:20s}]  cluster_prec={cl_prec:.2f}  kw_overlap={kw_overlap:.2f}"
                      f"  citations={n_citations}  ({provider}) {'(disclaimer)' if has_disclaimer else ''}")
            else:
                print(f"  [{q['topic']:20s}]  cluster_prec={cl_prec:.2f}  kw_overlap={kw_overlap:.2f}  (no LLM)")
        except Exception as e:
            print(f"  [{q['topic']:20s}]  cluster_prec={cl_prec:.2f}  kw_overlap={kw_overlap:.2f}  (LLM error: {e})")

        results.append({
            "query": query,
            "topic": q["topic"],
            "cluster_precision@10": cl_prec,
            "keyword_overlap": kw_overlap,
            "n_retrieved": len(retrieved),
            "top_clusters": list(Counter(d.get("cluster_name","") for d in retrieved[:10]).most_common(3)),
            "llm": llm_result,
        })

    # Summary
    summary = {
        "n_queries": len(results),
        "avg_cluster_precision": float(np.mean([r["cluster_precision@10"] for r in results])),
        "avg_keyword_overlap":   float(np.mean([r["keyword_overlap"] for r in results])),
    }

    llm_results = [r for r in results if r["llm"] is not None]
    if llm_results:
        summary["avg_citations"]      = float(np.mean([r["llm"]["n_citations"] for r in llm_results]))
        summary["avg_citation_coverage"] = float(np.mean([r["llm"]["citation_coverage"] for r in llm_results]))
        summary["disclaimer_rate"]    = float(np.mean([r["llm"]["has_disclaimer"] for r in llm_results]))

    print(f"\n  SUMMARY:")
    print(f"    Avg cluster precision@10: {summary['avg_cluster_precision']:.3f}")
    print(f"    Avg keyword overlap:      {summary['avg_keyword_overlap']:.3f}")
    if "avg_citations" in summary:
        print(f"    Avg citations per answer: {summary['avg_citations']:.1f}")
        print(f"    Citation coverage:        {summary['avg_citation_coverage']:.1%}")
        print(f"    Disclaimer rate:          {summary['disclaimer_rate']:.1%}")

    return {"summary": summary, "details": results}


# ══════════════════════════════════════════════════════════════════════════════
#  SUITE D: LLM-AS-JUDGE (RAGAS-Style Answer Quality)
# ══════════════════════════════════════════════════════════════════════════════

JUDGE_FAITHFULNESS_PROMPT = """You are an evaluation judge. Score the following answer for FAITHFULNESS.

QUESTION: {query}

RETRIEVED EVIDENCE:
{evidence}

GENERATED ANSWER:
{answer}

FAITHFULNESS means: every claim in the answer is supported by the retrieved evidence.
Score from 1 (many unsupported claims) to 5 (all claims grounded in evidence).

Reply with ONLY a JSON object: {{"score": <1-5>, "reason": "<one sentence>"}}"""

JUDGE_RELEVANCE_PROMPT = """You are an evaluation judge. Score the following answer for RELEVANCE.

QUESTION: {query}

GENERATED ANSWER:
{answer}

RELEVANCE means: the answer directly addresses the question and is useful to a clinician.
Score from 1 (off-topic or useless) to 5 (directly and completely answers the question).

Reply with ONLY a JSON object: {{"score": <1-5>, "reason": "<one sentence>"}}"""

JUDGE_COMPLETENESS_PROMPT = """You are an evaluation judge. Score the following answer for COMPLETENESS.

QUESTION: {query}

GENERATED ANSWER:
{answer}

COMPLETENESS means: the answer covers the key aspects of the question (not just one narrow angle).
A low score if important aspects are missing. A high score if comprehensive given the evidence.
Score from 1 (very incomplete) to 5 (thorough coverage).

Reply with ONLY a JSON object: {{"score": <1-5>, "reason": "<one sentence>"}}"""


def _judge_call(client, model, prompt_template, **kwargs):
    """Send a single judge prompt and parse the JSON score."""
    prompt = prompt_template.format(**kwargs)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        text = response.choices[0].message.content.strip()
        # Parse JSON from response
        match = re.search(r'\{[^}]+\}', text)
        if match:
            return json.loads(match.group())
        return {"score": 0, "reason": "parse error"}
    except Exception as e:
        return {"score": 0, "reason": str(e)}


def eval_answer_quality(meta, index, bm25, embedder):
    """
    RAGAS-style LLM-as-Judge evaluation:
      For each gold-standard query → retrieve → generate answer → judge on:
        - Faithfulness (grounded in evidence?)
        - Relevance (answers the question?)
        - Completeness (covers key aspects?)

    Requires GROQ_API_KEY or OPENAI_API_KEY.
    """
    from lib.retriever import retrieve

    try:
        from openai import OpenAI
    except ImportError:
        print("  openai package not installed — skipping answer quality eval.")
        return None

    if GROQ_KEY:
        client = OpenAI(api_key=GROQ_KEY, base_url="https://api.groq.com/openai/v1")
        provider = "Groq"
    elif OPENAI_KEY:
        client = OpenAI(api_key=OPENAI_KEY)
        provider = "OpenAI"
    else:
        print("  No LLM API key — skipping answer quality eval.")
        return None

    print(f"\n{'='*70}")
    print(f"  ANSWER QUALITY EVALUATION (LLM-as-Judge via {provider})")
    print(f"  {len(DOCTOR_GOLD)} queries × 3 dimensions (Faithfulness, Relevance, Completeness)")
    print(f"{'='*70}")

    results = []
    for q in DOCTOR_GOLD:
        query = q["query"]
        query_vec = embedder.encode_single(query)

        retrieved = retrieve(
            query_text=query, query_vec=query_vec,
            index=index, meta=meta, bm25=bm25,
            top_k_faiss=TOP_K_FAISS, top_k_bm25=TOP_K_BM25,
            top_k_output=TOP_K_OUTPUT, min_score=MIN_SCORE, rrf_k=RRF_K,
        )

        # Generate answer (short prompt)
        prompt = _build_eval_doctor_prompt(query, retrieved[:5])
        if GROQ_KEY:
            time.sleep(8)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
        )
        answer = resp.choices[0].message.content.strip()

        # Build evidence summary for judge
        evidence = "\n".join(
            f"[{i}] {d.get('title','?')[:80]}\n    {(d.get('chunk_text') or '')[:150]}"
            for i, d in enumerate(retrieved[:5], 1)
        )

        # Judge calls (with rate limiting)
        if GROQ_KEY:
            time.sleep(8)
        faith = _judge_call(client, LLM_MODEL, JUDGE_FAITHFULNESS_PROMPT,
                           query=query, evidence=evidence, answer=answer)
        if GROQ_KEY:
            time.sleep(8)
        relev = _judge_call(client, LLM_MODEL, JUDGE_RELEVANCE_PROMPT,
                           query=query, answer=answer)
        if GROQ_KEY:
            time.sleep(8)
        compl = _judge_call(client, LLM_MODEL, JUDGE_COMPLETENESS_PROMPT,
                           query=query, answer=answer)

        results.append({
            "query": query,
            "topic": q["topic"],
            "faithfulness": faith,
            "relevance": relev,
            "completeness": compl,
            "answer_length": len(answer),
            "answer_preview": answer[:200],
        })

        print(f"  [{q['topic']:20s}]  faith={faith.get('score',0)}  relev={relev.get('score',0)}  "
              f"compl={compl.get('score',0)}")

    # Summary
    valid = [r for r in results if all(
        r[dim].get("score", 0) > 0 for dim in ["faithfulness", "relevance", "completeness"]
    )]
    summary = {"n_queries": len(results), "n_valid": len(valid)}
    if valid:
        for dim in ["faithfulness", "relevance", "completeness"]:
            scores = [r[dim]["score"] for r in valid]
            summary[f"avg_{dim}"] = float(np.mean(scores))
            summary[f"std_{dim}"] = float(np.std(scores))
        summary["avg_overall"] = float(np.mean([
            summary["avg_faithfulness"], summary["avg_relevance"], summary["avg_completeness"]
        ]))

    print(f"\n  SUMMARY (scale 1-5):")
    if valid:
        print(f"    Faithfulness:  {summary['avg_faithfulness']:.2f} ± {summary['std_faithfulness']:.2f}")
        print(f"    Relevance:     {summary['avg_relevance']:.2f} ± {summary['std_relevance']:.2f}")
        print(f"    Completeness:  {summary['avg_completeness']:.2f} ± {summary['std_completeness']:.2f}")
        print(f"    Overall:       {summary['avg_overall']:.2f} / 5.00")
    else:
        print(f"    No valid judge results.")

    return {"summary": summary, "details": results}


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def _save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  Plot saved: {path}")


def plot_keyword_eval(kw_results, out_dir):
    """Bar charts for keyword prediction F1 — overall + per-cluster."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summ = kw_results["summary"]
    cluster_summ = kw_results["cluster_summary"]

    # --- 1. Overall metrics bar ---
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics = ["Exact\nPrecision", "Exact\nRecall", "Exact\nF1", "Soft\nPrecision", "Soft\nRecall", "Soft\nF1"]
    values  = [summ["exact_prec_mean"], summ["exact_rec_mean"], summ["exact_f1_mean"],
               summ["soft_prec_mean"], summ["soft_rec_mean"], summ["soft_f1_mean"]]
    colors  = ["#4f86c6", "#4f86c6", "#2d5aa0", "#6dbf67", "#6dbf67", "#3d8a37"]
    bars = ax.bar(metrics, values, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.2)
    ax.set_ylabel("Score")
    ax.set_title(f"Keyword Prediction — Exact vs Soft Matching  (n={summ['n_articles']})", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save_fig(fig, out_dir / "keyword_prediction_overall.png")

    # --- 2. Per-cluster F1 comparison ---
    sorted_cl = sorted(cluster_summ.items(), key=lambda x: x[1]["n"], reverse=True)
    names   = [c[:35] for c, _ in sorted_cl]
    exact_f1 = [cs["exact_f1_mean"] for _, cs in sorted_cl]
    soft_f1  = [cs["soft_f1_mean"] for _, cs in sorted_cl]
    counts   = [cs["n"] for _, cs in sorted_cl]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5)))
    y = np.arange(len(names))
    h = 0.35
    ax.barh(y + h/2, exact_f1, h, label="Exact F1", color="#4f86c6", edgecolor="white")
    ax.barh(y - h/2, soft_f1, h, label="Soft F1", color="#6dbf67", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n} (n={c})" for n, c in zip(names, counts)], fontsize=8)
    ax.set_xlabel("F1 Score")
    ax.set_title("Keyword F1 by Cluster", fontweight="bold")
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save_fig(fig, out_dir / "keyword_prediction_by_cluster.png")

    # --- 3. F1 distribution histogram ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    exact_vals = [r["exact_f1"] for r in kw_results["details"]]
    soft_vals  = [r["soft_f1"] for r in kw_results["details"]]

    axes[0].hist(exact_vals, bins=20, color="#4f86c6", edgecolor="white", alpha=0.8)
    axes[0].axvline(np.mean(exact_vals), color="tomato", linestyle="--", label=f"mean={np.mean(exact_vals):.3f}")
    axes[0].set_title("Exact F1 Distribution", fontweight="bold")
    axes[0].set_xlabel("F1 Score")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].hist(soft_vals, bins=20, color="#6dbf67", edgecolor="white", alpha=0.8)
    axes[1].axvline(np.mean(soft_vals), color="tomato", linestyle="--", label=f"mean={np.mean(soft_vals):.3f}")
    axes[1].set_title("Soft F1 Distribution", fontweight="bold")
    axes[1].set_xlabel("F1 Score")
    axes[1].legend()

    plt.tight_layout()
    _save_fig(fig, out_dir / "keyword_f1_distribution.png")


def plot_retrieval_ablation(ret_results, out_dir):
    """Grouped bar chart comparing Hybrid vs FAISS vs BM25."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = ret_results["summaries"]
    metrics = ["MRR@15", "P@5", "P@10", "NDCG@15", "Cluster\nP@10"]
    keys    = ["mrr@15", "precision@5", "precision@10", "ndcg@15", "cluster_prec@10"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    w = 0.25

    colors = {"hybrid": "#2d5aa0", "faiss_only": "#4f86c6", "bm25_only": "#b0b0b0"}
    labels_map = {"hybrid": "Hybrid (FAISS+BM25)", "faiss_only": "FAISS-only", "bm25_only": "BM25-only"}

    for j, method in enumerate(["hybrid", "faiss_only", "bm25_only"]):
        vals = [summaries[method][k] for k in keys]
        bars = ax.bar(x + j * w, vals, w, label=labels_map[method], color=colors[method], edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x + w)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Ablation — Hybrid vs FAISS-only vs BM25-only", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save_fig(fig, out_dir / "retrieval_ablation.png")


def plot_doctor_eval(doc_results, out_dir):
    """Per-query evaluation chart for doctor mode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    details = doc_results["details"]
    topics = [d["topic"] for d in details]
    cl_prec = [d["cluster_precision@10"] for d in details]
    kw_over = [d["keyword_overlap"] for d in details]

    fig, ax = plt.subplots(figsize=(10, max(4, len(topics) * 0.5)))
    y = np.arange(len(topics))
    h = 0.35
    ax.barh(y + h/2, cl_prec, h, label="Cluster Precision@10", color="#2d5aa0", edgecolor="white")
    ax.barh(y - h/2, kw_over, h, label="Keyword Overlap", color="#6dbf67", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(topics, fontsize=9)
    ax.set_xlabel("Score")
    ax.set_title("Doctor Mode — Retrieval Quality per Topic", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save_fig(fig, out_dir / "doctor_mode_quality.png")

    # If LLM results available, add citation chart
    llm_details = [d for d in details if d["llm"] is not None]
    if llm_details:
        fig, ax = plt.subplots(figsize=(10, max(4, len(llm_details) * 0.5)))
        topics_llm = [d["topic"] for d in llm_details]
        cit_cov = [d["llm"]["citation_coverage"] for d in llm_details]
        n_cit   = [d["llm"]["n_citations"] for d in llm_details]

        y = np.arange(len(topics_llm))
        bars = ax.barh(y, cit_cov, color="#e07b39", edgecolor="white", height=0.6)
        for bar, nc in zip(bars, n_cit):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f"{nc} refs", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(topics_llm, fontsize=9)
        ax.set_xlabel("Citation Coverage (refs / retrieved papers)")
        ax.set_title("Doctor Mode — LLM Citation Quality", fontweight="bold")
        ax.set_xlim(0, 1.2)
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        _save_fig(fig, out_dir / "doctor_mode_citations.png")


def plot_answer_quality(judge_results, out_dir):
    """Radar-style bar chart for LLM-as-Judge scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if judge_results is None:
        return

    details = judge_results["details"]
    topics = [d["topic"] for d in details]
    dims = ["faithfulness", "relevance", "completeness"]
    colors_dim = {"faithfulness": "#2d5aa0", "relevance": "#6dbf67", "completeness": "#e07b39"}

    # --- Per-topic grouped bars ---
    fig, ax = plt.subplots(figsize=(12, max(4, len(topics) * 0.6)))
    y = np.arange(len(topics))
    h = 0.25
    for j, dim in enumerate(dims):
        scores = [d[dim].get("score", 0) for d in details]
        ax.barh(y + (j - 1) * h, scores, h, label=dim.capitalize(),
                color=colors_dim[dim], edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(topics, fontsize=9)
    ax.set_xlabel("Score (1-5)")
    ax.set_xlim(0, 5.5)
    ax.set_title("Answer Quality — LLM-as-Judge (RAGAS-style)", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add average line
    summ = judge_results["summary"]
    if "avg_overall" in summ:
        ax.axvline(summ["avg_overall"], color="tomato", linestyle="--", linewidth=1.5,
                   label=f"Overall avg: {summ['avg_overall']:.2f}")
        ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    _save_fig(fig, out_dir / "answer_quality_judge.png")

    # --- Summary bar ---
    if "avg_faithfulness" in summ:
        fig, ax = plt.subplots(figsize=(7, 4))
        dim_names = ["Faithfulness", "Relevance", "Completeness", "Overall"]
        dim_vals  = [summ["avg_faithfulness"], summ["avg_relevance"],
                     summ["avg_completeness"], summ["avg_overall"]]
        dim_cols  = ["#2d5aa0", "#6dbf67", "#e07b39", "#8b4dbd"]
        bars = ax.bar(dim_names, dim_vals, color=dim_cols, edgecolor="white", width=0.6)
        for bar, val in zip(bars, dim_vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                    f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 5.5)
        ax.set_ylabel("Score (1-5)")
        ax.set_title("Average Answer Quality Scores", fontweight="bold")
        ax.axhline(y=3, color="gray", linestyle=":", alpha=0.5, label="Neutral (3.0)")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        _save_fig(fig, out_dir / "answer_quality_summary.png")


def plot_summary_dashboard(kw_results, ret_results, doc_results, out_dir):
    """Single-page summary dashboard combining all metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Pipeline Evaluation Dashboard — Plan A (S-PubMedBERT + FAISS + BM25 Hybrid)",
                 fontweight="bold", fontsize=14, y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # --- Panel 1: Keyword F1 overview ---
    ax1 = fig.add_subplot(gs[0, 0])
    kw_s = kw_results["summary"]
    cats = ["Exact F1", "Soft F1"]
    vals = [kw_s["exact_f1_mean"], kw_s["soft_f1_mean"]]
    errs = [kw_s["exact_f1_std"], kw_s["soft_f1_std"]]
    ax1.bar(cats, vals, yerr=errs, color=["#4f86c6", "#6dbf67"], edgecolor="white", capsize=5)
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax1.text(i, v + e + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax1.set_title("Keyword Prediction F1", fontweight="bold", fontsize=10)
    ax1.set_ylim(0, max(vals) * 1.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --- Panel 2: Retrieval ablation ---
    ax2 = fig.add_subplot(gs[0, 1])
    ret_s = ret_results["summaries"]
    methods_display = ["Hybrid", "FAISS", "BM25"]
    metric_key = "ndcg@15"
    vals2 = [ret_s["hybrid"][metric_key], ret_s["faiss_only"][metric_key], ret_s["bm25_only"][metric_key]]
    cols2 = ["#2d5aa0", "#4f86c6", "#b0b0b0"]
    bars2 = ax2.bar(methods_display, vals2, color=cols2, edgecolor="white")
    for bar, v in zip(bars2, vals2):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_title("Retrieval NDCG@15 (Ablation)", fontweight="bold", fontsize=10)
    ax2.set_ylim(0, max(vals2) * 1.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # --- Panel 3: Doctor mode quality ---
    ax3 = fig.add_subplot(gs[0, 2])
    doc_s = doc_results["summary"]
    cats3 = ["Cluster\nPrecision", "Keyword\nOverlap"]
    vals3 = [doc_s["avg_cluster_precision"], doc_s["avg_keyword_overlap"]]
    if "avg_citation_coverage" in doc_s:
        cats3.append("Citation\nCoverage")
        vals3.append(doc_s["avg_citation_coverage"])
    bars3 = ax3.bar(cats3, vals3, color=["#2d5aa0", "#6dbf67", "#e07b39"][:len(cats3)], edgecolor="white")
    for bar, v in zip(bars3, vals3):
        ax3.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax3.set_title("Doctor Mode Quality", fontweight="bold", fontsize=10)
    ax3.set_ylim(0, 1.1)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # --- Panel 4: Per-cluster F1 horizontal bars ---
    ax4 = fig.add_subplot(gs[1, :2])
    cl_s = kw_results["cluster_summary"]
    sorted_cl = sorted(cl_s.items(), key=lambda x: x[1]["n"], reverse=True)
    names = [f"{c[:30]} (n={cs['n']})" for c, cs in sorted_cl]
    exact = [cs["exact_f1_mean"] for _, cs in sorted_cl]
    soft  = [cs["soft_f1_mean"] for _, cs in sorted_cl]
    y = np.arange(len(names))
    ax4.barh(y + 0.17, exact, 0.34, label="Exact", color="#4f86c6")
    ax4.barh(y - 0.17, soft, 0.34, label="Soft", color="#6dbf67")
    ax4.set_yticks(y)
    ax4.set_yticklabels(names, fontsize=7)
    ax4.set_title("Keyword F1 by Cluster", fontweight="bold", fontsize=10)
    ax4.legend(fontsize=8)
    ax4.invert_yaxis()
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # --- Panel 5: Doctor mode per-topic ---
    ax5 = fig.add_subplot(gs[1, 2])
    doc_d = doc_results["details"]
    topics = [d["topic"] for d in doc_d]
    kw_ov = [d["keyword_overlap"] for d in doc_d]
    y5 = np.arange(len(topics))
    ax5.barh(y5, kw_ov, color="#6dbf67", edgecolor="white", height=0.6)
    ax5.set_yticks(y5)
    ax5.set_yticklabels(topics, fontsize=7)
    ax5.set_title("Doctor: Keyword Overlap", fontweight="bold", fontsize=10)
    ax5.invert_yaxis()
    ax5.set_xlim(0, 1.05)
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)

    _save_fig(fig, out_dir / "evaluation_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def save_results(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types for JSON serialisation
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=convert)
    print(f"  Results saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline Evaluation & Benchmarking")
    parser.add_argument("--suite", choices=["keyword", "retrieval", "doctor", "judge", "all"],
                        default="all", help="Which eval suite to run")
    parser.add_argument("--sample", type=int, default=200,
                        help="Sample size for keyword/retrieval eval (default 200)")
    args = parser.parse_args()

    # Load index + embedder once
    import pickle
    from lib.embedder import Embedder
    from lib.retriever import load_index

    print("Loading embedder and index ...")
    embedder = Embedder(EMBED_MODEL, hf_token=HF_TOKEN)
    index, meta, bm25 = load_index(FAISS_INDEX, FAISS_META, BM25_PATH)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    t0 = time.time()

    # --- SUITE A: Keywords ---
    if args.suite in ("keyword", "all"):
        kw_results = eval_keyword_prediction(meta, index, bm25, embedder, sample_n=args.sample)
        save_results(kw_results, EVAL_DIR / "keyword_eval.json")
        plot_keyword_eval(kw_results, PLOTS_DIR)
        all_results["keyword"] = kw_results

    # --- SUITE B: Retrieval ---
    if args.suite in ("retrieval", "all"):
        ret_results = eval_retrieval_ablation(meta, index, bm25, embedder, sample_n=args.sample)
        save_results(ret_results, EVAL_DIR / "retrieval_eval.json")
        plot_retrieval_ablation(ret_results, PLOTS_DIR)
        all_results["retrieval"] = ret_results

    # --- SUITE C: Doctor ---
    if args.suite in ("doctor", "all"):
        doc_results = eval_doctor_mode(meta, index, bm25, embedder)
        save_results(doc_results, EVAL_DIR / "doctor_eval.json")
        plot_doctor_eval(doc_results, PLOTS_DIR)
        all_results["doctor"] = doc_results

    # --- SUITE D: LLM-as-Judge ---
    if args.suite in ("judge", "all"):
        judge_results = eval_answer_quality(meta, index, bm25, embedder)
        if judge_results:
            save_results(judge_results, EVAL_DIR / "judge_eval.json")
            plot_answer_quality(judge_results, PLOTS_DIR)
            all_results["judge"] = judge_results

    # --- Dashboard ---
    if args.suite == "all" and "keyword" in all_results and "retrieval" in all_results and "doctor" in all_results:
        plot_summary_dashboard(all_results["keyword"], all_results["retrieval"],
                              all_results["doctor"], PLOTS_DIR)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  EVALUATION COMPLETE  ({elapsed:.0f}s)")
    print(f"  Results: {EVAL_DIR}")
    print(f"  Plots:   {PLOTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
