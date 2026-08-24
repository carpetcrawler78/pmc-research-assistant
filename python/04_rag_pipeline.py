"""
04_rag_pipeline.py
Capstone 1 — PMC Research Assistant  v2

STAGE 4 — Hybrid RAG pipeline with two modes.

Mode 1 — keyword (default):
  Given a new paper (title + abstract), predicts MeSH-style keywords.
  Uses retrieval-based aggregation + optional LLM refinement.

Mode 2 — doctor:
  Given a natural-language clinical/research question, generates an
  evidence-based answer citing specific papers from the PMC corpus.
  Requires OPENAI_API_KEY.

Both modes share the same retrieval core:
  1. Embed query with S-PubMedBERT-MS-MARCO
  2. Retrieve top-k via FAISS + BM25 (RRF fusion)
  3. Mode-specific output (keywords or narrative answer)

Input:
  big_data/faiss_v2/pubmedbert.index
  big_data/faiss_v2/pubmedbert_meta.pkl
  big_data/faiss_v2/bm25.pkl

Output: keyword predictions or evidence-based answer (console + optional JSON)

Usage:
    python python/04_rag_pipeline.py                         # keyword mode (interactive)
    python python/04_rag_pipeline.py --mode doctor           # doctor mode  (interactive)
    python python/04_rag_pipeline.py --title "..." --abstract "..."
    python python/04_rag_pipeline.py --mode doctor --query "What are current CRISPR approaches for cancer?"
    python python/04_rag_pipeline.py --llm
    python python/04_rag_pipeline.py --json-out results.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from dotenv import load_dotenv

from config import (
    BM25_PATH,
    EMBED_MODEL,
    FAISS_INDEX,
    FAISS_META,
    GROQ_KEY,
    HF_TOKEN,
    LLM_MODEL,
    LLM_PROVIDER,
    MIN_SCORE,
    OPENAI_KEY,
    PROJECT_ROOT,
    RRF_K,
    TFIDF_MODEL,
    TOP_K_BM25,
    TOP_K_FAISS,
    TOP_K_OUTPUT,
    TOP_K_TFIDF_DIRECT,
)
from lib.embedder import Embedder
from lib.plotter import plot_keyword_scores, plot_retrieval_scores
from lib.retriever import load_index, retrieve

PLOTS_DIR = PROJECT_ROOT / "plots"

load_dotenv()


# ── Keyword normalisation ─────────────────────────────────────────────────────

def _norm(kw: str) -> str:
    """Lowercase, strip apostrophe artefacts, collapse whitespace."""
    kw = kw.lower().strip()
    kw = re.sub(r"['\u2019\u2018`]s?\b", "", kw)
    kw = re.sub(r"\s+", " ", kw)
    return kw.strip()


def normalise_counter(counter: Counter) -> Counter:
    """Merge trailing-s duplicates and apostrophe artefacts."""
    out: Counter = Counter()
    for kw, cnt in counter.items():
        base = _norm(kw)
        stem = base.rstrip("s") if base.endswith("s") and len(base) > 3 else base
        canonical = stem if stem in {_norm(k) for k in counter} else base
        out[canonical] += cnt
    return out


# ── TF-IDF direct prediction ──────────────────────────────────────────────────

def tfidf_predict(title: str, abstract: str, top_k: int) -> list[str]:
    """Run baseline TF-IDF model directly on query; returns list of keywords."""
    if not TFIDF_MODEL.exists():
        return []
    try:
        import pickle
        with open(TFIDF_MODEL, "rb") as f:
            bundle = pickle.load(f)
        tfidf, svd, clf, mlb = (
            bundle["tfidf"], bundle["svd"], bundle["clf"], bundle["mlb"]
        )
        vec   = svd.transform(tfidf.transform([f"{title} {abstract}"]))
        proba = clf.predict_proba(vec)[0]
        top_i = proba.argsort()[::-1][:top_k]
        return [mlb.classes_[i] for i in top_i if proba[i] > 0]
    except Exception as e:
        print(f"  TF-IDF direct prediction failed: {e}")
        return []


# ── Keyword aggregation ───────────────────────────────────────────────────────

def aggregate_keywords(
    retrieved:    list[dict],
    direct_tfidf: list[str],
    top_k:        int,
) -> list[tuple[str, float]]:
    """
    Score-weighted aggregation:
      kwd_lst_cl       (ground-truth keywords of retrieved paper)  x2 x score
      tfidf_candidates (baseline predictions stored per article)   x1 x score
      direct_tfidf     (direct prediction on the query itself)     x1.5 (flat)
    """
    counter: Counter = Counter()
    for doc in retrieved:
        score = doc.get("retrieval_score", 0.0)
        for kw in doc.get("kwd_lst_cl", []):
            counter[_norm(kw)] += 2.0 * score
        for kw in doc.get("tfidf_candidates", []):
            counter[_norm(kw)] += 1.0 * score
    for kw in direct_tfidf:
        counter[_norm(kw)] += 1.5
    counter = normalise_counter(counter)
    return counter.most_common(top_k)


# ── LLM ──────────────────────────────────────────────────────────────────────

def build_llm_prompt(
    title:      str,
    abstract:   str,
    retrieved:  list[dict],
    candidates: list[tuple[str, float]],
) -> str:
    ctx = "\n".join(
        f"  [{i}] cluster: {d.get('cluster_name','?')}\n"
        f"       title: {(d.get('title') or '')[:100]}\n"
        f"       keywords: {', '.join(d.get('kwd_lst_cl', [])[:8])}"
        for i, d in enumerate(retrieved[:5], 1)
    )
    cands = ", ".join(kw for kw, _ in candidates)
    return f"""You are a scientific keyword expert for biomedical literature.

A researcher submitted the following paper:

TITLE: {title}

ABSTRACT:
{abstract}

I retrieved the {len(retrieved)} most similar papers from a PMC corpus.
Top 5 retrieved papers (cluster, title, keywords):

{ctx}

Keyword candidates ranked by retrieval frequency:
{cands}

Task: Return exactly 10 MeSH-style keywords describing the submitted paper.
- Prefer specific, informative terms over generic ones.
- Draw from the candidate list but extend or correct it if needed.
- Return ONLY a JSON array of 10 strings, nothing else.
Example: ["keyword1", "keyword2", ..., "keyword10"]
"""


def _get_llm_client():
    """Return (client, model_name) for whichever LLM provider is configured."""
    from openai import OpenAI   # openai SDK works for both OpenAI and Groq
    if GROQ_KEY:
        return OpenAI(api_key=GROQ_KEY, base_url="https://api.groq.com/openai/v1"), LLM_MODEL
    if OPENAI_KEY:
        return OpenAI(api_key=OPENAI_KEY), LLM_MODEL
    return None, None


def query_llm(prompt: str, model: str = LLM_MODEL) -> list[str]:
    client, model_name = _get_llm_client()
    if client is None:
        print("  WARNING: No LLM API key set (GROQ_API_KEY or OPENAI_API_KEY) — skipping.")
        return []
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"  LLM call failed: {e}")
        return []


# ── Doctor-mode LLM ─────────────────────────────────────────────────────────

def build_doctor_prompt(query: str, retrieved: list[dict]) -> str:
    """Build a prompt that turns retrieved papers into an evidence-based answer."""
    papers = "\n\n".join(
        f"[{i}] Title: {d.get('title', '?')}\n"
        f"    PMCID: {d.get('pmcid', '?')}\n"
        f"    Cluster: {d.get('cluster_name', '?')}\n"
        f"    Keywords: {', '.join(d.get('kwd_lst_cl', [])[:8])}\n"
        f"    Excerpt: {(d.get('chunk_text') or '')[:400]}"
        for i, d in enumerate(retrieved[:10], 1)
    )
    return f"""You are a biomedical research assistant helping a clinician.

QUESTION:
{query}

I searched a corpus of {len(retrieved)} PubMed Central articles and found the
following relevant papers. Use ONLY these sources to answer the question.

RETRIEVED PAPERS:
{papers}

INSTRUCTIONS:
1. Write a clear, evidence-based answer (3-5 paragraphs) in English.
2. Cite papers by their number [1], [2], etc. throughout your answer.
3. After the answer, list all cited papers with their full title and PMCID.
4. If the evidence is insufficient to fully answer, say so honestly.
5. Do NOT invent information that is not in the provided papers.

FORMAT:
Answer first, then a section "REFERENCES:" with numbered papers.
"""


def query_llm_text(prompt: str, model: str = LLM_MODEL) -> str:
    """Send prompt to LLM and return raw text (for doctor mode)."""
    client, model_name = _get_llm_client()
    if client is None:
        print("  WARNING: No LLM API key set (GROQ_API_KEY or OPENAI_API_KEY) — cannot use doctor mode.")
        return ""
    try:
        provider = "Groq" if GROQ_KEY else "OpenAI"
        print(f"  Sending to {provider} ({model_name}) ...")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  LLM call failed: {e}")
        return ""


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAGPipeline:
    """Load index once; run many queries interactively."""

    def __init__(self) -> None:
        self.embedder = Embedder(EMBED_MODEL, hf_token=HF_TOKEN)
        self.index, self.meta, self.bm25 = load_index(
            FAISS_INDEX, FAISS_META, BM25_PATH
        )

    def predict(
        self,
        title:        str,
        abstract:     str,
        top_k_faiss:  int  = TOP_K_FAISS,
        top_k_bm25:   int  = TOP_K_BM25,
        top_k_output: int  = TOP_K_OUTPUT,
        use_llm:      bool = False,
        llm_model:    str  = LLM_MODEL,
    ) -> dict:
        query_text = f"{title} {abstract}"
        query_vec  = self.embedder.encode_single(query_text)

        retrieved = retrieve(
            query_text=query_text,
            query_vec=query_vec,
            index=self.index,
            meta=self.meta,
            bm25=self.bm25,
            top_k_faiss=top_k_faiss,
            top_k_bm25=top_k_bm25,
            top_k_output=top_k_output,
            min_score=MIN_SCORE,
            rrf_k=RRF_K,
        )

        direct_tfidf = tfidf_predict(title, abstract, TOP_K_TFIDF_DIRECT)
        candidates   = aggregate_keywords(retrieved, direct_tfidf, top_k_output)

        llm_keywords = []
        if use_llm:
            prompt       = build_llm_prompt(title, abstract, retrieved, candidates)
            llm_keywords = query_llm(prompt, model=llm_model)

        return {
            "retrieved":           retrieved,
            "retrieval_keywords":  [kw for kw, _ in candidates],
            "retrieval_candidates": candidates,   # (keyword, score) for plotting
            "direct_tfidf":        direct_tfidf,
            "llm_keywords":        llm_keywords,
        }

    def doctor_query(
        self,
        query:        str,
        top_k_faiss:  int = TOP_K_FAISS,
        top_k_bm25:   int = TOP_K_BM25,
        top_k_output: int = TOP_K_OUTPUT,
        llm_model:    str = LLM_MODEL,
    ) -> dict:
        """Doctor mode: answer a clinical/research question with evidence."""
        query_vec = self.embedder.encode_single(query)

        retrieved = retrieve(
            query_text=query,
            query_vec=query_vec,
            index=self.index,
            meta=self.meta,
            bm25=self.bm25,
            top_k_faiss=top_k_faiss,
            top_k_bm25=top_k_bm25,
            top_k_output=top_k_output,
            min_score=MIN_SCORE,
            rrf_k=RRF_K,
        )

        prompt = build_doctor_prompt(query, retrieved)
        answer = query_llm_text(prompt, model=llm_model)

        return {
            "query":      query,
            "retrieved":  retrieved,
            "answer":     answer,
        }


# ── Output ────────────────────────────────────────────────────────────────────

def print_result(result: dict, title: str, abstract: str) -> None:
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"QUERY  : {title[:80]}")
    print(sep)
    print(f"\nRETRIEVED PAPERS ({len(result['retrieved'])})")
    for i, d in enumerate(result["retrieved"], 1):
        print(f"  {i:2}. [{d.get('retrieval_score',0):.4f}]"
              f" [{d.get('cluster_name','?')}]  {(d.get('title') or '')[:65]}")
    print(f"\nRETRIEVAL KEYWORDS")
    for i, kw in enumerate(result["retrieval_keywords"], 1):
        print(f"  {i:2}. {kw}")
    if result["direct_tfidf"]:
        print(f"\nDIRECT TF-IDF")
        for i, kw in enumerate(result["direct_tfidf"], 1):
            print(f"  {i:2}. {kw}")
    if result["llm_keywords"]:
        print(f"\nLLM-REFINED")
        for i, kw in enumerate(result["llm_keywords"], 1):
            print(f"  {i:2}. {kw}")
    print(sep)


def print_doctor_result(result: dict) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"QUESTION: {result['query']}")
    print(sep)
    print(f"\nRETRIEVED PAPERS ({len(result['retrieved'])})")
    for i, d in enumerate(result["retrieved"], 1):
        print(f"  {i:2}. [{d.get('retrieval_score',0):.4f}]"
              f"  {(d.get('title') or '')[:65]}")
        print(f"       PMCID: {d.get('pmcid','?')}  |  Cluster: {d.get('cluster_name','?')}")
    print(f"\n{'─' * 60}")
    print("ANSWER:\n")
    print(result["answer"])
    print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PMC RAG pipeline  v2  (keyword + doctor mode)")
    p.add_argument("--mode",     choices=["keyword", "doctor"], default="keyword",
                   help="keyword: predict keywords for a paper | doctor: answer a research question")
    p.add_argument("--query",    default=None, help="(doctor mode) the question to answer")
    p.add_argument("--title",    default=None, help="(keyword mode) paper title")
    p.add_argument("--abstract", default=None, help="(keyword mode) paper abstract")
    p.add_argument("--top-k",    type=int, default=TOP_K_FAISS)
    p.add_argument("--output-k", type=int, default=TOP_K_OUTPUT)
    p.add_argument("--llm",      action="store_true", help="(keyword mode) refine with LLM")
    p.add_argument("--json-out", default=None, metavar="PATH")
    return p.parse_args()


def main():
    args = parse_args()
    rag  = RAGPipeline()

    # ── DOCTOR MODE ──────────────────────────────────────────────────────
    if args.mode == "doctor":
        query = args.query
        if query is None:
            print("Enter your research question (or press Enter for demo):")
            query = input("  Question: ").strip()
            if not query:
                query = "What are the latest CRISPR-Cas9 approaches for treating cancer?"
                print(f"\n  [Demo] {query}")

        while True:
            print(f"\nSearching for: {query[:80]} ...")
            result = rag.doctor_query(
                query=query,
                top_k_faiss=args.top_k,
                top_k_output=args.output_k,
            )
            print_doctor_result(result)

            # Plot
            safe = re.sub(r"[^\w]", "_", query[:30]).strip("_")
            plot_retrieval_scores(
                result["retrieved"], query, PLOTS_DIR,
                fname=f"doctor_{safe}.png",
            )

            if args.json_out:
                out = {
                    "mode":     "doctor",
                    "query":    query,
                    "answer":   result["answer"],
                    "retrieved": [
                        {k: d.get(k) for k in
                         ("pmcid", "title", "cluster_name", "kwd_lst_cl",
                          "retrieval_score")}
                        for d in result["retrieved"]
                    ],
                }
                with open(args.json_out, "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2, ensure_ascii=False)
                print(f"Saved to {args.json_out}")

            print("\nNext question? (press Enter to exit)")
            query = input("  Question: ").strip()
            if not query:
                break
        return

    # ── KEYWORD MODE (default) ───────────────────────────────────────────
    title    = args.title
    abstract = args.abstract

    if title is None:
        print("Enter paper title (or press Enter for demo):")
        title = input("  Title: ").strip()
        if not title:
            title    = "LRRK2 mutations and their downstream effects on dopamine neurons in Parkinson disease"
            abstract = (
                "We investigated the role of LRRK2 G2019S gain-of-function mutations "
                "in mitochondrial dysfunction and dopaminergic neuron loss. Using iPSC-derived "
                "neurons, we show that LRRK2 hyperactivation impairs mitophagy, increases "
                "oxidative stress, and accelerates apoptosis. Pharmacological inhibition of "
                "LRRK2 kinase activity partially rescued neuron viability and restored "
                "mitochondrial membrane potential, suggesting LRRK2 as a therapeutic target."
            )
            print(f"\n  [Demo] {title}")

    if abstract is None:
        print("Enter abstract:")
        abstract = input("  Abstract: ").strip()

    # Interactive loop — model and index loaded once
    while True:
        result = rag.predict(
            title=title,
            abstract=abstract,
            top_k_faiss=args.top_k,
            top_k_output=args.output_k,
            use_llm=args.llm,
        )
        print_result(result, title, abstract)

        # ── Plots ─────────────────────────────────────────────────────────
        safe = re.sub(r"[^\w]", "_", title[:30]).strip("_")
        plot_retrieval_scores(
            result["retrieved"], title, PLOTS_DIR,
            fname=f"retrieval_{safe}.png",
        )
        plot_keyword_scores(
            result["retrieval_candidates"],
            PLOTS_DIR,
            fname=f"keywords_{safe}.png",
        )

        if args.json_out:
            out = {
                "mode":               "keyword",
                "title":              title,
                "abstract":           abstract,
                "retrieval_keywords": result["retrieval_keywords"],
                "llm_keywords":       result["llm_keywords"],
                "retrieved": [
                    {k: d.get(k) for k in
                     ("title", "cluster_name", "kwd_lst_cl", "tfidf_candidates",
                      "retrieval_score")}
                    for d in result["retrieved"]
                ],
            }
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"Saved to {args.json_out}")

        print("\nNext query? (press Enter to exit)")
        title = input("  Title: ").strip()
        if not title:
            break
        abstract = input("  Abstract: ").strip()


if __name__ == "__main__":
    main()
