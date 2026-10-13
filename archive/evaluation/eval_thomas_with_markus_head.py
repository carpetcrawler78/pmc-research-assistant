from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


LABELS = ["yes", "no", "maybe"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Thomas retrieval + Markus-style yes/no/maybe evaluation"
    )
    p.add_argument("--questions", default="data/golden69_questions.pkl")
    p.add_argument("--rag-py", default="python/04_rag_pipeline.py")
    p.add_argument("--top-k-faiss", type=int, default=3)
    p.add_argument("--top-k-bm25", type=int, default=3)
    p.add_argument("--top-k-output", type=int, default=1)
    p.add_argument("--sleep", type=float, default=2.0)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def parse_answer(text: str) -> str:
    text = str(text).strip().lower()
    if re.search(r"\byes\b", text):
        return "yes"
    if re.search(r"\bno\b", text):
        return "no"
    if re.search(r"\bmaybe\b", text):
        return "maybe"
    return "unknown"


def load_module(py_path: Path):
    spec = importlib.util.spec_from_file_location("rag_pipeline_mod", py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_markus_decision_prompt(question: str, retrieved: list[dict]) -> str:
    papers = []
    for i, d in enumerate(retrieved[:10], 1):
        title = d.get("title", "?")
        pmcid = d.get("pmcid", "?")
        excerpt = (d.get("chunk_text") or "")[:600]
        papers.append(
            f"[{i}] Title: {title}\n"
            f"    PMCID: {pmcid}\n"
            f"    Excerpt: {excerpt}"
        )

    papers_block = "\n\n".join(papers) if papers else "No retrieved evidence."

    return f"""You are answering a biomedical PubMedQA-style question.

QUESTION:
{question}

EVIDENCE:
{papers_block}

Task:
Return exactly one word:
yes
no
maybe

Rules:
- Answer "yes" if the evidence supports the question.
- Answer "no" if the evidence contradicts the question.
- Answer "maybe" if the evidence is mixed, unclear, or insufficient.
- Return only one word: yes, no, or maybe.
""".strip()


def main():
    args = parse_args()

    questions_path = Path(args.questions)
    rag_py_path = Path(args.rag_py)

    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    if not rag_py_path.exists():
        raise FileNotFoundError(f"RAG file not found: {rag_py_path}")

    df = pd.read_pickle(questions_path)
    if args.limit is not None:
        df = df.head(args.limit)

    mod = load_module(rag_py_path)
    RAGPipeline = mod.RAGPipeline
    query_llm_text = mod.query_llm_text
    LLM_MODEL = mod.LLM_MODEL

    rag = RAGPipeline()

    true_labels = []
    pred_labels = []
    answered = 0
    skipped = 0

    print(f"Model        : {LLM_MODEL}")
    print(f"Questions    : {len(df)}")
    print(f"Sleep/call   : {args.sleep:.1f}s")
    print("Context mode : THOMAS RETRIEVAL + MARKUS DECISION")
    print()

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        question = str(row["question"])
        gold = str(row["final_decision"]).strip().lower()

        try:
            result = rag.doctor_query(
                query=question,
                top_k_faiss=args.top_k_faiss,
                top_k_bm25=args.top_k_bm25,
                top_k_output=args.top_k_output,
            )

            prompt = build_markus_decision_prompt(question, result["retrieved"])
            raw_decision = query_llm_text(prompt, model=LLM_MODEL)
            pred = parse_answer(raw_decision)

            if pred == "unknown":
                skipped += 1
            else:
                answered += 1

        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate limit" in msg.lower():
                print(f"[{idx:02d}/{len(df)}] rate limit reached, stopping early")
                break
            pred = "unknown"
            skipped += 1

        true_labels.append(gold)
        pred_labels.append(pred)

        short_q = question[:70] + "..." if len(question) > 70 else question
        print(f"[{idx:02d}/{len(df)}] {short_q}")
        print(f"         gold={gold:<7} pred={pred}")

        time.sleep(args.sleep)

    valid_idx = [i for i, p in enumerate(pred_labels) if p in LABELS]
    true_valid = [true_labels[i] for i in valid_idx]
    pred_valid = [pred_labels[i] for i in valid_idx]

    print("\n" + "=" * 60)
    print("THOMAS RETRIEVAL + MARKUS DECISION EVALUATION")
    print("=" * 60)
    print(f"Answered    : {answered}/{len(true_labels)}  (skipped={skipped} unknowns)")

    if not true_valid:
        print("Accuracy    : n/a")
        print("\n  yes   : precision=0.000  recall=0.000  f1=0.000")
        print("  no    : precision=0.000  recall=0.000  f1=0.000")
        print("  maybe : precision=0.000  recall=0.000  f1=0.000")
        print("\nConfusion matrix (rows=actual, cols=pred):")
        print("             yes      no   maybe")
        print("  yes   :      0       0       0")
        print("  no    :      0       0       0")
        print("  maybe :      0       0       0")
        return

    acc = accuracy_score(true_valid, pred_valid)
    print(f"Accuracy    : {acc:.4f}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_valid,
        pred_valid,
        labels=LABELS,
        average=None,
        zero_division=0,
    )

    print()
    for i, label in enumerate(LABELS):
        print(
            f"  {label:<5}: precision={precision[i]:.3f}  "
            f"recall={recall[i]:.3f}  f1={f1[i]:.3f}"
        )

    cm = confusion_matrix(true_valid, pred_valid, labels=LABELS)

    print("\nConfusion matrix (rows=actual, cols=pred):")
    print("             yes      no   maybe")
    for i, label in enumerate(LABELS):
        print(
            f"  {label:<5}:"
            f" {cm[i, 0]:6d} {cm[i, 1]:7d} {cm[i, 2]:7d}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)