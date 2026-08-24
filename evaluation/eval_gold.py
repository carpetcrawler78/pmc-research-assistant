"""
eval_gold.py — Gold-Context PubMedQA evaluator
===============================================
Uses the gold abstract context stored INSIDE golden69_questions.pkl
(the 'context' column with original paper snippets), exactly like Markus does.

This isolates LLM quality from retrieval quality:
  - eval_gold.py  → Gold Context  + 8B LLM  (no retrieval needed)
  - eval.py       → RAG Retrieval + 8B LLM  (must find the right chunk)
  - Markus        → Gold Context  + fine-tuned flan-t5-large (supervised)

Run:
    python eval_gold.py
    python eval_gold.py --model llama-3.3-70b-versatile   # if TPD available
    python eval_gold.py --questions data/golden69_questions.pkl
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

load_dotenv()

LABELS = ["yes", "no", "maybe"]
DEFAULT_MODEL = "llama-3.1-8b-instant"   # separate 20k-TPM Groq budget
MAX_CONTEXT_CHARS = 1800                  # generous — gold context, no retrieval


def parse_args():
    p = argparse.ArgumentParser(description="Gold-context PubMedQA evaluation.")
    p.add_argument("--questions", default="data/golden69_questions.pkl")
    p.add_argument("--model",     default=DEFAULT_MODEL,
                   help="Groq model name (default: llama-3.1-8b-instant)")
    p.add_argument("--sleep",     type=float, default=3.0,
                   help="Seconds to sleep between Groq calls (default: 3)")
    p.add_argument("--sample",    type=int,   default=None,
                   help="Evaluate only N questions (for quick testing)")
    return p.parse_args()


def build_context_text(ctx_field) -> str:
    """
    The 'context' column is a dict with key 'contexts' → list[str].
    Join them into a single text block for the LLM.
    """
    if isinstance(ctx_field, dict):
        snippets = ctx_field.get("contexts", [])
        return " ".join(snippets)
    if isinstance(ctx_field, str):
        return ctx_field
    return ""


def classify(client, model: str, question: str, context: str, sleep_s: float) -> str:
    """
    Send question + gold context to LLM → parse yes/no/maybe.
    Prompt is more decisive than eval.py to reduce MAYBE over-prediction.
    """
    ctx_trunc = context[:MAX_CONTEXT_CHARS]

    prompt = (
        "You are a biomedical question answering system.\n"
        "Read the context and answer the question with EXACTLY ONE WORD: yes, no, or maybe.\n"
        "Rules:\n"
        "- Answer 'yes' if the context clearly supports the claim.\n"
        "- Answer 'no' if the context clearly contradicts or does not support it.\n"
        "- Answer 'maybe' ONLY if the evidence is genuinely mixed or incomplete.\n"
        "- Do NOT add any other words or punctuation.\n\n"
        f"Context:\n{ctx_trunc}\n\n"
        f"Question: {question}\n\n"
        "Answer (yes/no/maybe):"
    )

    for attempt in range(3):
        try:
            time.sleep(sleep_s + attempt * 5)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip().lower()
            # Parse robustly
            if re.search(r"\byes\b", raw):
                return "yes"
            if re.search(r"\bno\b", raw):
                return "no"
            if re.search(r"\bmaybe\b", raw):
                return "maybe"
            return "unknown"
        except Exception as e:
            msg = str(e).lower()
            if "tokens per day" in msg or "rate_limit_exceeded" in msg:
                print(f"  [TPD limit reached at attempt {attempt+1}]")
                return "unknown"
            print(f"  [Attempt {attempt+1}/3 failed: {type(e).__name__}]")
            if attempt < 2:
                time.sleep(15)
    return "unknown"


def main():
    args = parse_args()

    # --- Groq client ---
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        print("ERROR: GROQ_API_KEY not set in .env / environment.")
        return

    try:
        from openai import OpenAI
        client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
    except ImportError:
        print("Run: pip install openai")
        return

    # --- Load questions ---
    qpath = Path(args.questions)
    if not qpath.exists():
        print(f"ERROR: File not found: {qpath}")
        return

    df = pd.read_pickle(qpath).reset_index(drop=True)   # reset so i+1 == row number
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42).reset_index(drop=True)

    print(f"Model        : {args.model}")
    print(f"Questions    : {len(df)}")
    print(f"Sleep/call   : {args.sleep}s")
    print(f"Context mode : GOLD (from pickle, no retrieval)")
    print(f"Label dist   : {df['final_decision'].value_counts().to_dict()}")
    print()

    preds, golds = [], []

    for i, row in df.iterrows():
        q       = row["question"]
        label   = row["final_decision"].strip().lower()
        ctx_raw = row.get("context", "")
        ctx     = build_context_text(ctx_raw)

        print(f"[{i+1:02d}/{len(df)}] {q[:70]}...")
        if not ctx:
            print("         [!] No context found — predicting 'unknown'")
            pred = "unknown"
        else:
            pred = classify(client, args.model, q, ctx, args.sleep)

        print(f"         gold={label:<6} pred={pred}")
        preds.append(pred)
        golds.append(label)

    # --- Filter unknowns ---
    valid = [(g, p) for g, p in zip(golds, preds) if p != "unknown"]
    if not valid:
        print("\nNo valid predictions.")
        return

    g_clean, p_clean = zip(*valid)
    skipped = len(golds) - len(valid)

    # --- Metrics ---
    acc = accuracy_score(g_clean, p_clean)
    prec, rec, f1, _ = precision_recall_fscore_support(
        g_clean, p_clean, labels=LABELS, zero_division=0
    )
    cm = confusion_matrix(g_clean, p_clean, labels=LABELS)

    print("\n" + "=" * 60)
    print(f"GOLD-CONTEXT EVALUATION — {args.model}")
    print("=" * 60)
    print(f"Answered    : {len(valid)}/{len(golds)}  (skipped={skipped} unknowns)")
    print(f"Accuracy    : {acc:.4f}  (Markus baseline: 0.8261 with fine-tuned flan-t5)")
    print()
    for i, lbl in enumerate(LABELS):
        print(f"  {lbl:<6}: precision={prec[i]:.3f}  recall={rec[i]:.3f}  f1={f1[i]:.3f}")
    print()
    print("Confusion matrix (rows=actual, cols=pred):")
    print(f"          {'  '.join(f'{l:>6}' for l in LABELS)}")
    for i, lbl in enumerate(LABELS):
        print(f"  {lbl:<6}: {'  '.join(f'{cm[i,j]:6d}' for j in range(len(LABELS)))}")

    print()
    print("Interpretation:")
    rag_acc = 0.3333  # from eval.py run above
    gold_acc = acc
    gap = gold_acc - rag_acc
    print(f"  Gold-context  acc = {gold_acc:.4f}  (this script)")
    print(f"  RAG-retrieval acc = {rag_acc:.4f}  (eval.py)")
    print(f"  Retrieval gap     = {gap:+.4f}  ({'retrieval is bottleneck' if gap > 0.05 else 'LLM is bottleneck'})")
    print(f"  vs Markus (supervised flan-t5): {acc - 0.8261:+.4f}")
    print()
    print("NOTE: Markus uses fine-tuned flan-t5-large (supervised, trained on PubMedQA).")
    print("      This script uses zero-shot prompting — no training, no task-specific tuning.")


if __name__ == "__main__":
    main()
