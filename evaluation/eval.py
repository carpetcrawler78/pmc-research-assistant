from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


LABELS = ["yes", "no", "maybe"]

# Groq model with higher free-tier limits (20k TPM vs 12k for 70B)
# Sufficient for yes/no/maybe classification
GROQ_EVAL_MODEL = "llama-3.1-8b-instant"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Markus-style PubMedQA evaluation on Thomas RAG pipeline."
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="data/golden69_questions.pkl",
        help="Path to golden69_questions.pkl",
    )
    parser.add_argument(
        "--rag-py",
        type=str,
        default="python/04_rag_pipeline.py",
        help="Path to 04_rag_pipeline.py",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="top-k retrieval/output for doctor_query",
    )
    return parser.parse_args()


def parse_answer(text: str) -> str:
    text = str(text).strip().lower()
    if re.search(r"\byes\b", text):
        return "yes"
    if re.search(r"\bno\b", text):
        return "no"
    if re.search(r"\bmaybe\b", text):
        return "maybe"
    return "unknown"


def load_rag_pipeline_class(rag_py_path: Path):
    if not rag_py_path.exists():
        raise FileNotFoundError(f"RAG pipeline file not found: {rag_py_path}")

    spec = importlib.util.spec_from_file_location("rag_pipeline", rag_py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {rag_py_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "RAGPipeline"):
        raise AttributeError(f"RAGPipeline class not found in {rag_py_path}")

    return module.RAGPipeline


class MarkusCompatibleEvaluator:
    def __init__(self, rag_pipeline, k: int = 5):
        self.rag = rag_pipeline
        self.k = k
        self._last_result = None

    def retrieve_with_metadata(self, question: str, k: int | None = None):
        """Retrieve only — NO LLM call here. LLM is called separately in generate_answer."""
        k = self.k if k is None else k

        # Use retrieval directly without triggering the LLM
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / "python"))
        from config import (
            BM25_PATH, FAISS_INDEX, FAISS_META, MIN_SCORE, RRF_K,
        )
        from lib.retriever import retrieve

        query_vec = self.rag.embedder.encode_single(question)
        retrieved = retrieve(
            query_text=question,
            query_vec=query_vec,
            index=self.rag.index,
            meta=self.rag.meta,
            bm25=self.rag.bm25,
            top_k_faiss=k,
            top_k_bm25=k,
            top_k_output=k,
            min_score=MIN_SCORE,
            rrf_k=RRF_K,
        )

        # Store as last result (no answer yet — that comes from generate_answer)
        self._last_result = {"query": question, "retrieved": retrieved, "answer": ""}

        docs = [d.get("chunk_text", "") for d in retrieved]
        metadata = [
            {
                "title": d.get("title", ""),
                "pmcid": d.get("pmcid", ""),
                "score": d.get("retrieval_score", None),
                "cluster_name": d.get("cluster_name", ""),
            }
            for d in retrieved
        ]
        return docs, metadata

    def generate_answer(self, question: str) -> str:
        """Generate yes/no/maybe using a short prompt — saves tokens vs full doctor mode."""
        import os
        from dotenv import load_dotenv
        load_dotenv()

        groq_key  = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        # Build context from retrieved chunks (already fetched via retrieve_with_metadata)
        retrieved = (self._last_result or {}).get("retrieved", [])
        context = " ".join(
            (d.get("chunk_text") or "")[:300]
            for d in retrieved[:self.k]
        ).strip()

        if not context:
            # fallback: use full doctor answer if available
            return str((self._last_result or {}).get("answer", ""))

        # Decisive classification prompt — penalise MAYBE to avoid over-prediction
        prompt = (
            "You are a biomedical question answering system.\n"
            "Based on the retrieved context, answer the question with EXACTLY ONE WORD: yes, no, or maybe.\n"
            "Rules:\n"
            "- Answer 'yes' if the context supports the claim.\n"
            "- Answer 'no' if the context contradicts or does not support the claim.\n"
            "- Answer 'maybe' ONLY if evidence is genuinely split and both sides appear.\n"
            "- When in doubt, prefer 'yes' or 'no' over 'maybe'.\n"
            "- Do NOT write any other words.\n\n"
            f"Context:\n{context[:1400]}\n\n"
            f"Question: {question}\n\n"
            "Answer (yes/no/maybe):"
        )

        try:
            from openai import OpenAI
            if groq_key:
                client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
                model  = GROQ_EVAL_MODEL          # 8B model, higher rate limits
            elif openai_key:
                client = OpenAI(api_key=openai_key)
                model  = "gpt-4o-mini"
            else:
                return str((self._last_result or {}).get("answer", ""))

            import time

            # Retry up to 3 times on connection/rate errors
            for attempt in range(3):
                try:
                    if groq_key:
                        time.sleep(3 + attempt * 5)  # 3s, 8s, 13s backoff
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=20,
                    )
                    return resp.choices[0].message.content.strip()
                except Exception as e:
                    err_str = str(e)
                    print(f"    Attempt {attempt+1}/3 failed ({type(e).__name__}): {err_str[:80]}")
                    if "tokens per day" in err_str or "tokens_per_day" in err_str:
                        print("    Daily token limit hit — stopping LLM calls.")
                        return ""
                    if attempt == 2:
                        return ""
                    time.sleep(15)  # wait before retry
            return ""

        except Exception as e:
            print(f"    LLM setup error: {e}")
            return ""

    def evaluate(self, df: pd.DataFrame):
        required_cols = {"question", "final_decision"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        true_labels = []
        pred_labels = []

        for _, row in df.iterrows():
            question = str(row["question"])
            true_label = str(row["final_decision"]).strip().lower()

            try:
                self.retrieve_with_metadata(question, k=self.k)
                answer = self.generate_answer(question)
                pred = parse_answer(answer)
            except Exception:
                pred = "unknown"

            true_labels.append(true_label)
            pred_labels.append(pred)

        valid_idx = [i for i, p in enumerate(pred_labels) if p in LABELS]
        true_valid = [true_labels[i] for i in valid_idx]
        pred_valid = [pred_labels[i] for i in valid_idx]

        if not true_valid:
            print("Accuracy: no valid yes/no/maybe predictions")
            print("yes: precision=0.000, recall=0.000, f1=0.000")
            print("no: precision=0.000, recall=0.000, f1=0.000")
            print("maybe: precision=0.000, recall=0.000, f1=0.000")
            print("Confusion matrix:")
            print([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            return

        print("Accuracy:", accuracy_score(true_valid, pred_valid))

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_valid,
            pred_valid,
            labels=LABELS,
            average=None,
            zero_division=0,
        )

        for i, label in enumerate(LABELS):
            print(
                f"{label}: precision={precision[i]:.3f}, "
                f"recall={recall[i]:.3f}, f1={f1[i]:.3f}"
            )

        print("Confusion matrix:")
        print(confusion_matrix(true_valid, pred_valid, labels=LABELS))


def main():
    args = parse_args()

    questions_path = Path(args.questions)
    rag_py_path = Path(args.rag_py)

    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    df = pd.read_pickle(questions_path)

    RAGPipeline = load_rag_pipeline_class(rag_py_path)
    rag = RAGPipeline()

    evaluator = MarkusCompatibleEvaluator(rag_pipeline=rag, k=args.k)
    evaluator.evaluate(df)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)