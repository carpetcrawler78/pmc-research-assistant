import json
import subprocess
from pathlib import Path
from collections import defaultdict
import re

# -----------------------------
# Config
# -----------------------------
BASE_REMOTE = "origin"
BASE_BRANCH = "thomas"      # remote comparison baseline
LOCAL_BRANCH = "thomas"     # your local branch
TARGET_BRANCHES = [
    "main",
    "viktor",
    "marlo-80",
    "gettingdata",
    "thomas",
]
OUTDIR = Path("branch_analysis")
OUTDIR.mkdir(exist_ok=True)

KEYWORDS = {
    "chunking": [
        "chunk", "sentenc", "split", "tokenize", "recursivecharactertextsplitter"
    ],
    "embedding": [
        "embedding", "embed", "specter", "bert", "biobert", "pubmedbert",
        "sentence-transformers", "transformers"
    ],
    "retrieval": [
        "faiss", "retriev", "index", "vectorstore", "nearest", "cosine", "bm25"
    ],
    "rag": [
        "rag", "context", "prompt", "llm", "generation", "answer", "qa"
    ],
    "evaluation": [
        "evaluat", "metric", "precision", "recall", "f1", "accuracy",
        "benchmark", "mrr", "map", "ndcg", "hit@", "exact match", "rouge"
    ],
    "data_ingestion": [
        "xml", "nxml", "pubmed", "pmc", "parse", "iterparse", "gzip", "tarfile"
    ],
    "keyword_modeling": [
        "keyword", "kwd", "cluster", "binarizer", "onevsrest", "tfidf"
    ],
    "plots": [
        "plot", "matplotlib", "plt.", "seaborn", "plotly"
    ],
}

# -----------------------------
# Helpers
# -----------------------------
def run(cmd):
    result = subprocess.run(
        cmd,
        shell=True,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace"
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()

def git(cmd):
    return run(f"git {cmd}")

def branch_ref(remote_branch):
    return f"{BASE_REMOTE}/{remote_branch}"

def safe_read_git_file(ref_and_path):
    code, out, err = git(f"show {ref_and_path}")
    if code != 0:
        return None
    return out

def list_files(ref):
    code, out, err = git(f'ls-tree -r --name-only {ref}')
    if code != 0:
        return []
    return [x for x in out.splitlines() if x.endswith(".py") or x.endswith(".ipynb")]

def diff_name_status(a, b):
    code, out, err = git(f'diff --name-status {a}..{b} -- "*.py" "*.ipynb"')
    if code != 0:
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            rows.append((parts[0], parts[1]))
    return rows

def file_exists(ref, path):
    code, out, err = git(f'cat-file -e {ref}:{path}')
    return code == 0

def get_file_text(ref, path):
    return safe_read_git_file(f"{ref}:{path}")

def recent_commits_for_branch(branch):
    code, out, err = git(f'log --oneline --decorate -n 25 {branch_ref(branch)} -- "*.py" "*.ipynb"')
    if code != 0:
        return []
    return out.splitlines()

def local_uncommitted_diff():
    code, out, err = git('diff -- "*.py" "*.ipynb"')
    return out if code == 0 else ""

def local_status():
    code, out, err = git("status --short")
    return out.splitlines() if code == 0 else []

def detect_signals(text):
    text_low = text.lower()
    found = defaultdict(list)
    for category, terms in KEYWORDS.items():
        hits = []
        for term in terms:
            if term in text_low:
                hits.append(term)
        if hits:
            found[category] = sorted(set(hits))
    return found

def summarize_workflow(files_content):
    agg = defaultdict(set)
    for path, text in files_content.items():
        sig = detect_signals(text)
        for k, vals in sig.items():
            agg[k].update(vals)

    ordered = []
    if agg["data_ingestion"]:
        ordered.append("data ingestion/parsing")
    if agg["chunking"]:
        ordered.append("chunking / sentence splitting")
    if agg["embedding"]:
        ordered.append("embedding generation")
    if agg["retrieval"]:
        ordered.append("retrieval / index search")
    if agg["rag"]:
        ordered.append("RAG / prompting / answer generation")
    if agg["keyword_modeling"]:
        ordered.append("keyword modeling / clustering")
    if agg["evaluation"]:
        ordered.append("evaluation / benchmarking")
    if agg["plots"]:
        ordered.append("plots / visualization")

    return ordered, {k: sorted(v) for k, v in agg.items() if v}

def classify_branch(branch, files, files_content):
    workflow, signals = summarize_workflow(files_content)

    notes = []
    if any("evaluation" in f.lower() for f in files) or "evaluation" in signals:
        notes.append("has explicit evaluation artifacts")
    if any("rag" in f.lower() for f in files) or "rag" in signals:
        notes.append("has explicit RAG-related files/code")
    if any("cluster" in f.lower() or "keyword" in f.lower() for f in files):
        notes.append("has keyword/clustering work")
    if any("xml" in f.lower() or "data" in f.lower() for f in files):
        notes.append("has data ingestion/preprocessing work")

    return {
        "branch": branch,
        "file_count": len(files),
        "workflow_steps": workflow,
        "signals": signals,
        "notes": notes,
    }

# -----------------------------
# Main analysis
# -----------------------------
def main():
    report = {
        "branches": {},
        "comparisons_to_origin_thomas": {},
        "local_vs_origin_thomas": {},
        "local_status": local_status(),
    }

    # Verify branches
    code, out, err = git("branch -r")
    if code != 0:
        raise SystemExit(f"Could not list remote branches:\n{err}")

    remote_branches = out.splitlines()
    remote_branches = [x.strip() for x in remote_branches]
    available = set()
    for line in remote_branches:
        if "->" in line:
            continue
        available.add(line)

    # Analyze each target remote branch
    for b in TARGET_BRANCHES:
        ref = branch_ref(b)
        if ref not in available:
            report["branches"][b] = {"error": f"{ref} not found"}
            continue

        files = list_files(ref)

        files_content = {}
        for path in files:
            text = get_file_text(ref, path)
            if text is not None:
                files_content[path] = text[:400000]

        branch_info = classify_branch(b, files, files_content)
        branch_info["files"] = files
        branch_info["recent_commits"] = recent_commits_for_branch(b)
        report["branches"][b] = branch_info

        # Diff against origin/thomas
        if b != BASE_BRANCH:
            report["comparisons_to_origin_thomas"][b] = {
                "name_status": diff_name_status(branch_ref(BASE_BRANCH), ref)
            }

    # local thomas vs origin/thomas
    code, out, err = git(f'diff --name-status {branch_ref(BASE_BRANCH)}..{LOCAL_BRANCH} -- "*.py" "*.ipynb"')
    report["local_vs_origin_thomas"]["name_status"] = out.splitlines() if code == 0 else []
    report["local_vs_origin_thomas"]["uncommitted_diff_exists"] = bool(local_uncommitted_diff().strip())

    # Write machine-readable report
    json_path = OUTDIR / "capstone_branch_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Write human summary
    lines = []
    lines.append("# Capstone branch analysis")
    lines.append("")

    lines.append("## 1. Branch-by-branch workflow summary")
    for b in TARGET_BRANCHES:
        info = report["branches"].get(b, {})
        lines.append(f"### {b}")
        if "error" in info:
            lines.append(f"- ERROR: {info['error']}")
            lines.append("")
            continue
        lines.append(f"- Files (.py/.ipynb): {info['file_count']}")
        lines.append(f"- Workflow: {', '.join(info['workflow_steps']) if info['workflow_steps'] else 'no clear workflow detected'}")
        lines.append(f"- Notes: {', '.join(info['notes']) if info['notes'] else 'none'}")

        sig = info.get("signals", {})
        if sig:
            lines.append("- Detected signals:")
            for k, vals in sig.items():
                lines.append(f"  - {k}: {', '.join(vals[:12])}")
        else:
            lines.append("- Detected signals: none")

        commits = info.get("recent_commits", [])
        if commits:
            lines.append("- Recent relevant commits:")
            for c in commits[:8]:
                lines.append(f"  - {c}")
        lines.append("")

    lines.append("## 2. Differences vs origin/thomas")
    for b, comp in report["comparisons_to_origin_thomas"].items():
        lines.append(f"### {b} vs origin/thomas")
        rows = comp.get("name_status", [])
        if not rows:
            lines.append("- No .py/.ipynb differences detected")
        else:
            for status, path in rows:
                lines.append(f"- {status}: {path}")
        lines.append("")

    lines.append("## 3. Local thomas vs origin/thomas")
    ns = report["local_vs_origin_thomas"].get("name_status", [])
    if ns:
        for row in ns:
            lines.append(f"- {row}")
    else:
        lines.append("- No committed file differences between local thomas and origin/thomas in .py/.ipynb")
    if report["local_vs_origin_thomas"]["uncommitted_diff_exists"]:
        lines.append("- There are also uncommitted local changes in .py/.ipynb")
    else:
        lines.append("- No uncommitted local .py/.ipynb diff detected")
    lines.append("")

    lines.append("## 4. High-level interpretation template")
    lines.append("- gettingdata: likely data acquisition / parsing / preprocessing branch if XML/PubMed parsing dominates")
    lines.append("- main: likely integrated or most consolidated shared pipeline")
    lines.append("- viktor: inspect for alternative RAG pipeline / sentence chunking / evaluation logic")
    lines.append("- marlo-80: inspect for owner-specific experiments / baseline notebooks")
    lines.append("- thomas: your current comparison baseline")
    lines.append("")

    md_path = OUTDIR / "capstone_branch_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Done.\nJSON report: {json_path}\nMarkdown summary: {md_path}")

if __name__ == "__main__":
    main()