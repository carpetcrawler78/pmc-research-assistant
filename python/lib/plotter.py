"""
lib/plotter.py
Capstone 1 — PMC Research Assistant  v2

Visualisation helpers for intermediate pipeline results.

Functions
---------
plot_chunk_stats(meta, out_dir)
    Chunk-count per article + chunk-length distribution (saved after indexing).

plot_embedding_2d(index, meta, out_dir, sample_n)
    t-SNE 2-D scatter of chunk embeddings coloured by cluster (saved after indexing).

plot_retrieval_scores(retrieved, title, out_dir)
    Horizontal bar chart of retrieved papers and their RRF scores (per query).

plot_keyword_scores(candidates, out_dir)
    Horizontal bar chart of aggregated keyword candidates and weights (per query).
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless — no display required
import matplotlib.pyplot as plt
import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Plot saved: {path.name}")


def _short(text: str, n: int = 45) -> str:
    return text[:n] + "…" if len(text) > n else text


# ── 1. Chunk statistics (after 03_build_index) ───────────────────────────────

def plot_chunk_stats(meta: list[dict], out_dir: Path) -> None:
    """
    Two-panel figure:
      Left  — histogram of chunks-per-article
      Right — histogram of chunk character lengths
    """
    # chunks per article
    article_chunk_counts: Counter = Counter()
    chunk_lengths: list[int] = []

    for m in meta:
        article_chunk_counts[m["pmcid"]] += 1
        chunk_lengths.append(len(m["chunk_text"]))

    counts = list(article_chunk_counts.values())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Chunk Statistics (after indexing)", fontweight="bold")

    # left: chunks per article
    axes[0].hist(counts, bins=range(1, max(counts) + 2), color="#4f86c6", edgecolor="white")
    axes[0].set_xlabel("Chunks per article")
    axes[0].set_ylabel("Number of articles")
    axes[0].set_title(f"Chunks / article  (mean={np.mean(counts):.1f})")
    axes[0].axvline(np.mean(counts), color="tomato", linestyle="--", label=f"mean {np.mean(counts):.1f}")
    axes[0].legend(fontsize=8)

    # right: chunk character lengths
    axes[1].hist(chunk_lengths, bins=40, color="#6dbf67", edgecolor="white")
    axes[1].set_xlabel("Chunk length (chars)")
    axes[1].set_ylabel("Number of chunks")
    axes[1].set_title(f"Chunk lengths  (mean={np.mean(chunk_lengths):.0f} chars)")
    axes[1].axvline(np.mean(chunk_lengths), color="tomato", linestyle="--",
                    label=f"mean {np.mean(chunk_lengths):.0f}")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    _save(fig, out_dir / "chunk_stats.png")


# ── 2. 2-D embedding scatter (after 03_build_index) ──────────────────────────

def plot_embedding_2d(
    index,
    meta:     list[dict],
    out_dir:  Path,
    sample_n: int = 800,
) -> None:
    """
    t-SNE scatter plot of chunk embeddings coloured by cluster_name.
    Uses a random sample of at most *sample_n* chunks for speed.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  sklearn not found — skipping embedding plot.")
        return

    n = min(sample_n, index.ntotal)
    rng = np.random.default_rng(42)
    idx_sample = rng.choice(index.ntotal, size=n, replace=False)

    # Reconstruct vectors from FAISS (IndexFlatIP supports reconstruct)
    vecs = np.vstack([index.reconstruct(int(i)) for i in idx_sample])
    labels = [meta[i].get("cluster_name", "unknown") for i in idx_sample]

    print(f"  Running t-SNE on {n} chunks …")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=500)
    coords = tsne.fit_transform(vecs)

    unique_clusters = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", len(unique_clusters))
    color_map = {c: cmap(i) for i, c in enumerate(unique_clusters)}
    colors = [color_map[l] for l in labels]

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=8, alpha=0.6, linewidths=0)
    ax.set_title(f"t-SNE of chunk embeddings  (n={n}, S-PubMedBERT)", fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.axis("off")

    # Legend (max 20 clusters)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map[c], markersize=7, label=_short(c))
        for c in unique_clusters[:20]
    ]
    ax.legend(handles=handles, fontsize=7, loc="best",
              framealpha=0.8, ncol=2, title="Cluster")

    _save(fig, out_dir / "embedding_tsne.png")


# ── 3. Retrieval scores per query (after 04_rag_pipeline) ────────────────────

def plot_retrieval_scores(
    retrieved: list[dict],
    query_title: str,
    out_dir: Path,
    fname: str = "retrieval_scores.png",
) -> None:
    """
    Horizontal bar chart of retrieved articles coloured by cluster,
    x-axis = RRF retrieval score.
    """
    if not retrieved:
        return

    titles  = [_short(d.get("title") or "?", 50) for d in retrieved]
    scores  = [d.get("retrieval_score", 0.0) for d in retrieved]
    clusters = [d.get("cluster_name", "?") for d in retrieved]

    unique_cl = sorted(set(clusters))
    cmap = plt.cm.get_cmap("tab10", max(len(unique_cl), 1))
    color_map = {c: cmap(i) for i, c in enumerate(unique_cl)}
    colors = [color_map[c] for c in clusters]

    fig, ax = plt.subplots(figsize=(11, max(4, len(titles) * 0.45)))
    y_pos = range(len(titles) - 1, -1, -1)   # top-ranked at top
    ax.barh(list(y_pos), scores[::-1], color=colors[::-1], edgecolor="white", height=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(titles[::-1], fontsize=8)
    ax.set_xlabel("RRF retrieval score")
    ax.set_title(f"Retrieved papers\nQuery: {_short(query_title, 65)}", fontweight="bold")

    handles = [
        matplotlib.patches.Patch(facecolor=color_map[c], label=_short(c, 30))
        for c in unique_cl
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower right", title="Cluster")
    plt.tight_layout()
    _save(fig, out_dir / fname)


# ── 4. Keyword candidate scores per query (after 04_rag_pipeline) ────────────

def plot_keyword_scores(
    candidates: list[tuple[str, float]],
    out_dir: Path,
    fname: str = "keyword_scores.png",
) -> None:
    """
    Horizontal bar chart of aggregated keyword candidates and their
    weighted scores.
    """
    if not candidates:
        return

    keywords = [kw for kw, _ in candidates]
    scores   = [sc for _, sc in candidates]

    fig, ax = plt.subplots(figsize=(9, max(3, len(keywords) * 0.4)))
    y_pos = range(len(keywords) - 1, -1, -1)
    bars = ax.barh(list(y_pos), scores[::-1], color="#4f86c6", edgecolor="white", height=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(keywords[::-1], fontsize=9)
    ax.set_xlabel("Aggregated keyword score")
    ax.set_title("Keyword candidates  (score-weighted aggregation)", fontweight="bold")

    # value labels on bars
    for bar, val in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + max(scores) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7, color="#333")

    plt.tight_layout()
    _save(fig, out_dir / fname)
