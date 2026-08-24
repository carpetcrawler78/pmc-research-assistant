"""
02_enrich_cluster_labels.py
Capstone 1 — PMC Research Assistant

STAGE 2: Semantic Clustering — enrich DataFrame with cluster labels.

Reads clean_articles.pkl, optionally imputes keywords for articles with
too few keywords using the trained baseline model, then builds a keyword
co-occurrence matrix, applies Spectral Clustering, and writes
cluster_label / cluster_name / cluster_confidence back into an enriched pickle.

Keyword imputation (optional):
  Articles with fewer than MIN_KWDS_FOR_IMPUTATION clean keywords have their
  kwd_lst_cl supplemented from the baseline TF-IDF model predictions BEFORE
  clustering. The original kwd_lst_cl is preserved; imputed keywords are stored
  separately in kwd_imputed. This improves cluster coverage for articles that
  would otherwise receive cluster_label = -1.

Input:  big_data/baseline/pickle/clean_articles.pkl
        models/baseline_v1_clf.pkl   (only if IMPUTE_MISSING_KWDS = True)
Output: big_data/baseline/pickle/clean_articles_enriched.pkl

Usage:
    python python/02_enrich_cluster_labels.py
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
PICKLE_IN    = PROJECT_ROOT / "big_data" / "baseline" / "pickle" / "clean_articles.pkl"
PICKLE_OUT   = PROJECT_ROOT / "big_data" / "baseline" / "pickle" / "clean_articles_enriched.pkl"
MODEL_PATH   = PROJECT_ROOT / "models" / "baseline_v1_clf.pkl"

# ── Config ────────────────────────────────────────────────────────────────────
N_CLUSTERS  = 10
MIN_FREQ    = 10   # keywords that appear in fewer articles are dropped
RANDOM_SEED = 42

# ── Keyword imputation ────────────────────────────────────────────────────────
# Articles with fewer than this many clean keywords get TF-IDF predictions
# added to kwd_lst_cl before clustering (stored separately in kwd_imputed).
# Set to 0 to disable imputation entirely.
IMPUTE_MISSING_KWDS    = True
MIN_KWDS_FOR_IMPUTATION = 3   # impute if len(kwd_lst_cl) < this value
TOP_K_IMPUTE           = 8    # how many predicted keywords to add

# Optional manual overrides for cluster names.
# If empty ({}), names are always auto-generated from top keywords.
# If an entry exists for a cluster_id, it takes priority over auto-naming.
# NOTE: after re-clustering, IDs may shift — verify with inspect output before adding overrides.
CLUSTER_NAMES: dict[int, str] = {}


def load_data(path: Path) -> pd.DataFrame:
    print(f"Loading {path.name} ...")
    df = pickle.load(open(path, "rb"))
    print(f"  {len(df)} articles loaded")
    return df


def impute_keywords(
    df: pd.DataFrame,
    model_path: Path,
    min_kwds: int,
    top_k: int,
) -> pd.DataFrame:
    """
    For articles with fewer than `min_kwds` clean keywords, predict additional
    keywords using the trained TF-IDF baseline model and append them to
    kwd_lst_cl (for clustering only). The original list is preserved in
    kwd_lst_cl_orig; imputed keywords are stored in kwd_imputed.

    Articles with enough keywords are untouched (kwd_imputed = []).
    """
    if not model_path.exists():
        print(f"  ⚠  Baseline model not found at {model_path} — skipping imputation.")
        df["kwd_imputed"] = [[] for _ in range(len(df))]
        return df

    print(f"Loading baseline model for keyword imputation ...")
    bundle = pickle.load(open(model_path, "rb"))
    tfidf  = bundle["tfidf"]
    svd    = bundle["svd"]
    clf    = bundle["clf"]
    mlb    = bundle["mlb"]

    # Identify articles that need imputation
    needs = df["kwd_lst_cl"].apply(lambda k: len(k) < min_kwds)
    n_impute = needs.sum()
    print(f"  {n_impute} / {len(df)} articles have < {min_kwds} keywords → imputing")

    if n_impute == 0:
        df["kwd_imputed"] = [[] for _ in range(len(df))]
        return df

    # Build input text (same as training)
    text = (
        df["title"].fillna("") + " " +
        df["abstract"].fillna("") + " " +
        df["introduction"].fillna("")
    )

    sub_text = text[needs].tolist()
    X     = tfidf.transform(sub_text)
    X_lsa = svd.transform(X)

    proba   = clf.predict_proba(X_lsa)
    top_idx = np.argsort(proba, axis=1)[:, -top_k:][:, ::-1]
    predicted = [mlb.classes_[idx].tolist() for idx in top_idx]

    # Store imputed keywords and extend kwd_lst_cl for clustering
    df["kwd_lst_cl_orig"] = df["kwd_lst_cl"].copy()
    kwd_imputed = [[]] * len(df)
    kwd_lst_cl  = df["kwd_lst_cl"].tolist()

    pred_iter = iter(predicted)
    for i, (idx, row_needs) in enumerate(zip(df.index, needs)):
        if row_needs:
            preds = next(pred_iter)
            existing = set(kwd_lst_cl[i])
            new_kwds = [p for p in preds if p not in existing]
            kwd_imputed[i] = new_kwds
            kwd_lst_cl[i]  = kwd_lst_cl[i] + new_kwds

    df["kwd_lst_cl"] = kwd_lst_cl
    df["kwd_imputed"] = kwd_imputed

    imputed_counts = [len(x) for x in kwd_imputed if x]
    print(f"  ✓ Avg keywords added per imputed article: {np.mean(imputed_counts):.1f}")
    return df


def build_cooccurrence(df: pd.DataFrame, min_freq: int):
    """Build keyword co-occurrence matrix with log1p scaling."""
    print("Building co-occurrence matrix ...")

    # Count keyword frequency across all articles
    kw_freq: dict = defaultdict(int)
    for kws in df["kwd_lst_cl"].dropna():
        for kw in kws:
            kw_freq[kw] += 1

    # Keep only keywords above min_freq
    vocab = sorted(k for k, v in kw_freq.items() if v >= min_freq)
    kw2idx = {kw: i for i, kw in enumerate(vocab)}
    V = len(vocab)
    print(f"  {V} keywords after min_freq={min_freq} filter")

    # Build sparse co-occurrence matrix
    mat = lil_matrix((V, V), dtype=np.float32)
    for kws in df["kwd_lst_cl"].dropna():
        idxs = [kw2idx[kw] for kw in kws if kw in kw2idx]
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                mat[idxs[i], idxs[j]] += 1
                mat[idxs[j], idxs[i]] += 1

    # Log1p scaling to dampen high-frequency pairs
    adj = csr_matrix(mat)
    adj.data = np.log1p(adj.data)
    print(f"  Matrix: {V}×{V}, sparsity: {1 - adj.nnz / (V * V):.1%}")
    return adj, vocab, kw2idx


def cluster_keywords(adj, n_clusters: int, seed: int):
    """Apply Spectral Clustering on the co-occurrence adjacency matrix."""
    print(f"Running Spectral Clustering (n_clusters={n_clusters}) ...")
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed,
        n_init=10,
    )
    labels = sc.fit_predict(adj.toarray())
    print(f"  Cluster sizes: { {i: int((labels==i).sum()) for i in range(n_clusters)} }")
    return labels


def auto_name_clusters(df: pd.DataFrame, vocab: list, kw_labels: np.ndarray, top_n: int = 3) -> dict[int, str]:
    """
    Auto-generate cluster names from the most frequent keywords per cluster.
    Result: {cluster_id: "keyword1 / keyword2 / keyword3"}
    Used as fallback when CLUSTER_NAMES has no entry for a given cluster_id.
    """
    n_clusters = int(kw_labels.max()) + 1
    # Map each vocab keyword to its cluster
    kw_to_cluster = {vocab[i]: int(kw_labels[i]) for i in range(len(vocab))}

    cluster_counters: dict[int, defaultdict] = {i: defaultdict(int) for i in range(n_clusters)}
    for kws in df["kwd_lst_cl"].dropna():
        for kw in kws:
            if kw in kw_to_cluster:
                cluster_counters[kw_to_cluster[kw]][kw] += 1

    names = {}
    for cluster_id, counter in cluster_counters.items():
        # Take more candidates than needed so we have room to drop prefixes
        candidates = sorted(counter, key=counter.get, reverse=True)[:top_n * 3]
        # Prefix deduplication: drop kw_a if it has ≥ 3 words AND
        # another keyword kw_b starts with kw_a (kw_a is a prefix of kw_b)
        filtered = []
        candidates_set = set(candidates)
        for kw in candidates:
            words = kw.split()
            if len(words) >= 3:
                is_prefix = any(
                    other != kw and other.startswith(kw + " ")
                    for other in candidates_set
                )
                if is_prefix:
                    continue   # drop — a more specific term covers it
            filtered.append(kw)
            if len(filtered) == top_n:
                break
        names[cluster_id] = " / ".join(filtered) if filtered else f"Cluster {cluster_id}"
    return names


def assign_article_clusters(df: pd.DataFrame, kw2idx: dict, kw_labels: np.ndarray, auto_names: dict):
    """
    For each article, assign the dominant cluster of its keywords.
    Returns cluster_label (int), cluster_name (str), cluster_confidence (float).
    """
    article_labels, article_names, article_conf = [], [], []

    for kws in df["kwd_lst_cl"]:
        if not kws:
            article_labels.append(-1)
            article_names.append("Unknown")
            article_conf.append(0.0)
            continue

        matched = [kw_labels[kw2idx[kw]] for kw in kws if kw in kw2idx]
        if not matched:
            article_labels.append(-1)
            article_names.append("Unknown")
            article_conf.append(0.0)
            continue

        counts = defaultdict(int)
        for lbl in matched:
            counts[lbl] += 1

        dominant = max(counts, key=counts.get)
        confidence = counts[dominant] / len(matched)

        # Manual override takes priority, else use auto-generated name
        name = CLUSTER_NAMES.get(dominant) or auto_names.get(dominant, f"Cluster {dominant}")
        article_labels.append(int(dominant))
        article_names.append(name)
        article_conf.append(round(confidence, 3))

    df["cluster_label"]      = article_labels
    df["cluster_name"]       = article_names
    df["cluster_confidence"] = article_conf
    return df


def main():
    df = load_data(PICKLE_IN)

    # Optional: impute keywords for articles with too few before clustering
    if IMPUTE_MISSING_KWDS:
        print("\n── Keyword imputation ──")
        df = impute_keywords(df, MODEL_PATH, MIN_KWDS_FOR_IMPUTATION, TOP_K_IMPUTE)
    else:
        df["kwd_imputed"] = [[] for _ in range(len(df))]

    adj, vocab, kw2idx = build_cooccurrence(df, MIN_FREQ)
    kw_labels = cluster_keywords(adj, N_CLUSTERS, RANDOM_SEED)

    auto_names = auto_name_clusters(df, vocab, kw_labels)
    print("Auto-generated cluster names:")
    for cid, name in sorted(auto_names.items()):
        override = CLUSTER_NAMES.get(cid)
        tag = f"  → override: '{override}'" if override else ""
        print(f"  [{cid}] {name}{tag}")

    print("Assigning cluster labels to articles ...")
    df = assign_article_clusters(df, kw2idx, kw_labels, auto_names)

    # Restore original kwd_lst_cl (imputation was only for clustering)
    if "kwd_lst_cl_orig" in df.columns:
        df["kwd_lst_cl"] = df["kwd_lst_cl_orig"]
        df.drop(columns=["kwd_lst_cl_orig"], inplace=True)
        print("  ✓ kwd_lst_cl restored to original (imputed kwds in kwd_imputed)")

    # Quick sanity check
    print("\n── Cluster distribution ──")
    print(df["cluster_name"].value_counts().to_string())
    print(f"\nConfidence mean: {df['cluster_confidence'].mean():.3f}")
    print(f"Articles with no cluster match: {(df['cluster_label'] == -1).sum()}")

    # Save
    PICKLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PICKLE_OUT, "wb") as f:
        pickle.dump(df, f)
    print(f"\n✓ Saved enriched DataFrame → {PICKLE_OUT}")
    print(f"  Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()
