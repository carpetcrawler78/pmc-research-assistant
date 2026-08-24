import faiss
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")  # wichtig!


# === PATHS ===
FAISS_PATH = "big_data/faiss_v2/pubmedbert.index"
META_PATH  = "big_data/faiss_v2/pubmedbert_meta.pkl"

# === LOAD ===
index = faiss.read_index(FAISS_PATH)

with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

print(f"Loaded {len(meta)} meta entries")

# === RECONSTRUCT EMBEDDINGS ===
print("Reconstructing embeddings...")
vecs = np.vstack([index.reconstruct(i) for i in range(index.ntotal)])

# === BUILD DATAFRAME ===
df = pd.DataFrame(meta)
df["embedding"] = list(vecs)

# === GROUP BY CLUSTER ===
clusters = df["cluster_name"].unique()

centroids = []
sizes = []
labels = []

for cl in clusters:
    sub = df[df["cluster_name"] == cl]
    emb = np.vstack(sub["embedding"].values)

    centroid = emb.mean(axis=0)

    centroids.append(centroid)
    sizes.append(len(sub))
    labels.append(cl)

centroids = np.vstack(centroids)

# === DIMENSION REDUCTION ===
pca = PCA(n_components=2)
coords = pca.fit_transform(centroids)

# === PLOT ===
plt.figure(figsize=(10, 8))

sizes_scaled = np.array(sizes) / max(sizes) * 2000  # scale bubbles

plt.scatter(coords[:, 0], coords[:, 1], s=sizes_scaled, alpha=0.6)

for i, label in enumerate(labels):
    plt.text(coords[i, 0], coords[i, 1], label, fontsize=9)

plt.title("Biomedical Topic Cloud (based on embedding centroids)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)

plt.savefig("cluster_cloud.png", dpi=300, bbox_inches="tight")
print("Saved to cluster_cloud.png")