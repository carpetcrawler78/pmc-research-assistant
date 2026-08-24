"""
Generate comparison plots for PubMedQA evaluation results.
Run: python plots/evaluation/pubmedqa_comparison.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Measured results ──────────────────────────────────────────────────────────
systems = ["Thomas\nRAG Retrieval\n(eval.py)", "Thomas\nGold Context\n(eval_gold.py)", "Markus\nflan-t5-large\n(supervised)"]
accuracy = [0.3333, 0.6812, 0.8261]

# Per-class F1
yes_f1   = [0.467, 0.800, 0.884]
no_f1    = [0.293, 0.650, 0.844]
maybe_f1 = [0.162, 0.000, 0.000]

# ── Colour palette ─────────────────────────────────────────────────────────────
C_RAG    = "#4C72B0"  # blue
C_GOLD   = "#55A868"  # green
C_MARKUS = "#C44E52"  # red

colors = [C_RAG, C_GOLD, C_MARKUS]

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("PubMedQA Evaluation: Plan-A RAG vs Markus Baseline",
             fontsize=14, fontweight="bold", y=1.01)

# ── Plot 1: Accuracy comparison with gap annotations ──────────────────────────
ax1 = axes[0]
bars = ax1.bar(range(3), accuracy, color=colors, width=0.55, zorder=3,
               edgecolor="white", linewidth=1.2)
ax1.set_ylim(0, 1.05)
ax1.set_ylabel("Accuracy", fontsize=11)
ax1.set_title("Overall Accuracy", fontsize=12, fontweight="bold")
ax1.set_xticks(range(3))
ax1.set_xticklabels(systems, fontsize=9)
ax1.yaxis.grid(True, alpha=0.3, zorder=0)
ax1.set_axisbelow(True)

for bar, acc in zip(bars, accuracy):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{acc:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

# Gap arrows
def annotate_gap(ax, x1, x2, y, gap, color="#333333"):
    ax.annotate("", xy=(x2 - 0.27, y), xytext=(x1 + 0.27, y),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.5))
    ax.text((x1 + x2) / 2, y + 0.015, f"+{gap:.1%}", ha="center",
            va="bottom", fontsize=9, color=color, fontweight="bold")

annotate_gap(ax1, 0, 1, 0.72, 0.3479, "#2E86AB")   # retrieval gap
annotate_gap(ax1, 1, 2, 0.88, 0.1449, "#E84855")   # supervised gap

ax1.text(0.5, 0.67, "retrieval\nbottleneck", ha="center", fontsize=8,
         color="#2E86AB", style="italic")
ax1.text(1.5, 0.83, "supervised\ngap", ha="center", fontsize=8,
         color="#E84855", style="italic")

# ── Plot 2: Per-class F1 grouped bar ──────────────────────────────────────────
ax2 = axes[1]
x = np.arange(3)
w = 0.25
r1 = ax2.bar(x - w, yes_f1,   w, label="YES",   color="#2ECC71", zorder=3)
r2 = ax2.bar(x,     no_f1,    w, label="NO",    color="#E74C3C", zorder=3)
r3 = ax2.bar(x + w, maybe_f1, w, label="MAYBE", color="#F39C12", zorder=3)

ax2.set_ylim(0, 1.0)
ax2.set_ylabel("F1 Score", fontsize=11)
ax2.set_title("Per-Class F1", fontsize=12, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(["RAG\nRetrieval", "Gold\nContext", "Markus\nflan-t5"], fontsize=9)
ax2.yaxis.grid(True, alpha=0.3, zorder=0)
ax2.set_axisbelow(True)
ax2.legend(fontsize=9, loc="upper left")

for bars in [r1, r2, r3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.02:
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                     f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)

# ── Plot 3: Confusion matrix (Gold Context run) ────────────────────────────────
ax3 = axes[2]
labels = ["yes", "no", "maybe"]
cm = np.array([[34, 1, 3],
               [8, 13, 3],
               [5, 2, 0]])

im = ax3.imshow(cm, cmap="Blues", vmin=0, vmax=38)
ax3.set_title("Gold-Context Confusion Matrix\n(Thomas, 8B zero-shot)", fontsize=11, fontweight="bold")
ax3.set_xticks(range(3))
ax3.set_yticks(range(3))
ax3.set_xticklabels([f"pred\n{l}" for l in labels], fontsize=9)
ax3.set_yticklabels([f"actual\n{l}" for l in labels], fontsize=9)
ax3.set_xlabel("Predicted", fontsize=10)
ax3.set_ylabel("Actual", fontsize=10)

for i in range(3):
    for j in range(3):
        val = cm[i, j]
        color = "white" if val > 18 else "black"
        ax3.text(j, i, str(val), ha="center", va="center",
                 fontsize=14, fontweight="bold", color=color)

plt.colorbar(im, ax=ax3, shrink=0.8)

# ── Footer note ────────────────────────────────────────────────────────────────
fig.text(0.5, -0.04,
         "Dataset: PubMedQA golden69 (YES=38, NO=24, MAYBE=7)  |  "
         "RAG: S-PubMedBERT + BM25 Hybrid, 46,853 chunks  |  "
         "LLM: llama-3.1-8b-instant (zero-shot)  |  Markus: fine-tuned flan-t5-large",
         ha="center", fontsize=8, color="#555555", style="italic")

plt.tight_layout()
outpath = "plots/evaluation/pubmedqa_comparison.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Saved: {outpath}")
