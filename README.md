# PMC Research Assistant

Capstone I team project by Markus, Thomas and Viktor.

The project works with scientific articles from the PubMed Central (PMC) Open Access
subset. It started with keyword prediction for papers and was extended to a retrieval
system with an LLM on top: given a paper title, it suggests keywords; given a
free-text question, it answers with citations to papers from the corpus.

## Data

- Source: PMC Open Access subset, group `oa_noncomm` (XML bulk packages from the NCBI FTP
  server). The download URLs are listed in [`data/urls.txt`](data/urls.txt):
  - `oa_noncomm_xml.PMC003xxxxxx.baseline.2026-01-23.tar.gz` (main data set)
  - `oa_noncomm_xml.incr.2026-03-04.tar.gz` (small set for first tests)
- Extraction: [`python/01_data_extraction.py`](python/01_data_extraction.py) parses the XML
  (title, authors, keywords, abstract and body sections). Only research articles with
  keywords and the main sections are kept; rejected files and the reason are logged in
  [`data/extraction_errors.csv`](data/extraction_errors.csv).
- Cleaning and EDA: [`notebooks/02_cleaning_eda.ipynb`](notebooks/02_cleaning_eda.ipynb).
  The cleaned set used for the index has 7,702 articles.
- XML structure analysis: [`tools/xml_structure_analysis.py`](tools/xml_structure_analysis.py),
  results in `data/xml_tag_frequencies.csv` and `data/xml_attr_frequencies.csv`.

### Licence of the data

The articles come from the PMC Open Access subset. According to PMC, licence terms vary
per article, and users are responsible for following the licence stated in each article.
The `oa_noncomm` group contains articles whose licences do not allow commercial use.

Because of these per-article licences, the extracted article texts are **not included**
in this repository. To create the data set yourself:

1. Download the archives listed in [`data/urls.txt`](data/urls.txt) into `big_data/`.
2. In [`python/01_data_extraction.py`](python/01_data_extraction.py) set `RUN_TYPE`
   (`"baseline"` or `"incr"`), `TESTRUN = False` and `SAVE_PICKLE = True`
   (optionally `SAVE_CSV = True`), then run `python python/01_data_extraction.py`.
   Output: `big_data/baseline/pickle/raw_articles.pkl`.
3. Run [`notebooks/02_cleaning_eda.ipynb`](notebooks/02_cleaning_eda.ipynb).
   Output: `big_data/baseline/pickle/clean_articles.pkl`.

All later steps (`python/02_*` to `python/05_*`, the notebooks `03`–`05`,
[`colab_pipeline.ipynb`](colab_pipeline.ipynb)) read these pickles or files built from
them under `big_data/`. No script or notebook in this repository reads a CSV copy of the
article texts.

## What was built

| Step | What | Files |
|---|---|---|
| Keyword prediction (baseline) | TF-IDF → LSA → LinearSVC (one-vs-rest), multi-label | [`notebooks/03_baseline_keyword_prediction.ipynb`](notebooks/03_baseline_keyword_prediction.ipynb) |
| Clustering | Keyword co-occurrence matrix + spectral clustering; cluster labels added to each article | [`python/02_enrich_cluster_labels.py`](python/02_enrich_cluster_labels.py), [`notebooks/04_clustering_semantic.ipynb`](notebooks/04_clustering_semantic.ipynb) |
| Keyword candidates | Top-k predictions of the baseline model stored per article | [`python/03_enrich_tfidf_candidates.py`](python/03_enrich_tfidf_candidates.py) |
| Index | Chunking of title + abstract + introduction (1,600 characters, 200 overlap), embeddings with `pritamdeka/S-PubMedBert-MS-MARCO`, FAISS (`IndexFlatIP`) and BM25 | [`python/03_build_index.py`](python/03_build_index.py), [`python/lib/`](python/lib/), [`colab_pipeline.ipynb`](colab_pipeline.ipynb) (GPU run on Colab) |
| Hybrid retrieval | FAISS + BM25, merged with Reciprocal Rank Fusion (k = 60) | [`python/lib/retriever.py`](python/lib/retriever.py), parameters in [`python/config.py`](python/config.py) |
| RAG pipeline | Keyword mode (keywords from retrieved papers, optional LLM refinement) and question mode ("doctor" mode, answer with citations). LLM via Groq (`llama-3.3-70b-versatile`) or OpenAI (`gpt-4o-mini`) | [`python/04_rag_pipeline.py`](python/04_rag_pipeline.py) |
| Evaluation | Keyword prediction, retrieval ablation, question mode | [`python/05_evaluation.py`](python/05_evaluation.py) |
| Answer comparison | 8 questions × 3 variants (BM25 + Llama 3.1 8B, hybrid + Llama 3.1 8B, hybrid + Llama 3.3 70B) | [`python/eval_v1_bm25only.py`](python/eval_v1_bm25only.py), [`python/eval_v2_hybrid_8b.py`](python/eval_v2_hybrid_8b.py), [`python/eval_v3_hybrid_70b.py`](python/eval_v3_hybrid_70b.py) |

Additional experiments: a RAG baseline evaluated on PubMedQA-style yes/no questions
([`notebooks/RAG_Models/`](notebooks/RAG_Models/), [`notebooks/Evaluation/Evaluation3.ipynb`](notebooks/Evaluation/Evaluation3.ipynb),
[`evaluation/eval.py`](evaluation/eval.py), [`evaluation/eval_gold.py`](evaluation/eval_gold.py)).
The question file used there (`golden69_questions.pkl`) is not part of the repository.

## Results

All numbers below are taken from the JSON files in [`evaluation/`](evaluation/), produced by
[`python/05_evaluation.py`](python/05_evaluation.py). Plots are in
[`plots/evaluation/`](plots/evaluation/).

**Keyword prediction** ([`evaluation/keyword_eval.json`](evaluation/keyword_eval.json)).
50 sampled articles. The title is used as the query, the article itself is excluded from
the retrieved papers, and 15 keywords are predicted and compared with the author keywords.

| Metric | Exact match | Soft match |
|---|---|---|
| Precision | 0.117 | 0.168 |
| Recall | 0.329 | 0.477 |
| F1 | 0.170 (± 0.116) | 0.245 (± 0.141) |

**Retrieval ablation** ([`evaluation/retrieval_eval.json`](evaluation/retrieval_eval.json)).
50 sampled articles, title as query. A retrieved article counts as relevant if it belongs
to the same keyword cluster as the query article.

| Retrieval | MRR@15 | P@5 | P@10 | nDCG@15 | Cluster precision@10 |
|---|---|---|---|---|---|
| Hybrid (RRF) | 0.567 | 0.440 | 0.474 | 0.411 | 0.476 |
| FAISS only | 0.582 | 0.468 | 0.418 | 0.347 | 0.462 |
| BM25 only | 0.605 | 0.468 | 0.400 | 0.348 | 0.440 |

Hybrid retrieval is best on P@10, nDCG@15 and cluster precision@10, but not on MRR@15
and P@5.

**Question mode** ([`evaluation/doctor_eval.json`](evaluation/doctor_eval.json)).
10 questions with expected topic clusters and keywords.

| Metric | Value |
|---|---|
| Cluster precision@10 | 0.81 |
| Keyword overlap | 0.70 |
| Citations per answer | 4.5 |
| Citation coverage | 0.90 |
| Answers with disclaimer | 0.80 |

**Answer comparison** ([`evaluation/custom_v1_bm25.json`](evaluation/custom_v1_bm25.json),
[`evaluation/custom_v2_hybrid_8b.json`](evaluation/custom_v2_hybrid_8b.json),
[`evaluation/custom_v3_hybrid_70b.json`](evaluation/custom_v3_hybrid_70b.json)).
The answers of all 24 runs are collected in
[`evaluation/answers_for_review.md`](evaluation/answers_for_review.md) for manual review.

### Known limitation

[`evaluation/session_summary_2026-03-22.md`](evaluation/session_summary_2026-03-22.md)
documents a bug in the title extraction: titles contained the reference list of the paper.
The extraction was fixed and the chunk texts in the index metadata were patched, but the
FAISS vectors were not recomputed at that point.

## Repository structure

| Path | Content |
|---|---|
| `python/` | Pipeline scripts (`01`–`05`), evaluation variants, `config.py`, `lib/` (chunker, embedder, indexer, retriever, plotter) |
| `notebooks/` | Extraction, cleaning/EDA, keyword baseline, clustering, FAISS index; `RAG_Models/` and `Evaluation/` for the PubMedQA experiments |
| `evaluation/` | Evaluation results (JSON), answers for review, evaluation scripts |
| `data/` | XML structure statistics, extraction log, download URLs |
| `plots/` | EDA and evaluation plots |
| `docs/`, `presentations/` | Pipeline diagrams and presentation material |
| `tools/` | XML structure analysis, sample extraction |
| `colab_pipeline.ipynb` | Index build and retrieval on Google Colab (GPU) |
| `create_cluster_cloud.py` | PCA plot of the cluster embedding centroids (`cluster_cloud.png`) |
| `archive/` | Earlier notebook and script versions, personal evaluation notebooks, working notes; kept for reference, not part of the pipeline |

Large intermediate files (`big_data/`, `models/`) are not part of the repository.

## Setup

Python 3.11 (see `.python-version`).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

The RAG pipeline reads `GROQ_API_KEY` or `OPENAI_API_KEY` (and optionally `HF_TOKEN`)
from a `.env` file, see [`python/config.py`](python/config.py).
