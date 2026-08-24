# PMC Scientific Papers — Keyword Prediction Pipeline

```mermaid
flowchart TD
  RAW["raw_articles.pkl"]
  CLEAN["clean_articles.pkl"]
  MODEL["baseline_v1_clf.pkl"]
  ENRICHED["clean_articles_enriched.pkl"]
  FI["specter.index"]
  FM["specter_meta.pkl"]
  OUT[/"Keyword Predictions"/]

  subgraph G1[" "]
    direction LR
    A1["XML / tar.gz parsing<br/>10K sample flag<br/>pandas, pathlib, xml.etree"] --- S1["1 Extraction"]
  end

  subgraph G2[" "]
    direction LR
    A2["semicolon and comma split<br/>expand_paren_abbrev<br/>IUPAC and junk filter<br/>bigram enrichment<br/>EDA distributions"] --- S2["2 Cleaning and EDA"]
  end

  subgraph G3[" "]
    direction LR
    A3["TfidfVectorizer<br/>TruncatedSVD (LSA)<br/>CalibratedClassifierCV (LinearSVC)<br/>MultiLabelBinarizer<br/>F1 threshold sweep"] --- S3["3 Baseline Model (train once)"]
  end

  subgraph G4[" "]
    direction LR
    A4["keyword imputation via baseline model<br/>co-occurrence matrix, log1p damping<br/>SpectralClustering on adjacency matrix<br/>auto cluster naming (top-N keywords)"] --- S4["4 Clustering"]
  end

  subgraph G5[" "]
    direction LR
    A5["allenai-specter: embed all articles (build)<br/>L2 normalisation<br/>FAISS IndexFlatIP: build and save index<br/>tfidf_candidates per article"] --- S5["5 Embedding and Index"]
  end

  subgraph G6[" "]
    direction LR
    A6["allenai-specter: embed query (load only)<br/>FAISS search: top-k retrieval, MIN_SCORE filter<br/>score-weighted keyword aggregation<br/>normalisation: apostrophe, trailing-s merge<br/>optional LLM refinement (OpenAI API)"] --- S6["6 RAG Pipeline"]
  end

  S1 --> RAW --> S2 --> CLEAN
  CLEAN --> S3 --> MODEL
  CLEAN --> S4
  MODEL --> S4
  S4 --> ENRICHED --> S5
  MODEL --> S5
  S5 --> FI
  S5 --> FM
  FI --> S6
  FM --> S6
  S6 --> OUT

  style G1 fill:none,stroke:none
  style G2 fill:none,stroke:none
  style G3 fill:none,stroke:none
  style G4 fill:none,stroke:none
  style G5 fill:none,stroke:none
  style G6 fill:none,stroke:none

  classDef nb   fill:#3b0764,stroke:#7c3aed,color:#e9d5ff
  classDef sc   fill:#0c4a6e,stroke:#0284c7,color:#bae6fd
  classDef art  fill:#064e3b,stroke:#059669,color:#a7f3d0,font-size:11px,padding:2px 6px
  classDef res  fill:#78350f,stroke:#d97706,color:#fde68a
  classDef note fill:#1e293b,stroke:#475569,color:#94a3b8

  class S2,S3 nb
  class S1,S4,S5,S6 sc
  class RAW,CLEAN,MODEL,ENRICHED,FI,FM art
  class OUT res
  class A1,A2,A3,A4,A5,A6 note
```

| Step | File | Purpose | Key Logic | Packages | Output |
|------|------|---------|-----------|----------|--------|
| 1 | `python/01_data_extraction.py` | Parse PMC XML archives or load 10K sample pickle into flat DataFrame | `USE_10K_SAMPLE` flag bypasses tar.gz extraction | pandas, pathlib, xml.etree | `raw_articles.pkl` — cols: pmcid, title, abstract, keywords, sections |
| 2 | `notebooks/02_cleaning_eda.ipynb` | Keyword cleaning pipeline + exploratory analysis | semicolon/comma split, `expand_paren_abbrev()`, `is_iupac_or_junk()`, `is_standalone_alnum_junk()`, bigram enrichment | pandas, re, unicodedata, matplotlib | `clean_articles.pkl` — cols: kwd_lst, kwd_lst_cl, n_keywords |
| 3 | `notebooks/03_baseline_keyword_prediction.ipynb` (train once) | Train multi-label keyword classifier — used as enhancer for clustering and similarity, not end-user output | TfidfVectorizer → TruncatedSVD (LSA) → CalibratedClassifierCV(LinearSVC) → MultiLabelBinarizer, F1 @ threshold sweep | sklearn | `baseline_v1_clf.pkl` — bundle: tfidf, svd, clf, mlb |
| 4 | `python/02_enrich_cluster_labels.py` | Assign semantic domain cluster per article via keyword co-occurrence | 1. Impute keywords for sparse articles via baseline model; 2. Sparse co-occurrence matrix, log1p damping; 3. SpectralClustering on adjacency; 4. Auto-name clusters from top-N keywords (prefix dedup for 3+ word terms) | scipy.sparse, sklearn SpectralClustering | `clean_articles_enriched.pkl` — cols: cluster_label, cluster_name, cluster_confidence, kwd_imputed |
| 5 | `python/03_enrich_tfidf_candidates.py` | Add baseline-predicted keyword candidates per article; build FAISS semantic index for similarity retrieval | Part A: baseline model → top-k tfidf_candidates per article; Part B: allenai-specter embeds "title [SEP] abstract" → L2-normalised → IndexFlatIP | sentence-transformers, faiss-cpu, tqdm | `clean_articles_enriched.pkl` + tfidf_candidates, `specter.index`, `specter_meta.pkl` |
| 6 | `python/04_rag_pipeline.py` | Core end-user feature: RAG-based keyword prediction for new papers via semantic retrieval | 1. Embed query (allenai-specter); 2. FAISS top-k, MIN_SCORE filter; 3. Score-weighted aggregation: kwd_lst_cl x2xscore, tfidf_candidates x1xscore, direct TF-IDF x1.5; 4. Keyword normalisation; 5. Optional LLM refinement (OpenAI API) | faiss-cpu, sentence-transformers, openai, python-dotenv | retrieval_keywords, direct_tfidf, llm_keywords (console / JSON) |
