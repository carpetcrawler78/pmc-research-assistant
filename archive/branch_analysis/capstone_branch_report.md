# Capstone branch analysis

## 1. Branch-by-branch workflow summary
### main
- Files (.py/.ipynb): 8
- Workflow: data ingestion/parsing, chunking / sentence splitting, embedding generation, retrieval / index search, RAG / prompting / answer generation, keyword modeling / clustering, evaluation / benchmarking, plots / visualization
- Notes: has explicit evaluation artifacts, has explicit RAG-related files/code, has data ingestion/preprocessing work
- Detected signals:
  - chunking: chunk, sentenc, split, tokenize
  - retrieval: index, retriev
  - rag: answer, context, llm, prompt, qa, rag
  - evaluation: accuracy, evaluat, f1, map, mrr, precision, recall
  - data_ingestion: gzip, nxml, parse, pmc, pubmed, tarfile, xml
  - keyword_modeling: keyword, kwd
  - plots: matplotlib, plot, plt.
  - embedding: bert, embed, embedding, sentence-transformers, transformers
- Recent relevant commits:
  - bcd53b6 (origin/main, origin/HEAD) Update evaluation.ipynb
  - 2dd20a5 First evaluation script
  - ce32bec RAG Baseline added
  - 0f0142c Upload Scripts
  - ec5f31f jounal filter modified
  - 23e917b Update 02_cleaning_EDA.ipynb
  - 357d37c add data_extraction.py and 02_cleaning_eda.ipynb
  - 562bd9c move old py files to /alt, add extraction_v1 and cleaning notebook

### viktor
- Files (.py/.ipynb): 23
- Workflow: data ingestion/parsing, chunking / sentence splitting, embedding generation, retrieval / index search, RAG / prompting / answer generation, keyword modeling / clustering, evaluation / benchmarking, plots / visualization
- Notes: has explicit evaluation artifacts, has explicit RAG-related files/code, has keyword/clustering work, has data ingestion/preprocessing work
- Detected signals:
  - chunking: sentenc, split
  - retrieval: bm25, cosine, faiss, index, nearest, retriev
  - rag: answer, context, generation, llm, prompt, qa, rag
  - evaluation: accuracy, benchmark, evaluat, f1, map, metric, mrr, ndcg, precision, recall
  - data_ingestion: gzip, nxml, parse, pmc, pubmed, tarfile, xml
  - keyword_modeling: binarizer, cluster, keyword, kwd, onevsrest, tfidf
  - plots: matplotlib, plot, plt., seaborn
  - embedding: bert, biobert, embed, embedding, specter, transformers
- Recent relevant commits:
  - 5484f6f (origin/viktor) undefined variables  removed
  - 4726abe (origin/gettingdata, gettingdata) bring selected notebooks and pipeline scripts from thomas into gettingdata
  - b4a8956 (myfork/gettingdata) add presentation + plot images
  - e7f7801 add FAISS index notebook
  - 52921dd Merge remote-tracking branch 'origin/main' into gettingdata
  - f3e7044 Merge remote-tracking branch 'origin/marlo-80' into gettingdata
  - a99b299 Merge remote-tracking branch 'origin/thomas' into gettingdata
  - c6e148b double nb ??

### marlo-80
- Files (.py/.ipynb): 8
- Workflow: data ingestion/parsing, chunking / sentence splitting, embedding generation, retrieval / index search, RAG / prompting / answer generation, keyword modeling / clustering, evaluation / benchmarking, plots / visualization
- Notes: has explicit evaluation artifacts, has explicit RAG-related files/code, has keyword/clustering work, has data ingestion/preprocessing work
- Detected signals:
  - chunking: chunk, sentenc, split, tokenize
  - retrieval: cosine, index, nearest, retriev
  - rag: answer, context, generation, llm, prompt, qa, rag
  - evaluation: accuracy, evaluat, f1, map, metric, mrr, precision
  - data_ingestion: nxml, parse, pmc, pubmed, tarfile, xml
  - keyword_modeling: binarizer, cluster, keyword, kwd, tfidf
  - embedding: bert, embed, embedding, sentence-transformers, transformers
  - plots: matplotlib, plot, seaborn
- Recent relevant commits:
  - 2870341 (origin/marlo-80) gitignore
  - 79be710 Cleanup
  - 17287fa Cleanup
  - 2c15ccb Cleanup
  - 33c3751 Imported Datset Folder
  - d212654 First Retrieval Benchmarks
  - 496b5dd parser1
  - 19c3f3f organizing

### gettingdata
- Files (.py/.ipynb): 23
- Workflow: data ingestion/parsing, chunking / sentence splitting, embedding generation, retrieval / index search, RAG / prompting / answer generation, keyword modeling / clustering, evaluation / benchmarking, plots / visualization
- Notes: has explicit evaluation artifacts, has explicit RAG-related files/code, has keyword/clustering work, has data ingestion/preprocessing work
- Detected signals:
  - chunking: sentenc, split
  - retrieval: bm25, cosine, faiss, index, nearest, retriev
  - rag: answer, context, generation, llm, prompt, qa, rag
  - evaluation: accuracy, benchmark, evaluat, f1, map, metric, mrr, ndcg, precision, recall
  - data_ingestion: gzip, nxml, parse, pmc, pubmed, tarfile, xml
  - keyword_modeling: binarizer, cluster, keyword, kwd, onevsrest, tfidf
  - plots: matplotlib, plot, plt., seaborn
  - embedding: bert, biobert, embed, embedding, specter, transformers
- Recent relevant commits:
  - 4726abe (origin/gettingdata, gettingdata) bring selected notebooks and pipeline scripts from thomas into gettingdata
  - b4a8956 (myfork/gettingdata) add presentation + plot images
  - e7f7801 add FAISS index notebook
  - 52921dd Merge remote-tracking branch 'origin/main' into gettingdata
  - f3e7044 Merge remote-tracking branch 'origin/marlo-80' into gettingdata
  - a99b299 Merge remote-tracking branch 'origin/thomas' into gettingdata
  - c6e148b double nb ??
  - 6438c23 more efficient sorted_keywords impl

### thomas
- Files (.py/.ipynb): 20
- Workflow: data ingestion/parsing, chunking / sentence splitting, embedding generation, retrieval / index search, RAG / prompting / answer generation, keyword modeling / clustering, evaluation / benchmarking, plots / visualization
- Notes: has explicit evaluation artifacts, has explicit RAG-related files/code, has keyword/clustering work, has data ingestion/preprocessing work
- Detected signals:
  - chunking: sentenc, split
  - retrieval: bm25, cosine, faiss, index, nearest, retriev
  - rag: answer, context, generation, llm, prompt, qa, rag
  - evaluation: accuracy, benchmark, evaluat, f1, map, metric, mrr, ndcg, precision, recall
  - data_ingestion: gzip, nxml, parse, pmc, pubmed, tarfile, xml
  - keyword_modeling: binarizer, cluster, keyword, kwd, onevsrest, tfidf
  - plots: matplotlib, plot, plt., seaborn
  - embedding: bert, biobert, embed, embedding, specter, transformers
- Recent relevant commits:
  - cf42cae feat: prefix deduplication in cluster auto-naming (3+ word terms only)
  - e289eff feat: auto-generate cluster names from top keywords
  - 4e95541 feat: improve RAG pipeline with score-weighting, min-score filter, direct TF-IDF, keyword normalisation
  - 3b78c18 feat: load HF_TOKEN from .env for all SentenceTransformer usages
  - 3a4c23b restructure project: add numbered pipeline, plots, data artifacts, and RAG module
  - 5c2307f update
  - 282f873 Clustering
  - 73a7879 pick 39e60a3 Update squash c775a45 fix exclude pkl squash e986c8a fix gitignore

## 2. Differences vs origin/thomas
### main vs origin/thomas
- A: Dataset/read_all_first_draft.ipynb
- R098: notebooks/01_data_extraction.ipynb	Dataset/read_all_optimized.ipynb
- A: notebooks/01_carpetcrawler78-data_extraction.ipynb
- A: notebooks/02_baseline.ipynb
- M: notebooks/02_cleaning_eda.ipynb
- A: notebooks/03_RAG_Baseline.ipynb
- D: notebooks/03_baseline_keyword_prediction.ipynb
- D: notebooks/04_clustering_semantic.ipynb
- D: notebooks/05_build_faiss_index.ipynb
- D: notebooks/baseline_II_v1.ipynb
- D: notebooks/baseline_v0.ipynb
- D: notebooks/cleaning_eda_v4.ipynb
- D: notebooks/cleaning_eda_v5.ipynb
- D: notebooks/cleaning_eda_v6.ipynb
- D: notebooks/clustering_v1.ipynb
- A: notebooks/evaluation.ipynb
- D: python/01_data_extraction.py
- D: python/02_enrich_cluster_labels.py
- D: python/03_enrich_tfidf_candidates.py
- D: python/04_rag_pipeline.py
- D: python/build_faiss_index.py
- R085: python/data_extraction_v2.py	python/data_extraction.py
- D: python/data_extraction_v3.py
- D: tools/extract_samples.py
- D: tools/xml_structure_analysis.py

### viktor vs origin/thomas
- A: Dataset/read_all_first_draft.ipynb
- A: Dataset/read_all_optimized.ipynb
- R080: notebooks/05_build_faiss_index.ipynb	notebooks/build_FAISS_index.ipynb
- M: notebooks/clustering_v1.ipynb
- A: notebooks/myStart.ipynb
- A: notebooks/read_all_first_draft.ipynb
- A: notebooks/read_all_optimized.ipynb
- M: python/02_enrich_cluster_labels.py
- D: python/03_enrich_tfidf_candidates.py
- D: python/04_rag_pipeline.py
- M: python/build_faiss_index.py

### marlo-80 vs origin/thomas
- A: Dataset/read_all_first_draft.ipynb
- R098: notebooks/01_data_extraction.ipynb	Dataset/read_all_optimized.ipynb
- A: FirstRag/LSA_embedding.ipynb
- A: FirstRag/RAG_Baseline.ipynb
- A: FirstRag/RAG_Baseline_v2.ipynb
- A: clustering_marlo-80.ipynb
- A: getting_data.ipynb
- A: myStart.ipynb
- D: notebooks/02_cleaning_eda.ipynb
- D: notebooks/03_baseline_keyword_prediction.ipynb
- D: notebooks/04_clustering_semantic.ipynb
- D: notebooks/05_build_faiss_index.ipynb
- D: notebooks/baseline_II_v1.ipynb
- D: notebooks/baseline_v0.ipynb
- D: notebooks/cleaning_eda_v4.ipynb
- D: notebooks/cleaning_eda_v5.ipynb
- D: notebooks/cleaning_eda_v6.ipynb
- D: notebooks/clustering_v1.ipynb
- D: python/01_data_extraction.py
- D: python/02_enrich_cluster_labels.py
- D: python/03_enrich_tfidf_candidates.py
- D: python/04_rag_pipeline.py
- D: python/build_faiss_index.py
- D: python/data_extraction_v2.py
- D: python/data_extraction_v3.py
- D: tools/extract_samples.py
- D: tools/xml_structure_analysis.py

### gettingdata vs origin/thomas
- A: Dataset/read_all_first_draft.ipynb
- A: Dataset/read_all_optimized.ipynb
- R080: notebooks/05_build_faiss_index.ipynb	notebooks/build_FAISS_index.ipynb
- M: notebooks/clustering_v1.ipynb
- A: notebooks/myStart.ipynb
- A: notebooks/read_all_first_draft.ipynb
- A: notebooks/read_all_optimized.ipynb
- M: python/02_enrich_cluster_labels.py
- D: python/03_enrich_tfidf_candidates.py
- D: python/04_rag_pipeline.py
- M: python/build_faiss_index.py

## 3. Local thomas vs origin/thomas
- A	python/03_build_index.py
- M	python/04_rag_pipeline.py
- A	python/config.py
- A	python/lib/__init__.py
- A	python/lib/chunker.py
- A	python/lib/embedder.py
- A	python/lib/indexer.py
- A	python/lib/plotter.py
- A	python/lib/retriever.py
- There are also uncommitted local changes in .py/.ipynb

## 4. High-level interpretation template
- gettingdata: likely data acquisition / parsing / preprocessing branch if XML/PubMed parsing dominates
- main: likely integrated or most consolidated shared pipeline
- viktor: inspect for alternative RAG pipeline / sentence chunking / evaluation logic
- marlo-80: inspect for owner-specific experiments / baseline notebooks
- thomas: your current comparison baseline
